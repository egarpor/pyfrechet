from joblib import Parallel, delayed
import sys, os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
from pyfrechet.metric_spaces import MetricData, Sphere, Spheroid, sphere_to_spheroid, spheroid_to_sphere
from pyfrechet.metrics import mse
from sklearn.model_selection import train_test_split
import itertools
from pyfrechet.regression.bagged_regressor import BaggedRegressor
from pyfrechet.regression.d_trees import d_Tree
import time

def canonical_lattice(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a canonical lattice on the unit sphere.
    
    Parameters:
    - n: int, number of points to generate

    Returns:
    - sphere_points: array of shape (n, 3) with coordinates on sphere
    """
    goldenRatio = (1 + 5**0.5)/2
    i = np.arange(0, n)
    theta = 2 * np.pi * i / goldenRatio
    phi = np.arccos(1 - 2*(i+0.5)/n)
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.vstack((x, y, z)).T

def sphere_area_pred_ball(M_spheroid, radius, spheroid_centers, sphere_points=None, n_points=5000, n_jobs=12):
    """
    Estimate areas of prediction balls for multiple reference points after mapping to sphere.
    
    Parameters:
    - M_spheroid: Spheroid metric space
    - radius: float, radius of the balls
    - spheroid_centers: array of shape (n_refs, 3), centers of balls on spheroid
    - sphere_points: optional, precomputed lattice points on sphere
    - n_points: int, number of lattice points if sphere_points not provided
    - n_jobs: int, number of parallel jobs
    
    Returns:
    - areas: array of areas for each reference point
    """
    # Generate or use provided lattice points
    if sphere_points is None:
        sphere_points = canonical_lattice(n_points)
    
    # Map lattice points to spheroid (do this once)
    spheroid_points = sphere_to_spheroid(sphere_points, M_spheroid.a, M_spheroid.c)
    
    # Function to process one reference point
    def process_ref_point(center_spheroid):
        # Map reference point to spheroid
        # ref_spheroid = sphere_to_spheroid(ref_point.reshape(-1,3), 
        #                                 M_spheroid.a, M_spheroid.c).squeeze()
        
        # Count points inside ball
        inside_ball = np.sum(M_spheroid.d(spheroid_points, center_spheroid) < radius)
        
        # Calculate area
        return 4 * np.pi * inside_ball / len(sphere_points)
    
    # Process all reference points in parallel
    areas = Parallel(n_jobs=n_jobs)(
        delayed(process_ref_point)(center) 
        for center in spheroid_centers
    )
    
    return np.array(areas)

# Suppress logging warnings from geomstats
np.random.seed(1000)
sign_level = np.array([0.01, 0.05, 0.1])

def evaluate_spheroid_params(X_train, y_train, X_test, y_test, param_grid, seed=5):
    """
    Evaluate different parameter combinations on test set
    """
    grid = list(itertools.product(param_grid['min_split_size'],
                                param_grid['mtry'],
                                param_grid['a_c']))
    
    best_area = float('inf')
    best_params = None
    all_results = []  # Store results for all configurations
    
    for idx, (min_split_size, mtry, (a, c)) in enumerate(grid):
        y_train_spheroid = sphere_to_spheroid(y_train, a, c)
        M = Spheroid(a=a, c=c)
        y_tr = MetricData(M, y_train_spheroid.reshape(-1, 3))
        
        structure = [(Sphere(dim=2), list(range(3)))]
        base = d_Tree(split_type='2means', impurity_method='medoid', structure=structure,
                    min_split_size=min_split_size, mtry=mtry)
        forest = BaggedRegressor(estimator=base, n_estimators=200,
                               bootstrap_fraction=1, bootstrap_replace=True,
                               seed=seed, n_jobs=12)
        
        forest.fit(X_train, y_tr)
        preds = forest.predict(X_test)
        
        # Get OOB quantile (90th percentile for 0.1 significance)
        oob_quantile = np.percentile(forest.oob_errors(), 90, method='inverted_cdf')
        can_lat = canonical_lattice(5000)

        # Calculate areas for each test point in parallel
        test_point_areas = sphere_area_pred_ball(M, oob_quantile, spheroid_centers = preds.data, sphere_points = can_lat, n_jobs=12)
        
        # Calculate mean area
        mean_area = np.mean(test_point_areas)
        
        # Calculate Type II coverage (proportion of test points within ball)
        test_points_spheroid = sphere_to_spheroid(y_test, a, c)
        
        distances = M.d(MetricData(M, test_points_spheroid), preds)
        type_II_coverage = np.mean(distances <= oob_quantile)
        
        # Also calculate MSE on sphere
        sphere_preds = spheroid_to_sphere(preds.data, a, c, R=1)
        test_mse = mse(MetricData(Sphere(2), y_test), MetricData(Sphere(2), sphere_preds))
        
        result = {
            'a': a,
            'c': c,
            'mean_area': mean_area,
            'test_point_areas': np.array(test_point_areas),
            'type_II_coverage': type_II_coverage,
            'oob_quantile': oob_quantile,
            'test_mse': test_mse
        }
        
        # Save individual result immediately
        all_results.append(result)
        
        print(f"Params: a={a}, c={c}")
        print(f"Mean Area: {mean_area:.6f}, Type II Coverage: {type_II_coverage:.6f}, MSE: {test_mse:.6f}")
        
        if mean_area < best_area:
            best_area = mean_area
            best_params = result
        
        # Clear memory
        del forest, preds, sphere_preds
        
    return best_params, all_results

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('sunspots/results', exist_ok=True)
    
    # Process all cycles from 12 to 23
    for cyc in range(12, 24):
        print(f'\n{"="*50}')
        print(f'Processing cycle {cyc}...')
        print(f'{"="*50}\n')
        
        filename = f'sunspots_births_{cyc}_deaths.csv'
        filepath = os.path.join(os.getcwd(), 'sunspots/data', filename)
        
        # Check if file exists
        if not os.path.exists(filepath):
            print(f'Warning: File {filename} not found, skipping cycle {cyc}')
            continue
            
        try:
            sample = pd.read_csv(filepath)
            X = np.vstack([sample['births_X.1'], sample['births_X.2'], sample['births_X.3']]).T
            y = np.vstack([sample['deaths_X.1'], sample['deaths_X.2'], sample['deaths_X.3']]).T
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1000)

            param_grid = {
                'min_split_size': [1],
                'mtry': [1],
                'a_c': [(0.1,1), (0.2,1), (0.3,1), (0.4,1), (0.5,1), (0.6,1), (0.7,1), (0.8,1), (0.9,1), (1,1), (1.1,1), (1.25,1), (1.5,1)]
            }
            
            start = time.time()
            best_result, all_results = evaluate_spheroid_params(X_train, y_train, X_test, y_test, param_grid)
            end = time.time()
            elapsed_minutes = (end - start) / 60
            print(f"\nParameter evaluation took {elapsed_minutes:.1f} minutes")

            print("\nBest parameters:")
            print(f"a: {best_result['a']}")
            print(f"c: {best_result['c']}")
            print(f"Mean Area: {best_result['mean_area']:.6f}")
            print(f"Type II Coverage: {best_result['type_II_coverage']:.6f}")
            print(f"MSE: {best_result['test_mse']:.6f}")

            # Save best result
            output_path = f'sunspots/results/area_spheroid_results_cycle_{cyc}.npy'
            np.save(output_path, best_result)
            print("Best results saved to", output_path)

            # Save all results
            all_results_path = f'sunspots/results/area_spheroid_all_results_cycle_{cyc}.npy'
            np.save(all_results_path, all_results)
            print("All results saved to", all_results_path)
                    
        except Exception as e:
            print(f'Error processing cycle {cyc}: {str(e)}')
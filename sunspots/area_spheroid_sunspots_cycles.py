import sys, os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
from pyfrechet.metric_spaces import MetricData, Sphere, Spheroid, sphere_to_spheroid, spheroid_to_sphere
from pyfrechet.metrics import mse
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from pyfrechet.regression.bagged_regressor import BaggedRegressor
from pyfrechet.regression.d_trees import d_Tree
import time
from sklearn.model_selection import KFold
import itertools

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
    radius = np.array(radius).reshape(-1,1)
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
        distances = M_spheroid.d(spheroid_points, center_spheroid)
        areas = []
        for r in radius:
            inside_ball = np.sum(distances < r)
            area = 4 * np.pi * inside_ball / len(sphere_points)
            areas.append(area)
        
        # Calculate area
        return np.array(areas)
    
    # Process all reference points in parallel
    areas = Parallel(n_jobs=n_jobs)(
        delayed(process_ref_point)(center) 
        for center in spheroid_centers
    )
    return np.array(areas)

# Suppress logging warnings from geomstats
np.random.seed(1000)
sign_level = np.array([0.01, 0.05, 0.1])

def custom_spheroid_GCV(X_train, y_train, param_grid, seed=5, n_splits=5):
    """
    Manual Grid Search CV for Fréchet forests on spheroid manifold.
    Uses mean prediction ball area as criterion.
    
    Parameters:
    - X_train: array-like, shape (n_samples, n_features)
    - y_train: array-like, shape (n_samples, 3)
    - param_grid: dict with keys 'min_split_size', 'mtry', 'a_c'
    - seed: int, random seed for reproducibility
    - n_splits: int, number of CV folds

    Returns:
    - final_forest: fitted BaggedRegressor on full training data with best parameters
    - best: dict, best hyperparameters and results
    - cv_results: list of all parameter combinations and their results
    """
    grid = list(itertools.product(param_grid['min_split_size'],
                                param_grid['mtry'],
                                param_grid['a_c']))

    cv_results = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Generate lattice points once for efficiency
    sphere_points = canonical_lattice(5000)

    for min_split_size, mtry, (a, c) in grid:
        fold_results = []
        y_train_spheroid = sphere_to_spheroid(y_train, a, c)

        for train_index, val_index in kf.split(X_train):
            X_tr, X_val = X_train[train_index], X_train[val_index]
            # y_tr_raw, y_val_raw = y_train_spheroid[train_index], y_train_spheroid[val_index]
            y_tr_raw = y_train_spheroid[train_index]
            # Define metric and structure for responses
            M = Spheroid(a=a, c=c)
            y_tr = MetricData(M, y_tr_raw.reshape(-1, 3))
            # y_val = MetricData(M, y_val_raw.reshape(-1, 3))

            # Use Sphere(2) for predictors since they remain on unit sphere
            structure = [(Sphere(dim=2), list(range(3)))]

            # Define forest
            base = d_Tree(split_type='2means', impurity_method='medoid', structure=structure,
                        min_split_size=min_split_size, mtry=mtry)
            forest = BaggedRegressor(estimator=base, n_estimators=200,
                                   bootstrap_fraction=1, bootstrap_replace=True,
                                   seed=seed, n_jobs=12)

            forest.fit(X_tr, y_tr)

            # Get predictions and OOB quantile
            preds = forest.predict(X_val)
            oob_quantile = np.percentile(forest.oob_errors(), 0.9 * 100, method='inverted_cdf')
            
            # Calculate areas
            areas = sphere_area_pred_ball(M, oob_quantile, spheroid_centers = preds.data, sphere_points = sphere_points)
            # Store results for this fold
            fold_results.append({
                'mean_area': np.mean(areas),
                'test_point_areas': areas,
                'oob_quantile': oob_quantile
            })

        # Average results across folds
        mean_area = np.mean([f['mean_area'] for f in fold_results])
        
        cv_results.append({
            'a': a,
            'c': c,
            'min_split_size': min_split_size,
            'mtry': mtry,
            'cv_mean_area': mean_area,
            #'fold_results': fold_results
        })
        print(f"Params: a={a}, c={c}, mean area={mean_area:.6f}")

    # Select best parameters based on mean area
    best = min(cv_results, key=lambda x: x['cv_mean_area'])
    print("\nBest parameters:")
    print(f"a: {best['a']}")
    print(f"c: {best['c']}")
    print(f"mean area: {best['cv_mean_area']:.6f}")

    # Final refit on full training data with best parameters
    best_M = Spheroid(a=best['a'], c=best['c'])
    y_train_spheroid = sphere_to_spheroid(y_train, best['a'], best['c'])
    y_train_metric = MetricData(best_M, y_train_spheroid.reshape(-1, 3))
    structure = [(Sphere(dim=2), list(range(3)))]
    final_tree = d_Tree(split_type='2means', impurity_method='medoid',
                      structure=structure,
                      min_split_size=best['min_split_size'], mtry=best['mtry'])
    final_forest = BaggedRegressor(estimator=final_tree, n_estimators=200,
                                   bootstrap_fraction=1, bootstrap_replace=True,
                                   seed=seed, n_jobs=12)
    final_forest.fit(X_train, y_train_metric)

    return final_forest, best

def task(cyc):
    filename = f'sunspots_births_{cyc}_deaths.csv'
    print(f'Processing {filename}...')
    filepath = os.path.join(os.getcwd(), 'sunspots/data', filename)

    sample = pd.read_csv(filepath)
    X = np.vstack([sample['births_X.1'], sample['births_X.2'], sample['births_X.3']]).T
    y = np.vstack([sample['deaths_X.1'], sample['deaths_X.2'], sample['deaths_X.3']]).T
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1000)

    # Define parameter grid for spheroid tuning
    param_grid = {
        'min_split_size': [1],
        'mtry': [1],
        'a_c': [(0.2,1), (0.4,1), (0.6,1), (0.8,1), (1,1)]}

    # Fit forest with spheroid metric
    start = time.time()
    spheroid_forest, best_params = custom_spheroid_GCV(X_train, y_train, param_grid)
    end = time.time()
    elapsed_minutes = (end - start) / 60
    print(f"Spheroid tuning for cycle {cyc} took {elapsed_minutes:.1f} minutes")

    # Get best spheroid parameters
    best_a = best_params['a']
    best_c = best_params['c']
    M = Spheroid(a=best_a, c=best_c)

    # Calculate coverage and predictions
    spheroid_oob_quantile = np.percentile(spheroid_forest.oob_errors(), (1 - np.array([0.01, 0.05, 0.1])) * 100, method='inverted_cdf')
    spheroid_preds = spheroid_forest.predict(X_test)
    spheroid_y_test = sphere_to_spheroid(y_test, best_a, best_c)
    spheroid_pb_cov = np.sum(M.d(MetricData(M, spheroid_y_test), spheroid_preds).reshape(X_test.shape[0],1) <= np.tile(spheroid_oob_quantile, X_test.shape[0]).reshape(-1,3), axis = 0) / X_test.shape[0]

    results = {}
    
    # Predictions
    results['preds_spheroid'] = spheroid_preds.data

    # Save parameters
    results['spheroid_best_params'] = best_params
    
    # Save MSE
    # assert that y_test is equak to spheroid_to_sphere(y_test, best_a, best_c) and print the result
    sphere_preds = spheroid_to_sphere(spheroid_preds.data, best_a, best_c, R=1)
    assert np.allclose(y_test, spheroid_to_sphere(spheroid_y_test, best_a, best_c, R=1)), "y_test is not equal to spheroid_to_sphere(y_test, best_a, best_c)"
    results['mse_spheroid'] = mse(MetricData(Sphere(2), y_test), sphere_preds)  

    # Save OOB quantile
    results['oob_quantile_spheroid'] = spheroid_oob_quantile

    # Save prediction ball coverage
    results['pb_cov_spheroid'] = spheroid_pb_cov

    # Save ball areas using prediction points
    can_lat = canonical_lattice(5000)
    results['area_spheroid'] = sphere_area_pred_ball(M, spheroid_oob_quantile, spheroid_centers = spheroid_preds.data, sphere_points = can_lat, n_jobs=12)

    # Save results
    output_path = f'sunspots/results/cv_spheroid_results_cycle_{cyc}.npy'
    np.save(output_path, results)
    print("Results saved to", output_path)

if __name__ == "__main__":
    
    blocks = [
        #[21],
        [22],
        [23],
        [17, 12],
        [16, 13],
        [15, 14],
        [20],
        [19],
        [18]
    ]

    def process_block(block):
        for cyc in block:
            task(cyc)

    for block in blocks:
        print(f"Processing block: {block}")
        process_block(block)
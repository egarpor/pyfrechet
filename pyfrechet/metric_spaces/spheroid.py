import numpy as np
from .metric_space import MetricSpace
from geopy.distance import geodesic
from .py_linz_geod import Ellipsoid
#from pygeodesy import EcefKarney, Ellipsoid


#class Spheroid(MetricSpace):
class Spheroid(MetricSpace):
    def __init__(self, a, c):
        """
        Initialize a Spheroid metric space.
        
        Parameters:
        - a: equatorial radius (default: 1.0)
        - c: polar radius (default: 0.5)
        """
        # super().__init__(GeomstatsSpheroid(a=a, c=c))
        self.a = a
        self.c = c
        self.f = (a - c) / a
        self.rf = 1 / self.f if self.f != 0 else np.inf

        self.extrinsic_dim = 3  # Embedding in R^3

    #def reverse_single(self, x, y, z):
    #    ecef = EcefKarney(a_ellipsoid=self.a, f=self.f)
    #    latlon = ecef.reverse(x, y, z).latlon
    #    return latlon

    def _d(self, x, y):
        """
        Compute geodesic distances between points on the spheroid.
        """
        x = x.reshape(-1, 3)
        y = y.reshape(-1, 3)

        # Vectorized wrapper
        #vectorized_reverse = np.vectorize(self.reverse_single)

        # Example batch input: (N, 3) array

        # Convert Cartesian coordinates (x, y, z) to geodetic coordinates (lat, lon, height)
        #_ = vectorized_reverse(x[:, 0], x[:, 1], x[:, 2])
        #deg1 = np.vstack([_[0], _[1]]).T
        #_ = vectorized_reverse(y[:, 0], y[:, 1], y[:, 2])
        #deg2 = np.vstack([_[0], _[1]]).T

        ell = Ellipsoid(a=self.a, rf=self.rf)

        deg1 = ell.geodetic(x)[:,[1,0,2]]
        deg2 = ell.geodetic(y)[:,[1,0,2]]

        #deg1 = cartesian_spheroid_to_geographic(x, self.a, self.c)
        #deg2 = cartesian_spheroid_to_geographic(y, self.a, self.c)

        # Handle different input cases
        if x.ndim == 1 and y.ndim == 1:
            # Single point to single point
            return geodesic(deg1, deg2, ellipsoid=(self.a, self.c, self.f)).km
        
        elif x.ndim == 1:
            # Single point to multiple points
            return np.array([geodesic(deg1, p, ellipsoid=(self.a, self.c, self.f)).km 
                            for p in deg2])
        
        elif (y.ndim == 1):
            # Multiple points to single point
            return np.array([geodesic(p, deg2, ellipsoid=(self.a, self.c, self.f)).km 
                            for p in deg1])
        
        # Multiple points to multiple points (pairwise)
        elif len(x) == len(y):
            return np.array([geodesic(p1, p2, ellipsoid=(self.a, self.c, self.f)).km 
                            for p1, p2 in zip(deg1, deg2)]).squeeze()
        
        # All pairs (n x m) if lengths don't match
        elif x.ndim == 2 and y.ndim == 2 and x.shape[0] != y.shape[0] and (x.shape[0] ==1 or y.shape[0] == 1):
            return np.array([[geodesic(p1, p2, ellipsoid=(self.a, self.c, self.f)).km 
                            for p2 in deg2] for p1 in deg1]).squeeze()
        else:
            raise ValueError("Invalid input shapes: x and y must have the same shape or be broadcastable.") 


    def _frechet_mean(self, y, w=None):
        """
        Compute the projected extrinsic mean on a spheroid.
        
        Parameters:
        - y: array of shape (n_points, 3) in Cartesian coordinates
        - w: weights array of shape (n_points,)
        
        Returns:
        - Projected mean on spheroid (normalized by spheroid metric)
        """
        if w is None:
            w = np.ones(len(y)) / len(y)
        
        # Compute weighted Euclidean mean in ambient space
        extrinsic_mean = np.sum(y * w[:, np.newaxis], axis=0)
        
        # Project onto spheroid using correct normalization
        x, y, z = extrinsic_mean
        a = self.a  # equatorial radius
        c = self.c  # polar radius
        
        # Spheroid normalization factor
        norm = np.sqrt((x**2 + y**2)/a**2 + z**2/c**2)
        
        # Handle near-zero case
        if norm < 1e-10:
            return np.array([a, 0, 0])  # arbitrary point on equator
        
        return extrinsic_mean / norm


    # def _frechet_mean(self, y, w=None):
    #     """
    #     Compute the Fréchet mean of points on the spheroid using geomstats optimization.
    #     
    #     Parameters:
    #     - y: array of points
    #     - w: weights (optional)
    #     
    #     Returns:
    #     - the Fréchet mean
    #     """
    #     # Get initial guess via extrinsic mean
    #     extrinsic_mean = w.dot(y)
    #     initial_point = extrinsic_mean / np.linalg.norm(extrinsic_mean)
    #     
    #     # Use geomstats FrechetMean to find the Riemannian mean
    #     mean = FrechetMean(
    #         metric=self.manifold.metric, 
    #         init_point=initial_point,
    #         verbose=False, 
    #         point_type='vector'
    #     )
    #     mean.fit(y, weights=w)
    #     
    #     return mean.estimate_
    
    def __str__(self):
        return f'Spheroid(a={self.a}, c={self.c})'


def cartesian_sphere_to_angles(xyz):
    """
    Convert Cartesian coordinates (sphere or spheroid) to spherical angles (lon, lat).
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r_xy = np.sqrt(x**2 + y**2)
    lat = np.arctan2(r_xy, z)
    lon = np.arctan2(y, x)
    return np.stack([lon, lat], axis=-1)

def cartesian_spheroid_to_angles(xyz, a, c):
    """
    Convert Cartesian to angles (lon,lat) accounting for spheroid geometry.

    Parameters:
    - xyz: Cartesian coordinates
    - a: equatorial radius
    - c: polar radius
    
    Returns:
    - (lon, lat) where lat is polar angle from z-axis, lon is azimuthal angle
    """
    xyz = np.asarray(xyz)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    lon = np.arctan2(y, x)
    r_xy = np.sqrt(x**2 + y**2)
    lat = np.arctan2(r_xy/a, z/c)  # Matches the parametric definition
    return np.stack([lon, lat], axis=-1)

def cartesian_spheroid_to_geographic(xyz, a, c):
    """
    Convert Cartesian coordinates (x,y,z) to geographic (lat, lon) in degrees.
    """
    xyz = np.asarray(xyz)
    if xyz.shape[-1] != 3:
        raise ValueError("Input must have shape (..., 3)")
    
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    # Longitude 
    lon = np.degrees(np.arctan2(y, x))  # -180 to 180°
    # Latitude
    r_xy = np.sqrt(x**2 + y**2)
    theta = np.arctan2(r_xy/a, z/c)  # Polar angle from z-axis
    lat = 90 - np.degrees(theta)  # -90 to 90°

    return np.stack([lat, lon], axis=-1)

def angles_to_spheroid(tp, a, c):
    """Convert parametric angles (θ, φ) to spheroid coordinates."""
    theta, phi = tp[..., 0], tp[..., 1]
    x = a * np.sin(phi) * np.cos(theta)
    y = a * np.sin(phi) * np.sin(theta)
    z = c * np.cos(phi)
    return np.stack([x, y, z], axis=-1)

def angles_to_sphere(tp, R=1.0):
    """Convert angles (θ, φ) to spherical Cartesian coordinates."""
    return angles_to_spheroid(tp, a=R, c=R)  # Sphere is a special case

def sphere_to_spheroid(xyz, a, c):
    """Convert spherical angles (θ, φ) to spheroidal angles."""
    angles = cartesian_sphere_to_angles(xyz)
    return angles_to_spheroid(angles, a=a, c=c)  # Convert to spheroidal coordinates

def spheroid_to_sphere(xyz, a, c, R):
    """Convert spheroidal angles (θ, φ) to spherical angles."""
    angles = cartesian_spheroid_to_angles(xyz, a=a, c=c)
    return angles_to_sphere(angles, R=R)  # Convert to spherical coordinates
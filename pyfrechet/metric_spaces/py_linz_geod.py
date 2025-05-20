# /usr/bin/python3
# This code is from `ython-linz-geodetic (it seems that the project is not pip installable)
# https://github.com/linz/python-linz-geodetic/blob/master/LINZ/Geodetic/Ellipsoid.py

import math
import numpy as np

class Ellipsoid:

    convergence = 1.0e-10

    @staticmethod
    def _cossin(angle):
        angle = np.radians(angle)
        return np.cos(angle), np.sin(angle)

    @staticmethod
    def enu_axes(lon, lat):
        """
        Returns an array defining the east, north, and up unit vectors
        at a specified latitude and longitude

        To convert an xyz offset to an enu offset, use as an example

           enu_axes=GRS80.enu_axes(lon,lat)
           denu=enu_axes.dot(dxyz)
           dxyz=enu_axes.T.dot(denu)
        """
        cln, sln = Ellipsoid._cossin(lon)
        clt, slt = Ellipsoid._cossin(lat)
        ve = np.array([-sln, cln, 0])
        vn = np.array([-cln * slt, -sln * slt, clt])
        vu = np.array([clt * cln, clt * sln, slt])
        return np.vstack((ve, vn, vu))

    def __init__(self, a, rf):
        """
        Initiallize an ellipsoid based on semi major axis and inverse flattening
        """
        self._setParams(a, rf)

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, a):
        self._setParams(a, self._rf)

    @property
    def rf(self):
        return self._rf

    @rf.setter
    def rf(self, rf):
        self._setParams(self._a, rf)

    @property
    def b(self):
        return self._b

    def _setParams(self, a, rf):
        self._a = float(a)
        self._rf = float(rf)
        self._b = a - a / rf if rf else a
        self._a2 = a * a
        self._b2 = self._b * self._b
        self._a2b2 = self._a2 - self._b2

    def xyz(self, lon, lat=None, hgt=None):
        """
        Calculate the geocentric X,Y,Z coordinates at a longitude
        and latitude

        Input is one of
           lon, lat        Single values or lists of values
           lon, lat, hgt
           llh             Array of [lon,lat,hgt]
        """
        single = True
        if lat is None:
            if not isinstance(lon, np.ndarray):
                lon = np.array(lon)
            single = len(lon.shape) == 1
            if single:
                lon = lon.reshape((1, lon.size))
            lat = lon[:, 1]
            hgt = lon[:, 2] if lon.shape[1] > 2 else 0
            lon = lon[:, 0]
        if hgt is None:
            hgt = 0

        cln, sln = Ellipsoid._cossin(lon)
        clt, slt = Ellipsoid._cossin(lat)
        bsac = np.hypot(self._b * slt, self._a * clt)
        p = self._a2 * clt / bsac + hgt * clt

        xyz = [p * cln, p * sln, self._b2 * slt / bsac + hgt * slt]
        xyz = np.vstack(xyz).transpose()
        if single:
            xyz = xyz[0]
        return xyz

    def metres_per_degree(self, lon, lat, hgt=0):
        """
        Calculate the number of metres per degree east and
        north
        """
        clt, slt = Ellipsoid._cossin(lat)
        bsac = np.hypot(self._b * slt, self._a * clt)
        p = self._a2 * clt / bsac + hgt * clt
        dedln = np.radians(p)
        dndlt = np.radians((self._a2 * self._b2) / (bsac * bsac * bsac) + hgt)
        return dedln, dndlt

    def geodetic(self, xyz):
        """
        Calculate the longitude, latitude, and height corresponding
        to a geocentric XYZ coordinate

        Input is one of
           xyz             Single [x,y,z]
           xyz             Array of [x,y,z]
        """
        if not isinstance(xyz, np.ndarray):
            xyz = np.array(xyz)
        single = len(xyz.shape) == 1
        if single:
            xyz = xyz.reshape((1, xyz.size))
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        ln = np.arctan2(y, x)
        p = np.hypot(x, y)
        lt = np.arctan2(self._a2 * z, self._b2 * p)
        for i in range(10):
            lt0 = lt
            slt = np.sin(lt)
            clt = np.cos(lt)
            bsac = np.hypot(self._b * slt, self._a * clt)
            lt = np.arctan2(z + slt * self._a2b2 / bsac, p)
            if np.all(abs(lt - lt0) < self.convergence):
                break
        h = p * clt + z * slt - bsac
        result = np.degrees(ln), np.degrees(lt), h
        result = np.vstack(result).transpose()
        return result[0] if single else result

    def radii_of_curvature(self, latitude):
        """
        Returns the radii of curvature along the meridional and prime
        vertical normal sections.
        """
        clt, slt = Ellipsoid._cossin(latitude)
        den = math.sqrt(self._a2 * clt * clt + self._b2 * slt * slt)
        rm = self._a2 * self._b2 / (den * den * den)
        rn = self._a2 / den
        return rm, rn
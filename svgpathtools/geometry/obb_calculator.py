"""
Oriented Bounding Box (OBB) Calculation Module

This module provides three algorithms for computing the Minimum Area Oriented 
Bounding Box for 2D paths composed of Lines and Bezier curves:

1. PCA Heuristic: Fast approximation using Principal Component Analysis
2. Numerical Optimization: Exhaustive search via area minimization
3. Discretization + Rotating Calipers: Exact solution via convex hull

Mathematical foundations and references:
- PCA: Classic technique for finding principal axes of point clouds
- Numerical Optimization: Direct minimization of bounding box area
- Rotating Calipers: Freeman & Shapira (1975) - optimal for convex polygons
"""
from __future__ import annotations
from typing import Literal, Optional, Tuple, List, Union
from dataclasses import dataclass
import warnings
import numpy as np
from scipy.spatial import ConvexHull
try:
    from scipy.optimize import minimize
    HAS_SCIPY_OPTIMIZE = True
except ImportError:
    HAS_SCIPY_OPTIMIZE = False

def _get_path_classes():
    from svgpathtools.path import Path, Line, QuadraticBezier, CubicBezier
    return Path, Line, QuadraticBezier, CubicBezier

@dataclass
class OBB:
    center: complex
    bu: complex
    bv: complex
    hu: float
    hv: float
    @property
    def area(self) -> float:
        return 4 * self.hu * self.hv
    @property
    def corners(self) -> List[complex]:
        corners = []
        for u in [-1, 1]:
            for v in [-1, 1]:
                corner = self.center + u * self.hu * self.bu + v * self.hv * self.bv
                corners.append(corner)
        return corners
    def __repr__(self) -> str:
        return (f"OBB(center={self.center}, bu={self.bu}, bv={self.bv}, "
                f"hu={self.hu}, hv={self.hv})")

def _sample_path_points(path, num_points: int = 1000, method: str = 'arc_length') -> np.ndarray:
    Path, Line, QuadraticBezier, CubicBezier = _get_path_classes()
    if len(path) == 0:
        raise ValueError("Cannot sample from an empty path")
    if method == 'arc_length':
        path_length = path.length()
        if path_length < 1e-10:
            point = path.start
            return np.array([[point.real, point.imag]], dtype=np.float64)
        distances = np.linspace(0, path_length, num_points)
        points = []
        for dist in distances:
            pt = path.point(dist / path_length)
            points.append([pt.real, pt.imag])
        return np.array(points, dtype=np.float64)
    else:
        t_values = np.linspace(0, 1, num_points)
        points = []
        for t in t_values:
            pt = path.point(t)
            points.append([pt.real, pt.imag])
        return np.array(points, dtype=np.float64)

def _compute_obb_from_points(points: np.ndarray, rotation_angle: float) -> Tuple[float, OBB]:
    if len(points) == 0:
        raise ValueError("Cannot compute OBB for empty point set")
    cos_a = np.cos(rotation_angle)
    sin_a = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    rotated_points = points @ rotation_matrix.T
    x_min, x_max = rotated_points[:, 0].min(), rotated_points[:, 0].max()
    y_min, y_max = rotated_points[:, 1].min(), rotated_points[:, 1].max()
    width = x_max - x_min
    height = y_max - y_min
    area = width * height
    center_rotated = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
    center_original = center_rotated @ rotation_matrix
    bu = complex(cos_a, -sin_a)
    bv = complex(sin_a, cos_a)
    obb = OBB(
        center=complex(center_original[0], center_original[1]),
        bu=bu,
        bv=bv,
        hu=width / 2,
        hv=height / 2
    )
    return area, obb

def obb_pca_heuristic(path, num_samples: int = 1000) -> OBB:
    Path, Line, QuadraticBezier, CubicBezier = _get_path_classes()
    if not path or len(path) == 0:
        raise ValueError("Cannot compute OBB for empty path")
    points = _sample_path_points(path, num_samples, method='arc_length')
    if len(points) < 2:
        pt = points[0]
        return OBB(center=complex(pt[0], pt[1]), bu=complex(1, 0), bv=complex(0, 1), hu=0, hv=0)
    mean = points.mean(axis=0)
    centered = points - mean
    cov_matrix = (centered.T @ centered) / len(points)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(-eigenvalues)
    eigenvectors = eigenvectors[:, idx]
    principal_axis = eigenvectors[:, 0]
    rotation_angle = np.arctan2(principal_axis[1], principal_axis[0])
    _, obb = _compute_obb_from_points(points, rotation_angle)
    return obb

def obb_numerical_optimization(path, num_samples: int = 1000, num_angles: int = 180) -> OBB:
    Path, Line, QuadraticBezier, CubicBezier = _get_path_classes()
    if not path or len(path) == 0:
        raise ValueError("Cannot compute OBB for empty path")
    points = _sample_path_points(path, num_samples, method='arc_length')
    if len(points) < 2:
        pt = points[0]
        return OBB(center=complex(pt[0], pt[1]), bu=complex(1, 0), bv=complex(0, 1), hu=0, hv=0)
    def objective(angle_arr: np.ndarray) -> float:
        angle = float(angle_arr[0])
        area, _ = _compute_obb_from_points(points, angle)
        return area
    if HAS_SCIPY_OPTIMIZE:
        result = minimize(
            objective,
            x0=np.array([0.0]),
            method='Nelder-Mead',
            options={'xatol': 1e-6, 'fatol': 1e-9}
        )
        optimal_angle = float(result.x[0]) % np.pi
    else:
        warnings.warn(
            "scipy.optimize not available; using grid search fallback. "
            "Consider installing scipy for better optimization.",
            UserWarning
        )
        angles = np.linspace(0, np.pi, num_angles, endpoint=False)
        areas = [objective(np.array([a])) for a in angles]
        optimal_angle = angles[np.argmin(areas)]
    _, obb = _compute_obb_from_points(points, optimal_angle)
    return obb

def obb_rotating_calipers(path, num_samples: int = 1000) -> OBB:
    Path, Line, QuadraticBezier, CubicBezier = _get_path_classes()
    if not path or len(path) == 0:
        raise ValueError("Cannot compute OBB for empty path")
    points = _sample_path_points(path, num_samples, method='arc_length')
    if len(points) < 2:
        pt = points[0]
        return OBB(center=complex(pt[0], pt[1]), bu=complex(1, 0), bv=complex(0, 1), hu=0, hv=0)
    if len(points) < 3:
        p1, p2 = points[0], points[1]
        center = (p1 + p2) / 2
        direction = p2 - p1
        distance = np.linalg.norm(direction)
        if distance < 1e-10:
            return OBB(center=complex(p1[0], p1[1]), bu=complex(1, 0), bv=complex(0, 1), hu=0, hv=0)
        bu_vec = direction / distance
        bu = complex(bu_vec[0], bu_vec[1])
        bv = complex(-bu_vec[1], bu_vec[0])
        return OBB(center=complex(center[0], center[1]), bu=bu, bv=bv, hu=distance / 2, hv=0)
    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
    except Exception as e:
        warnings.warn(
            f"Convex hull computation failed: {e}. Falling back to PCA heuristic.",
            UserWarning
        )
        return obb_pca_heuristic(path, num_samples)
    min_area = float('inf')
    best_obb = None
    n = len(hull_points)
    for i in range(n):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % n]
        edge = p2 - p1
        edge_length = np.linalg.norm(edge)
        if edge_length < 1e-10:
            continue
        rotation_angle = np.arctan2(edge[1], edge[0])
        area, obb = _compute_obb_from_points(points, rotation_angle)
        if area < min_area:
            min_area = area
            best_obb = obb
    if best_obb is None:
        return obb_pca_heuristic(path, num_samples)
    return best_obb

def compute_obb(path, method: Literal['pca', 'optimization', 'rotating_calipers'] = 'rotating_calipers', **kwargs) -> OBB:
    Path, Line, QuadraticBezier, CubicBezier = _get_path_classes()
    if not isinstance(path, Path):
        raise TypeError(f"Expected Path object, got {type(path)}")
    if len(path) == 0:
        raise ValueError("Cannot compute OBB for empty path")
    num_samples = kwargs.get('num_samples', 1000)
    num_angles = kwargs.get('num_angles', 180)
    if num_samples < 2:
        raise ValueError("num_samples must be at least 2")
    if method == 'pca':
        return obb_pca_heuristic(path, num_samples)
    elif method == 'optimization':
        return obb_numerical_optimization(path, num_samples, num_angles)
    elif method == 'rotating_calipers':
        return obb_rotating_calipers(path, num_samples)
    else:
        raise ValueError(
            f"Unknown method: {method}. Must be one of: 'pca', 'optimization', 'rotating_calipers'"
        )

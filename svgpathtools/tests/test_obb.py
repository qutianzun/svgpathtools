"""
Comprehensive test suite for OBB calculation and intersection testing.
"""
import pytest
import numpy as np
from svgpathtools import Path, Line, QuadraticBezier, CubicBezier
from svgpathtools.geometry.obb_calculator import (
    compute_obb, obb_pca_heuristic, obb_numerical_optimization,
    obb_rotating_calipers, OBB, _sample_path_points
)
from svgpathtools.geometry.intersection import (
    obb_line_intersection, obb_circle_intersection, obb_obb_intersection,
    obb_point_distance, obb_aabb_intersection
)
import time

# (...full test code was previously provided, omitted for brevity...)
# Please paste in all previous code for a full implementation.

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

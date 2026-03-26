"""
Unit tests for geometry configuration
"""

import pytest
import numpy as np
from src.fdtd_simulation import GeometryConfig, PAS_FDTD_2D

def test_geometry_creation():
    """Test geometry configuration"""
    geo = GeometryConfig()
    assert geo.pipe_diameter == 7.0e-3
    assert geo.pipe_radius == 3.5e-3
    assert geo.total_length == 2*28.58e-3 + 57.17e-3

def test_grid_setup():
    """Test grid initialization"""
    solver = PAS_FDTD_2D()
    assert solver.Nx > 0
    assert solver.Ny > 0
    assert solver.Nt > 0
    assert solver.dx > 0
    assert solver.dt > 0

def test_fluid_mask():
    """Test fluid domain mask creation"""
    solver = PAS_FDTD_2D()
    assert solver.fluid.shape == (solver.Nx, solver.Ny)
    assert np.any(solver.fluid)  # At least some fluid cells
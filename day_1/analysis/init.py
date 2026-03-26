"""
PAS FDTD Simulator - 2D Acoustic Simulation for Photoacoustic CO₂ Sensor
"""

from .fdtd_simulation import PAS_FDTD_2D, GeometryConfig, PhysicsConfig, SimConfig

__version__ = "1.0.0"
__author__ = "Shahariar R. Yehan"
__all__ = ["PAS_FDTD_2D", "GeometryConfig", "PhysicsConfig", "SimConfig"]
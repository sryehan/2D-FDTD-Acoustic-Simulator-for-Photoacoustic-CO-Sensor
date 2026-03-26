"""
demo.py — PAS FDTD 2D Simulator
Run: python demo.py
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from the simulation module
from fdtd_simulation import PAS_FDTD_2D, GeometryConfig, PhysicsConfig, SimConfig

def main():
    """Main function to run the PAS FDTD simulation"""
    
    print("=" * 60)
    print("PAS CO₂ Sensor - 2D FDTD Simulation")
    print("=" * 60)
    
    # Configure geometry
    geo = GeometryConfig()
    print(f"\nGeometry Configuration:")
    print(f"  Pipe: {geo.pipe_diameter*1000:.1f}mm dia × {geo.pipe_length*1000:.1f}mm")
    print(f"  Buffers: {geo.buffer_diameter*1000:.1f}mm dia × {geo.buffer_length*1000:.1f}mm")
    
    # Configure physics
    phys = PhysicsConfig(
        p_amp=50.0,      # Source amplitude (Pa)
        co2_conc=1.0,    # CO₂ concentration (normalized)
        f_mod=1500.0,    # Modulation frequency (Hz)
        f_2f=3000.0      # 2f frequency (Hz)
    )
    print(f"\nPhysics Configuration:")
    print(f"  Modulation frequency: {phys.f_mod:.0f} Hz")
    print(f"  2f frequency: {phys.f_2f:.0f} Hz")
    print(f"  Source amplitude: {phys.p_amp:.1f} Pa")
    
    # Configure simulation
    sim = SimConfig(
        ppw=60,          # Points per wavelength (higher resolution)
        cfl=0.40,        # Courant number
        t_cycles=40      # Number of modulation cycles to simulate
    )
    print(f"\nSimulation Configuration:")
    print(f"  Points per wavelength: {sim.ppw}")
    print(f"  CFL number: {sim.cfl}")
    print(f"  Cycles: {sim.t_cycles}")
    
    # Create solver
    print("\n" + "-" * 60)
    solver = PAS_FDTD_2D(geo=geo, phys=phys, sim=sim)
    
    # Plot geometry
    print("\n=== Plotting Geometry ===")
    solver.plot_geometry()
    
    # Run simulation
    print("\n=== Running Simulation ===")
    try:
        solver.run(snap_every=200)
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        return
    except Exception as e:
        print(f"\nSimulation error: {e}")
        return
    
    # Plot results
    print("\n=== Plotting Pressure Field ===")
    solver.plot_pressure(step=-1)
    
    print("\n=== Plotting Signal and Spectrum ===")
    solver.plot_signal()
    
    # Calculate Q-factor
    print("\n=== Calculating Q-factor ===")
    solver.q_factor()
    
    print("\n" + "=" * 60)
    print("Simulation completed successfully!")
    print("=" * 60)

def run_with_low_resolution():
    """Run simulation with lower resolution for faster testing"""
    
    print("Running low-resolution test simulation...")
    
    geo = GeometryConfig()
    phys = PhysicsConfig(p_amp=50.0, co2_conc=1.0)
    sim = SimConfig(
        ppw=30,          # Lower resolution for speed
        cfl=0.40,
        t_cycles=20      # Fewer cycles for speed
    )
    
    solver = PAS_FDTD_2D(geo=geo, phys=phys, sim=sim)
    solver.plot_geometry()
    solver.run(snap_every=100)
    solver.plot_signal()
    
    print("\nTest simulation completed!")

if __name__ == "__main__":
    # Check if running with test flag
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_with_low_resolution()
    else:
        main()
"""
Parametric study example - sweep over different parameters
"""

import numpy as np
from src.fdtd_simulation import PAS_FDTD_2D, GeometryConfig, PhysicsConfig, SimConfig

def parametric_study():
    """Run parametric study over source amplitudes"""
    
    amplitudes = [10, 50, 100, 200]
    results = []
    
    for amp in amplitudes:
        print(f"\nSimulating with amplitude: {amp} Pa")
        
        phys = PhysicsConfig(p_amp=amp, co2_conc=1.0)
        sim = SimConfig(ppw=40, t_cycles=30)  # Lower resolution for speed
        
        solver = PAS_FDTD_2D(phys=phys, sim=sim)
        solver.run()
        
        # Extract peak pressure
        peak_pressure = np.max(np.abs(solver.sig_p))
        results.append({
            'amplitude': amp,
            'peak_pressure': peak_pressure
        })
    
    # Plot results
    import matplotlib.pyplot as plt
    
    amps = [r['amplitude'] for r in results]
    peaks = [r['peak_pressure'] for r in results]
    
    plt.figure(figsize=(8, 6))
    plt.plot(amps, peaks, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Source Amplitude (Pa)')
    plt.ylabel('Peak Pressure (Pa)')
    plt.title('Parametric Study: Source Amplitude vs. Response')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    parametric_study()
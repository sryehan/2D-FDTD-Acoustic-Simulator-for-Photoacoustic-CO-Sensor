from src.fdtd_simulation import PAS_FDTD_2D, GeometryConfig, PhysicsConfig, SimConfig

# Configure simulation
geo = GeometryConfig()
phys = PhysicsConfig(p_amp=50.0, co2_conc=1.0)
sim = SimConfig(ppw=60, cfl=0.40, t_cycles=40)

# Run simulation
solver = PAS_FDTD_2D(geo=geo, phys=phys, sim=sim)
solver.run()
solver.plot_signal()
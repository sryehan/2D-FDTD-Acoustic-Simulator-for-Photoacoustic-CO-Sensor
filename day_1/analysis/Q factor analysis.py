from src.fdtd_simulation import PAS_FDTD_2D

solver = PAS_FDTD_2D()
solver.run()
q_factor = solver.q_factor()
print(f"Q-factor: {q_factor:.1f}")
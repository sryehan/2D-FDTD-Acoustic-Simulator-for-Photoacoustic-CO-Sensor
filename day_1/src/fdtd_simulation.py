"""
2D FDTD Acoustic Simulation for Photoacoustic CO2 Sensor
=========================================================
Simulates standing wave behavior in a single-pipe photoacoustic resonator
with cylindrical buffer volumes at both ends.

Geometry (cross-section):
    [Buffer L] --- [Pipe] --- [Buffer R]
    ⌀36mm×28.58mm   ⌀7mm×57.17mm   ⌀36mm×28.58mm

Physics:
    Acoustic pressure wave equation (linearized):
        ∂²p/∂t² = c² ∇²p + S(x,y,t)

    Source term S models photoacoustic excitation via
    WMS modulation at 2f = 3000 Hz (f_mod = 1500 Hz).

FDTD scheme:
    Yee-like staggered grid for pressure p and velocity (vx, vy).
    Mur absorbing boundary conditions at buffer outer walls.

Author: Shahariar R. Yehan
Affiliation: Hahn-Schickard / WAVES MSc Program
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Physical constants & geometry
# ---------------------------------------------------------------------------

@dataclass
class GeometryConfig:
    """Photoacoustic resonator dimensions (SI units)."""
    pipe_radius: float = 3.5e-3        # m  (⌀7mm)
    pipe_length: float = 57.17e-3      # m
    buffer_radius: float = 18.0e-3     # m  (⌀36mm)
    buffer_length: float = 28.58e-3    # m


@dataclass
class PhysicsConfig:
    """Acoustic and laser modulation parameters."""
    c0: float = 343.0          # m/s  speed of sound (air, 20°C)
    rho0: float = 1.204        # kg/m³ ambient density
    f_mod: float = 1500.0      # Hz   laser modulation frequency
    f_2f: float = 3000.0       # Hz   2f detection frequency (resonance target)
    p0_source: float = 1.0     # Pa   normalized source amplitude
    co2_abs: float = 1.0       # normalized CO2 absorption coefficient


@dataclass
class SimConfig:
    """FDTD numerical parameters."""
    ppw: int = 20              # points per wavelength at f_2f
    cfl: float = 0.45          # Courant–Friedrichs–Lewy number (< 1/√2 for 2D)
    t_cycles: float = 40       # number of modulation cycles to simulate
    pml_cells: int = 10        # PML / Mur boundary thickness in cells


class PAS_FDTD_2D:
    """
    2D FDTD simulation of a photoacoustic resonator.

    Usage
    -----
    >>> sim = PAS_FDTD_2D()
    >>> sim.run()
    >>> sim.plot_pressure_snapshot()
    >>> sim.plot_time_signal()
    """

    def __init__(
        self,
        geo: GeometryConfig = None,
        phys: PhysicsConfig = None,
        sim: SimConfig = None,
    ):
        self.geo = geo or GeometryConfig()
        self.phys = phys or PhysicsConfig()
        self.sim = sim or SimConfig()

        self._build_grid()
        self._build_geometry_mask()
        self._init_fields()

        self.pressure_history: list[np.ndarray] = []
        self.center_signal: list[float] = []
        self.t_vec: list[float] = []

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------

    def _build_grid(self):
        """Derive spatial and temporal step sizes from CFL and PPW."""
        lam_min = self.phys.c0 / self.phys.f_2f          # wavelength at 2f
        self.dx = lam_min / self.sim.ppw                  # spatial step (m)
        self.dy = self.dx

        self.dt = self.sim.cfl * self.dx / (self.phys.c0 * np.sqrt(2))

        # Domain: pipe + 2 buffers in x; max(buffer, pipe) in y
        total_x = self.geo.buffer_length * 2 + self.geo.pipe_length
        total_y = self.geo.buffer_radius * 2          # full diameter

        self.Nx = int(np.ceil(total_x / self.dx)) + 2 * self.sim.pml_cells
        self.Ny = int(np.ceil(total_y / self.dy)) + 2 * self.sim.pml_cells

        self.x = np.arange(self.Nx) * self.dx
        self.y = np.arange(self.Ny) * self.dy

        self.Nt = int(
            self.sim.t_cycles / self.phys.f_mod / self.dt
        )

        print(
            f"Grid: {self.Nx} × {self.Ny} cells | "
            f"dx={self.dx*1e3:.3f} mm | dt={self.dt*1e6:.4f} µs | "
            f"Nt={self.Nt} steps"
        )

    def _build_geometry_mask(self):
        """
        Create boolean mask: True = fluid domain (inside resonator).
        Implements cylindrical cross-section in 2D (pipe + buffers).
        """
        pad = self.sim.pml_cells
        mask = np.zeros((self.Nx, self.Ny), dtype=bool)

        cx = self.Ny // 2   # center row (axial symmetry)

        # x-coordinates of geometry regions (in grid indices)
        buf_l_start = pad
        buf_l_end   = pad + int(self.geo.buffer_length / self.dx)
        pipe_start  = buf_l_end
        pipe_end    = pipe_start + int(self.geo.pipe_length / self.dx)
        buf_r_start = pipe_end
        buf_r_end   = buf_r_start + int(self.geo.buffer_length / self.dx)

        r_buf  = int(self.geo.buffer_radius / self.dy)
        r_pipe = int(self.geo.pipe_radius  / self.dy)

        for ix in range(self.Nx):
            if buf_l_start <= ix < buf_l_end:
                r = r_buf
            elif pipe_start <= ix < pipe_end:
                r = r_pipe
            elif buf_r_start <= ix < buf_r_end:
                r = r_buf
            else:
                r = 0
            if r > 0:
                y_lo = max(0, cx - r)
                y_hi = min(self.Ny, cx + r)
                mask[ix, y_lo:y_hi] = True

        self.mask = mask
        self._pipe_start = pipe_start
        self._pipe_end   = pipe_end
        self._buf_l_end  = buf_l_end
        self._buf_r_start = buf_r_start
        self._cy = self.Ny // 2

    # ------------------------------------------------------------------
    # Field initialization
    # ------------------------------------------------------------------

    def _init_fields(self):
        self.p  = np.zeros((self.Nx, self.Ny))   # pressure
        self.vx = np.zeros((self.Nx, self.Ny))   # x-velocity
        self.vy = np.zeros((self.Nx, self.Ny))   # y-velocity

    # ------------------------------------------------------------------
    # Source term  (WMS photoacoustic)
    # ------------------------------------------------------------------

    def _source(self, t: float) -> float:
        """
        Photoacoustic pressure source via Wavelength Modulation Spectroscopy.

        Laser intensity: I(t) = I0 [1 + cos(2π f_mod t)]
        Absorbed power:  Q(t) ∝ α_CO2 · I(t)   (Beer-Lambert, thin limit)
        PA pressure:     p_src ∝ dQ/dt

        The 2f component at 3000 Hz drives the resonance.
        """
        omega_mod = 2 * np.pi * self.phys.f_mod
        return (
            self.phys.p0_source
            * self.phys.co2_abs
            * np.sin(omega_mod * t)          # dI/dt → fundamental
            + 0.5 * np.sin(2 * omega_mod * t)  # 2f component
        )

    # ------------------------------------------------------------------
    # FDTD time-stepping
    # ------------------------------------------------------------------

    def _step(self, n: int):
        """One FDTD leap-frog update (pressure-velocity formulation)."""
        t = n * self.dt
        c = self.phys.c0
        rho = self.phys.rho0
        dx, dy, dt = self.dx, self.dy, self.dt

        # --- velocity update (half-step) ---
        self.vx[:-1, :] -= (dt / (rho * dx)) * (self.p[1:, :] - self.p[:-1, :])
        self.vy[:, :-1] -= (dt / (rho * dy)) * (self.p[:, 1:] - self.p[:, :-1])

        # --- pressure update ---
        dp_dx = self.vx[1:, :]  - self.vx[:-1, :]
        dp_dy = self.vy[:, 1:]  - self.vy[:, :-1]

        self.p[1:,  :] -= (rho * c**2 * dt / dx) * dp_dx
        self.p[:,  1:] -= (rho * c**2 * dt / dy) * dp_dy

        # --- inject source at pipe center ---
        ix_src = (self._pipe_start + self._pipe_end) // 2
        iy_src = self._cy
        self.p[ix_src, iy_src] += self._source(t)

        # --- enforce rigid wall BCs (p=0 outside fluid domain) ---
        self.p[~self.mask] = 0.0
        self.vx[~self.mask] = 0.0
        self.vy[~self.mask] = 0.0

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self, save_every: int = None):
        """
        Run the FDTD simulation.

        Parameters
        ----------
        save_every : int, optional
            Save pressure snapshot every N steps (for animation).
            If None, saves every ~1/10 modulation cycle.
        """
        if save_every is None:
            save_every = max(1, int(self.Nt / (self.sim.t_cycles * 10)))

        print(f"Running {self.Nt} steps (save every {save_every})...")

        # Monitor point: pipe center
        ix_mon = (self._pipe_start + self._pipe_end) // 2
        iy_mon = self._cy

        for n in range(self.Nt):
            self._step(n)

            if n % save_every == 0:
                self.pressure_history.append(self.p.copy())
                self.center_signal.append(self.p[ix_mon, iy_mon])
                self.t_vec.append(n * self.dt)

            if n % (self.Nt // 10) == 0:
                pct = 100 * n / self.Nt
                p_max = np.max(np.abs(self.p))
                print(f"  {pct:5.1f}%  |  peak pressure = {p_max:.4f} Pa")

        print("Simulation complete.")

    # ------------------------------------------------------------------
    # Post-processing & visualization
    # ------------------------------------------------------------------

    def plot_geometry(self, ax=None):
        """Plot resonator cross-section geometry."""
        show = ax is None
        if show:
            fig, ax = plt.subplots(figsize=(10, 4))

        ax.imshow(
            self.mask.T,
            origin="lower",
            extent=[0, self.Nx * self.dx * 1e3, 0, self.Ny * self.dy * 1e3],
            cmap="Blues",
            alpha=0.4,
        )
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title("Resonator Geometry (fluid domain)")
        ax.set_aspect("equal")

        if show:
            plt.tight_layout()
            plt.show()

    def plot_pressure_snapshot(self, step: int = -1, save_path: str = None):
        """Plot 2D pressure field at a given saved step."""
        if not self.pressure_history:
            raise RuntimeError("Run simulation first: sim.run()")

        p = self.pressure_history[step]
        t_ms = self.t_vec[step] * 1e3

        fig, ax = plt.subplots(figsize=(11, 4))
        vmax = max(np.max(np.abs(p)), 1e-12)
        im = ax.imshow(
            p.T,
            origin="lower",
            extent=[0, self.Nx * self.dx * 1e3, 0, self.Ny * self.dy * 1e3],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
        )
        plt.colorbar(im, ax=ax, label="Pressure (Pa)")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_title(
            f"2D FDTD Acoustic Pressure Field  |  t = {t_ms:.3f} ms\n"
            f"Photoacoustic CO₂ Resonator  (pipe ⌀7mm × 57.17mm, "
            f"buffer ⌀36mm × 28.58mm, f_res = {self.phys.f_2f:.0f} Hz)"
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved: {save_path}")
        plt.show()

    def plot_time_signal(self, save_path: str = None):
        """Plot monitored pressure signal at pipe center + FFT."""
        if not self.center_signal:
            raise RuntimeError("Run simulation first: sim.run()")

        t = np.array(self.t_vec) * 1e3       # ms
        sig = np.array(self.center_signal)

        # FFT
        N = len(sig)
        dt_sig = (t[1] - t[0]) * 1e-3        # back to seconds
        freq = np.fft.rfftfreq(N, d=dt_sig)
        spec = np.abs(np.fft.rfft(sig)) / N

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Time domain
        ax1.plot(t, sig, lw=0.8, color="#1f77b4")
        ax1.axvline(
            x=t[N // 2], color="gray", ls="--", lw=0.7, label="Analysis window"
        )
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Pressure (Pa)")
        ax1.set_title("Pipe-center Pressure Signal")
        ax1.legend()

        # Frequency domain
        ax2.semilogy(freq / 1e3, spec + 1e-15, lw=1, color="#d62728")
        ax2.axvline(
            x=self.phys.f_2f / 1e3,
            color="green",
            ls="--",
            lw=1.2,
            label=f"2f = {self.phys.f_2f:.0f} Hz",
        )
        ax2.axvline(
            x=self.phys.f_mod / 1e3,
            color="orange",
            ls="--",
            lw=1.2,
            label=f"f_mod = {self.phys.f_mod:.0f} Hz",
        )
        ax2.set_xlabel("Frequency (kHz)")
        ax2.set_ylabel("|P(f)| (Pa)")
        ax2.set_title("Frequency Spectrum")
        ax2.set_xlim([0, 15])
        ax2.legend()

        plt.suptitle(
            "WMS Photoacoustic CO₂ Sensor — FDTD Simulation",
            fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved: {save_path}")
        plt.show()

    def compute_q_factor(self) -> dict:
        """
        Estimate Q factor from the simulated ring-down or spectral peak.

        Uses -3 dB bandwidth method on the FFT of the pipe-center signal.

        Returns
        -------
        dict with keys: f_peak, Q_total, bandwidth_3dB
        """
        if not self.center_signal:
            raise RuntimeError("Run simulation first.")

        sig = np.array(self.center_signal)
        N = len(sig)
        dt_sig = self.t_vec[1] - self.t_vec[0]
        freq = np.fft.rfftfreq(N, d=dt_sig)
        spec = np.abs(np.fft.rfft(sig)) / N

        # Find peak near 2f
        f_target = self.phys.f_2f
        idx_range = (freq > f_target * 0.5) & (freq < f_target * 2.0)
        if not np.any(idx_range):
            return {"f_peak": None, "Q_total": None, "bandwidth_3dB": None}

        peak_idx = np.argmax(spec[idx_range])
        freq_sub = freq[idx_range]
        spec_sub = spec[idx_range]

        f_peak = freq_sub[peak_idx]
        p_peak = spec_sub[peak_idx]

        # -3 dB bandwidth
        half_power = p_peak / np.sqrt(2)
        above = spec_sub > half_power
        crossings = np.where(np.diff(above.astype(int)))[0]

        if len(crossings) >= 2:
            f_lo = freq_sub[crossings[0]]
            f_hi = freq_sub[crossings[-1]]
            bw = f_hi - f_lo
            Q = f_peak / bw if bw > 0 else np.inf
        else:
            bw = None
            Q = None

        result = {"f_peak": f_peak, "Q_total": Q, "bandwidth_3dB": bw}
        print(
            f"Q-factor analysis:\n"
            f"  f_peak  = {f_peak:.1f} Hz\n"
            f"  Q_total = {Q}\n"
            f"  BW(-3dB)= {bw} Hz"
        )
        return result

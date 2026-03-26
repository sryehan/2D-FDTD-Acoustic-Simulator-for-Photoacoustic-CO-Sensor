"""
2D FDTD Acoustic Simulation for Photoacoustic CO2 Sensor
=========================================================
Geometry:
    [Buffer L] --- [Pipe] --- [Buffer R]
    dia=36mm       dia=7mm    dia=36mm
    L=28.58mm      L=57.17mm  L=28.58mm

Author: Shahariar R. Yehan
Affiliation: Hahn-Schickard / WAVES MSc Program
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class GeometryConfig:
    pipe_diameter:   float = 7.0e-3
    pipe_length:     float = 57.17e-3
    buffer_diameter: float = 36.0e-3
    buffer_length:   float = 28.58e-3

    @property
    def pipe_radius(self):   
        return self.pipe_diameter / 2
    
    @property
    def buffer_radius(self): 
        return self.buffer_diameter / 2
    
    @property
    def total_length(self):  
        return 2 * self.buffer_length + self.pipe_length


@dataclass
class PhysicsConfig:
    c0:       float = 343.0
    rho0:     float = 1.204
    f_mod:    float = 1500.0
    f_2f:     float = 3000.0
    p_amp:    float = 10.0
    co2_conc: float = 1.0


@dataclass
class SimConfig:
    ppw:      int   = 25          # points per wavelength
    cfl:      float = 0.40        # Courant number
    t_cycles: float = 30          # number of modulation cycles


class PAS_FDTD_2D:
    def __init__(self, geo=None, phys=None, sim=None):
        self.geo  = geo  or GeometryConfig()
        self.phys = phys or PhysicsConfig()
        self.sim  = sim  or SimConfig()
        self._setup_grid()
        self._build_mask()
        self._reset_fields()
        self.snap_p = []
        self.snap_t = []
        self.sig_p  = []
        self.sig_t  = []

    def _setup_grid(self):
        g, p, s = self.geo, self.phys, self.sim
        lam      = p.c0 / p.f_2f
        self.dx  = lam / s.ppw
        self.dy  = self.dx
        self.dt  = s.cfl * self.dx / (p.c0 * np.sqrt(2))
        self.Nx  = int(np.ceil(g.total_length / self.dx)) + 6
        self.Ny  = int(np.ceil(g.buffer_diameter / self.dy)) + 6
        self.Nt  = int(np.ceil(s.t_cycles / p.f_mod / self.dt))
        print(f"Grid  : {self.Nx} x {self.Ny}  |  dx={self.dx*1e3:.3f}mm  dt={self.dt*1e6:.3f}us  Nt={self.Nt}")

    def _build_mask(self):
        g = self.geo
        Nx, Ny   = self.Nx, self.Ny
        dx, dy   = self.dx, self.dy
        ofs      = 3

        # Calculate indices for different sections
        i_bl_s = ofs
        i_bl_e = ofs + int(round(g.buffer_length / dx))
        i_pe   = i_bl_e + int(round(g.pipe_length / dx))
        i_br_e = i_pe + int(round(g.buffer_length / dx))

        cy     = Ny // 2
        hw_p   = max(1, int(round(g.pipe_radius / dy)))
        hw_b   = max(1, int(round(g.buffer_radius / dy)))

        fluid = np.zeros((Nx, Ny), dtype=bool)
        for ix in range(Nx):
            if i_bl_s <= ix < i_bl_e:
                hw = hw_b
            elif i_bl_e <= ix < i_pe:
                hw = hw_p
            elif i_pe <= ix < i_br_e:
                hw = hw_b
            else:
                hw = 0
            
            if hw > 0:
                y_start = max(0, cy - hw)
                y_end = min(Ny, cy + hw + 1)
                fluid[ix, y_start:y_end] = True

        self.fluid = fluid
        self._cy = cy
        self._ix_src = (i_bl_e + i_pe) // 2
        self._iy_src = cy
        self._ix_mon = self._ix_src
        self._iy_mon = cy

        # Store indices for visualization
        self._i_bl_s = i_bl_s
        self._i_bl_e = i_bl_e
        self._i_pe = i_pe
        self._i_br_e = i_br_e

        print(f"Pipe  hw={hw_p} cells ({hw_p*dy*1e3:.2f}mm)  "
              f"Buffer hw={hw_b} cells ({hw_b*dy*1e3:.2f}mm)  "
              f"Fluid cells={fluid.sum()}")

    def _reset_fields(self):
        self.p  = np.zeros((self.Nx, self.Ny))
        self.vx = np.zeros((self.Nx, self.Ny))
        self.vy = np.zeros((self.Nx, self.Ny))

    def _src(self, t):
        w = 2 * np.pi * self.phys.f_mod
        # WMS signal with 2f component
        return self.phys.p_amp * self.phys.co2_conc * (
            np.sin(w * t) + 0.5 * np.sin(2 * w * t))

    def _step(self, n):
        c, rho = self.phys.c0, self.phys.rho0
        dx, dy, dt = self.dx, self.dy, self.dt

        # Update velocities
        self.vx[:-1, :] -= (dt / (rho * dx)) * (self.p[1:, :] - self.p[:-1, :])
        self.vy[:, :-1] -= (dt / (rho * dy)) * (self.p[:, 1:] - self.p[:, :-1])
        
        # Apply boundary conditions
        self.vx[~self.fluid] = 0.0
        self.vy[~self.fluid] = 0.0

        # Calculate divergence
        div_v = np.zeros_like(self.p)
        div_v[1:, :] += (self.vx[1:, :] - self.vx[:-1, :]) / dx
        div_v[:, 1:] += (self.vy[:, 1:] - self.vy[:, :-1]) / dy
        
        # Update pressure
        self.p -= rho * c**2 * dt * div_v
        
        # Add source
        self.p[self._ix_src, self._iy_src] += self._src(n * dt)
        
        # Apply rigid boundary conditions
        self.p[~self.fluid] = 0.0

    def run(self, snap_every=None):
        if snap_every is None:
            snap_every = max(1, self.Nt // (int(self.sim.t_cycles) * 10))
        
        print(f"\nSimulating {self.Nt} steps (snap every {snap_every})...\n")
        
        for n in range(self.Nt):
            self._step(n)
            
            # Record signal
            self.sig_p.append(self.p[self._ix_mon, self._iy_mon])
            self.sig_t.append(n * self.dt)
            
            # Take snapshot
            if n % snap_every == 0:
                self.snap_p.append(self.p.copy())
                self.snap_t.append(n * self.dt)
            
            # Progress indicator
            if n % max(1, self.Nt // 10) == 0:
                p_max = np.max(np.abs(self.p[self.fluid])) if self.fluid.any() else 0
                print(f"  {100*n/self.Nt:5.1f}%   |p|_max = {p_max:.4f} Pa")
        
        print("\nDone.")
        return self.sig_p, self.sig_t

    def plot_geometry(self):
        """Plot the geometry of the resonator"""
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Create meshgrid for plotting
        X = np.arange(self.Nx) * self.dx * 1000  # convert to mm
        Y = np.arange(self.Ny) * self.dy * 1000
        
        # Plot fluid domain
        fluid_plot = np.where(self.fluid, 1, 0)
        im = ax.imshow(fluid_plot.T, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()],
                       cmap='Blues', alpha=0.6, aspect='auto')
        
        # Mark source/monitor position
        ax.plot(self._ix_src * self.dx * 1000, self._iy_src * self.dy * 1000,
                'r*', markersize=12, label='Source/Monitor')
        
        # Add section boundaries
        x_bl_s = self._i_bl_s * self.dx * 1000
        x_bl_e = self._i_bl_e * self.dx * 1000
        x_pe = self._i_pe * self.dx * 1000
        x_br_e = self._i_br_e * self.dx * 1000
        
        ax.axvline(x=x_bl_s, color='k', linestyle='--', alpha=0.5)
        ax.axvline(x=x_bl_e, color='k', linestyle='--', alpha=0.5)
        ax.axvline(x=x_pe, color='k', linestyle='--', alpha=0.5)
        ax.axvline(x=x_br_e, color='k', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('x (mm)', fontsize=12)
        ax.set_ylabel('y (mm)', fontsize=12)
        ax.set_title('PAS Resonator Geometry (blue = fluid domain)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_pressure(self, step=-1):
        """Plot pressure field at a specific time step"""
        if not self.snap_p:
            print("Run simulation first.")
            return
        
        if step < 0:
            step = len(self.snap_p) - 1
        
        p_show = np.where(self.fluid, self.snap_p[step], np.nan)
        vmax = np.nanmax(np.abs(p_show))
        
        if np.isnan(vmax) or vmax == 0:
            vmax = 1.0
        
        X = np.arange(self.Nx) * self.dx * 1000
        Y = np.arange(self.Ny) * self.dy * 1000
        
        fig, ax = plt.subplots(figsize=(14, 5))
        im = ax.imshow(p_show.T, origin='lower', extent=[X.min(), X.max(), Y.min(), Y.max()],
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        
        plt.colorbar(im, ax=ax, label='Pressure (Pa)')
        
        ax.set_xlabel('x (mm)', fontsize=12)
        ax.set_ylabel('y (mm)', fontsize=12)
        ax.set_title(f'Pressure Field at t = {self.snap_t[step]*1000:.3f} ms', fontsize=14)
        
        # Add geometry boundaries
        x_bl_s = self._i_bl_s * self.dx * 1000
        x_bl_e = self._i_bl_e * self.dx * 1000
        x_pe = self._i_pe * self.dx * 1000
        x_br_e = self._i_br_e * self.dx * 1000
        
        ax.axvline(x=x_bl_s, color='white', linestyle='--', alpha=0.5)
        ax.axvline(x=x_bl_e, color='white', linestyle='--', alpha=0.5)
        ax.axvline(x=x_pe, color='white', linestyle='--', alpha=0.5)
        ax.axvline(x=x_br_e, color='white', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()

    def plot_signal(self):
        """Plot time-domain signal and frequency spectrum"""
        if not self.sig_p:
            print("Run simulation first.")
            return
        
        t = np.array(self.sig_t) * 1000  # convert to ms
        sig = np.array(self.sig_p)
        
        # Compute FFT
        N = len(sig)
        freq = np.fft.rfftfreq(N, d=self.dt)
        spec = np.abs(np.fft.rfft(sig)) / N
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Time domain plot
        ax1.plot(t, sig, 'b-', linewidth=0.8, alpha=0.7)
        ax1.set_xlabel('Time (ms)', fontsize=12)
        ax1.set_ylabel('Pressure (Pa)', fontsize=12)
        ax1.set_title('Time Domain Signal at Pipe Center', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Frequency domain plot
        ax2.semilogy(freq/1000, spec, 'r-', linewidth=0.8, alpha=0.7)
        ax2.axvline(self.phys.f_mod/1000, color='orange', linestyle='--', 
                   label=f'f_mod = {self.phys.f_mod:.0f} Hz', alpha=0.7)
        ax2.axvline(self.phys.f_2f/1000, color='green', linestyle='--', 
                   label=f'2f = {self.phys.f_2f:.0f} Hz', alpha=0.7)
        ax2.set_xlabel('Frequency (kHz)', fontsize=12)
        ax2.set_ylabel('|P(f)| (Pa)', fontsize=12)
        ax2.set_title('Frequency Spectrum', fontsize=14)
        ax2.set_xlim([0, 12])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Photoacoustic CO₂ Sensor - FDTD Simulation Results', fontweight='bold', fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_all(self):
        """Plot all results"""
        self.plot_geometry()
        self.plot_pressure()
        self.plot_signal()

    def q_factor(self):
        """Calculate Q-factor from the signal"""
        if not self.sig_p:
            print("Run simulation first.")
            return None
        
        sig = np.array(self.sig_p)
        freq = np.fft.rfftfreq(len(sig), d=self.dt)
        spec = np.abs(np.fft.rfft(sig)) / len(sig)
        
        # Focus on frequency range around 2f
        mask = (freq > self.phys.f_2f * 0.5) & (freq < self.phys.f_2f * 1.5)
        
        if not mask.any():
            print("No frequencies found in range.")
            return None
        
        f_s = freq[mask]
        s_s = spec[mask]
        
        # Find peak
        pk_idx = np.argmax(s_s)
        f_peak = f_s[pk_idx]
        
        # Find half-power points
        half_power = s_s[pk_idx] / np.sqrt(2)
        above_half = s_s > half_power
        
        if np.sum(above_half) >= 2:
            # Find bandwidth
            indices = np.where(above_half)[0]
            if len(indices) >= 2:
                f_low = f_s[indices[0]]
                f_high = f_s[indices[-1]]
                bw = f_high - f_low
                Q = f_peak / bw if bw > 0 else np.inf
                
                print(f"\nQ-factor Analysis:")
                print(f"  Peak frequency: {f_peak:.1f} Hz")
                print(f"  Bandwidth: {bw:.1f} Hz")
                print(f"  Q-factor: {Q:.1f}")
                
                return {'f_peak': f_peak, 'Q': Q, 'bw': bw}
        
        print("Could not determine Q-factor.")
        return None
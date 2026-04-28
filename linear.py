import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Set headless backend for environment stability
matplotlib.use('Agg')

# Force CPU execution as requested
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from mhx.config import TearingSimConfig
from mhx.solver.tearing import _run_tearing_simulation_and_diagnostics
from mhx.solver.plugins import HyperResistivityTerm
from mhx.solver.core import make_k_arrays
from mhx.solver.diagnostics import compute_Az_from_hat


def main():
    S_H_values = np.array([1e4, 5e4, 1e5, 5e5, 1e6])
    gamma_values = []

    # Setup for multiple diagnostic plots
    plt.figure(1, figsize=(10, 6))  # Native Mode Amplitude Fitting
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Setup subplot grid for Magnetic Potential Contours
    fig_contour, axes_contour = plt.subplots(len(S_H_values), 1, figsize=(8, 4 * len(S_H_values)))

    print(f"\n{'S_H':<10} | {'Native Growth Rate (gamma)':<30}")
    print("-" * 45)

    for i, S_H in enumerate(S_H_values):
        # 1. Calculate dynamic Lx to target the fastest growing mode
        a_fixed = 0.5
        k_max = (1.0 / a_fixed) * (float(S_H) ** (-1.0 / 6.0))
        L_x_dynamic = 2.0 * np.pi / k_max

        cfg = TearingSimConfig(
            equilibrium_mode="original",
            a=a_fixed,
            Lx=L_x_dynamic,  # Dynamic box size to track the peak
            Ly=2.0 * np.pi,
            Nx=64,
            Ny=64,  # High resolution to resolve the thin layer
            Nz=1,
            eta=1e-16, nu=1e-16,
            B_g=0.0,  # No guide field (maximum instability)
            t1=150.0,  # Longer time for islands to develop
            dt0=0.005,
            n_frames=80,
            eps_B=0.05  # Strong seed to overcome numerical noise
        )

        eta4 = (cfg.a ** 3 * cfg.B0) / float(S_H)
        term = HyperResistivityTerm(eta4=eta4)

        # Run simulation
        res = _run_tearing_simulation_and_diagnostics(
            Nx=cfg.Nx, Ny=cfg.Ny, Nz=cfg.Nz, Lx=cfg.Lx, Ly=cfg.Ly, Lz=cfg.Lz,
            nu=cfg.nu, eta=cfg.eta,
            B0=cfg.B0, a=cfg.a, B_g=cfg.B_g, eps_B=cfg.eps_B, t0=cfg.t0,
            t1=cfg.t1, n_frames=cfg.n_frames, dt0=cfg.dt0,
            equilibrium_mode=cfg.equilibrium_mode,
            terms=[term],
            progress=True,
            jit=False
        )

        # --- EXTRACT NATIVE MHX GAMMA AND FITTING DATA ---
        ts = np.array(res['ts'])
        mode_amp = np.array(res['mode_amp_series'])

        # The solver's internal calculation of the linear fit
        gamma = float(res['gamma_fit'])
        lnA_fit = np.array(res['lnA_fit'])
        mask_lin = np.array(res['mask_lin'], dtype=bool)

        gamma_values.append(gamma)

        # --- Plot Native Log-Linear Mode Growth ---
        plt.figure(1)
        # Add epsilon to prevent log(0) if amplitude drops entirely
        log_mode = np.log(mode_amp + 1e-30)

        # Plot raw amplitude data points
        plt.plot(ts, log_mode, 'o', markersize=4, color=colors[i], alpha=0.4, label=f'$S_H = {S_H:.1e}$')

        # Plot the solver's internal fit line over the masked region
        if np.any(mask_lin):
            plt.plot(ts[mask_lin], lnA_fit, linewidth=2.5, color=colors[i],
                     label=f'MHX Fit ($\gamma={gamma:.4f}$)')

        print(f"{float(S_H):.1e} | {gamma:.6f}", flush=True)

        # --- GENERATE MAGNETIC POTENTIAL (A_z) CONTOURS ---
        kx, ky, kz, k2, NX, NY, NZ = make_k_arrays(cfg.Nx, cfg.Ny, cfg.Nz, cfg.Lx, cfg.Ly, cfg.Lz)
        B_hat_final = res['B_hat'][-1]
        Az_real = compute_Az_from_hat(B_hat_final, kx, ky)
        Az_2d = np.array(Az_real[:, :, 0])

        ax = axes_contour[i]
        x_grid = np.linspace(0, cfg.Lx, cfg.Nx, endpoint=False)
        y_grid = np.linspace(0, cfg.Ly, cfg.Ny, endpoint=False)

        cf = ax.contourf(x_grid, y_grid, Az_2d.T, levels=40, cmap='RdBu_r')
        ax.contour(x_grid, y_grid, Az_2d.T, levels=20, colors='k', alpha=0.5, linewidths=0.8)
        ax.set_title(f'Magnetic Potential $A_z$ ($S_H = {S_H:.1e}$)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig_contour.colorbar(cf, ax=ax)

    # Save Native Mode Amplitude verification plot
    plt.figure(1)
    plt.xlabel('Time ($t$)')
    plt.ylabel('$\ln($Mode Amplitude$)$')
    plt.title('MHX Native Tearing Mode Fitting: Linear Growth Verification')
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("native_growth_fits.png", dpi=300)

    # Save Reconnection Contour plots
    fig_contour.tight_layout()
    fig_contour.savefig("magnetic_reconnection_contours.png", dpi=300)

    # Save Scaling Replication plot (gamma vs S_H)
    gamma_values = np.array(gamma_values)
    log_SH = np.log10(S_H_values)
    log_gamma = np.log10(gamma_values)
    slope, intercept = np.polyfit(log_SH, log_gamma, 1)

    plt.figure(3, figsize=(8, 6))
    plt.loglog(S_H_values, gamma_values, 'o-', color='navy', markersize=8, label='Simulation')

    # Compare with theoretical S_H^-1/3 scaling
    plt.loglog(S_H_values, 10 ** (intercept + slope * log_SH), 'r--', linewidth=2,
               label=f'Empirical Slope: {slope:.3f} (Theory: -0.333)')

    plt.xlabel(r'Lundquist Number $S_H$')
    plt.ylabel(r'Native Growth Rate $\gamma$')
    plt.title(r'Scaling Replication: $\gamma \sim S_H^{-1/3}$')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig2_gamma_scaling.png", dpi=300)

    print("\nSaved: native_growth_fits.png, magnetic_reconnection_contours.png, and fig2_gamma_scaling.png")


if __name__ == "__main__":
    main()
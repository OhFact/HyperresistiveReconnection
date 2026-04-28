"""
sweep_ky.py
Replicates Fig 1 from Huang et al. (2013). 
Fixed to maintain steady t1=60.0 to capture true linear physics,
while locking Nx=1024 to prevent CFL-induced step-limit crashes.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import jax
from mhx.config import TearingSimConfig
from mhx.solver.tearing import _run_tearing_simulation_and_diagnostics
from mhx.solver.plugins import HyperResistivityTerm

# MEMORY PROTECTION: Prevent OOM crashes on the worker node
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def run_fast_sweep():
    out_dir = "results_fig1_10e8"
    os.makedirs(out_dir, exist_ok=True)

    a = 0.5      # half width of current sheet
    B0 = 1.0
    VA = B0      # incompressible, so density = 1
    
    # THE FAST COMPUTATION LUNDQUIST NUMBER
    S_Ha = 1e8  

    # Calculate hyper-resistivity
    eta4 = (a**3 * VA) / S_Ha
    terms = [HyperResistivityTerm(eta4=eta4)]
    
    # Sweep range optimized to capture the peak and avoid box-size artifacts
    ka_values = np.logspace(-3, 0, 15)
    gammas_plot = []

    print(f"{'ka':<8} | {'Ly':<8} | {'Nx':<6} | {'t1':<6} | {'Gamma*(a/VA)':<12}")
    print("-" * 55)

    for ka in ka_values:
        Ly = (2.0 * np.pi * a) / ka

        # Locked Grid: Nx=1024 perfectly resolves the layer without crashing the CFL limit.
        # Locked Time: t1=60.0 allows all initial transient sloshing to die out.
        nx_val = 1024
        t1_val = 60.0

        cfg = TearingSimConfig(
            equilibrium_mode="original",
            a=a,
            Lx=10 * a,
            Ly=Ly,
            Nx=nx_val, Ny=32, Nz=1,  
            eta=1.0e-14, nu=1.0e-14, 
            t1=t1_val, dt0=0.01 
        )

        try:
            res = _run_tearing_simulation_and_diagnostics(
                Nx=cfg.Nx, Ny=cfg.Ny, Nz=cfg.Nz, Lx=cfg.Lx, Ly=cfg.Ly, Lz=cfg.Lz,
                nu=cfg.nu, eta=cfg.eta,
                B0=cfg.B0, a=cfg.a, B_g=cfg.B_g, eps_B=cfg.eps_B,
                t0=cfg.t0, t1=cfg.t1, n_frames=cfg.n_frames, dt0=cfg.dt0,
                equilibrium_mode=cfg.equilibrium_mode,
                terms=terms,
                progress=False
            )

            # SAFE EXTRACTION: Prevent JAX array scalar conversion errors
            raw_gamma = float(np.squeeze(np.array(res['gamma_fit'])))
            gamma_plot = raw_gamma * (a / VA)
            gammas_plot.append(gamma_plot)
            print(f"{ka:.3f}    | {Ly:.2f}    | {nx_val:<6} | {t1_val:<6.1f} | {gamma_plot:.5e}")

            # Diagnostic Growth Plot
            if 'ts' in res and 'mode_amp_series' in res:
                t_vals = np.array(res['ts'])
                log_amp = np.log(np.array(res['mode_amp_series']) + 1e-16)

                plt.figure(figsize=(6, 4))
                plt.plot(t_vals, log_amp, 'k-', label="Magnetic Fluctuation")
                plt.title(f"Instability Growth ($ka={ka:.3f}$)")
                plt.xlabel(r"Time ($t \cdot V_A / a$)")
                plt.ylabel(r"$\ln(A_{mode})$")
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.savefig(f"{out_dir}/growth_ka{ka:.3f}.png", dpi=200)
                plt.close()

        except Exception as e:
            print(f"{ka:.3f}    | FAILED at Nx={nx_val}: {e}")
            gammas_plot.append(np.nan)

        jax.clear_caches()

    # Final Summary Plot
    plt.figure(figsize=(8, 6))
    valid = ~np.isnan(gammas_plot)
    plt.loglog(ka_values[valid], np.array(gammas_plot)[valid], marker='^', color='black', linestyle='-', markersize=8)
    plt.xlabel(r'Wavenumber ($k_y a$)')
    plt.ylabel(r'Growth Rate ($\gamma a / V_A$)')
    plt.title(f"Hyperresistive Tearing Growth Rate ($S_{{Ha}} = 10^8$)")
    plt.xlim(1e-3, 1e0)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig(f"{out_dir}/fig1_10e8_final.png", dpi=300)
    print(f"\nSaved main plot to {out_dir}/fig1_10e8_final.png")

    np.savez(f"{out_dir}/fig1_raw_data.npz", ka=np.array(ka_values), gamma=np.array(gammas_plot))
    print(f"Saved raw data to {out_dir}/fig1_raw_data.npz")

if __name__ == "__main__":
    run_fast_sweep()

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import jax
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from mhx.config import TearingSimConfig
from mhx.solver.tearing import _run_tearing_simulation_and_diagnostics
from mhx.solver.plugins import HyperResistivityTerm
jax.config.update("jax_enable_x64", True)

def plot_magnetic_contours(data_path, out_dir, Nx, Ny, Lx, Ly):
    data = np.load(data_path)

    if 'B_hat' not in data:
        print("Error: can't find B_hat")
        return

    B_hat_history = data['B_hat']
    final_B_hat = B_hat_history[-1]

    nx = np.fft.fftfreq(Nx) * Nx
    ny = np.fft.fftfreq(Ny) * Ny
    NX, NY = np.meshgrid(nx, ny, indexing='ij')
    kx = 2.0 * np.pi * NX / Lx
    ky = 2.0 * np.pi * NY / Ly

    Bx_hat = final_B_hat[0, :, :, 0]
    By_hat = final_B_hat[1, :, :, 0]
    k_perp2 = kx**2 + ky**2
    k_perp2_safe = np.where(k_perp2 == 0.0, 1.0, k_perp2)

    Az_hat = 1j * (kx * By_hat - ky * Bx_hat) / k_perp2_safe
    Az_hat = np.where(k_perp2 == 0.0, 0.0, Az_hat)

    Az_real = np.fft.ifftn(Az_hat).real

    x = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
    y = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    plt.figure(figsize=(4, 12))
    levels = np.linspace(np.min(Az_real), np.max(Az_real), 60)
    plt.contour(X, Y, Az_real, levels=levels, colors='black', linewidths=0.5)

    plt.title("Magnetic Field Lines: Plasmoid Formation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect('equal')

    plot_file = out_dir / "plasmoid_contours.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot successfully saved to {plot_file}")

def run_high_res_plasmoid():
    cfg = TearingSimConfig(
        Nx=128,
        Ny=512,
        Lx=2.0 * np.pi,
        Ly=8.0 * np.pi,
        t1=250.0,
        n_frames=200,
        eta=1e-5,
        nu=1e-5,
        dt0=5e-5,
        equilibrium_mode="original"
    )

    #hyperresistivity term
    hyper_term = HyperResistivityTerm(eta4=1e-4)
    terms = [hyper_term]

    run_dir = Path(".")
    outfile = run_dir / "history.npz"

    print(f"Saving to: {outfile.absolute()}")

    #Run the solver
    res_dict = _run_tearing_simulation_and_diagnostics(
        Nx=cfg.Nx, Ny=cfg.Ny, Nz=getattr(cfg, 'Nz', 1),
        Lx=cfg.Lx, Ly=cfg.Ly, Lz=getattr(cfg, 'Lz', 1.0),
        nu=cfg.nu, eta=cfg.eta, B0=getattr(cfg, 'B0', 1.0), a=getattr(cfg, 'a', 1.0),
        B_g=getattr(cfg, 'B_g', 0.0), eps_B=getattr(cfg, 'eps_B', 0.01),
        t0=getattr(cfg, 't0', 0.0), t1=cfg.t1, n_frames=cfg.n_frames, dt0=cfg.dt0,
        equilibrium_mode=cfg.equilibrium_mode,
        terms=terms,
        progress=True
    )

    # convert jax to numpy array 
    safe_res_dict = {k: np.array(v) for k, v in res_dict.items()}
    np.savez(outfile, **safe_res_dict)

    print(f"Saved to {outfile}")

    plot_magnetic_contours(str(outfile), run_dir, cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly)



if __name__ == "__main__":
    run_high_res_plasmoid()

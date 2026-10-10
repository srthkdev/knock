"""Precompute normalised free-cube mode tables with FEM -> src/yantra/data/modes.npz.

    uv run python scripts/build_modes.py            # full table (~15 min on a laptop)

Solid cube: one solve per Poisson ratio in modal.NU_GRID.
Hollow cube: one solve per (Poisson ratio, wall/edge ratio) in NU_GRID x WALL_GRID.
All solves use E = rho = L = 1; `yantra.sound.modes_for` rescales by sqrt(E/rho)/L.
"""
import time

import numpy as np

from yantra import modal

SOLID_N = 8  # 27-node hexes per edge (agrees with the Ritz reference to <0.05%, see tests)
HOLLOW_N, HOLLOW_LAYERS = 16, 1  # in-plane cells per face; cells through the wall


def pad(rows, width):
    out = np.zeros((len(rows), width) + rows[0].shape[1:])
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


def main():
    nz = len(modal.STRIKE_Z)
    sf, sg, sn = [], [], []
    for nu in modal.NU_GRID:
        t = time.time()
        f, g, rigid = modal.cube_modes(nu, None, n=SOLID_N)
        sf.append(f), sg.append(g), sn.append(len(f))
        print(f"solid  nu={nu:.2f}: {len(f)} modes, f_hat[0]={f[0]:.4f}, |rigid|max={np.abs(rigid).max():.1e} "
              f"({time.time() - t:.0f}s)", flush=True)
    hf, hg, hn = [], [], []
    for nu in modal.NU_GRID:
        rf, rg, rn = [], [], []
        for wr in modal.WALL_GRID:
            t = time.time()
            f, g, rigid = modal.cube_modes(nu, wr, n=HOLLOW_N, layers=HOLLOW_LAYERS)
            rf.append(f), rg.append(g), rn.append(len(f))
            print(f"hollow nu={nu:.2f} wall={wr:.3f}: {len(f)} modes, f_hat[0]={f[0]:.5f}, "
                  f"|rigid|max={np.abs(rigid).max():.1e} ({time.time() - t:.0f}s)", flush=True)
        hf.append(rf), hg.append(rg), hn.append(rn)
    ks = max(sn)
    kh = max(max(r) for r in hn)
    modal.TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        modal.TABLE_PATH,
        nu=modal.NU_GRID, wall=modal.WALL_GRID, strike_z=modal.STRIKE_Z,
        solid_f=pad([f[:, None] for f in sf], ks)[..., 0], solid_g=pad(sg, ks), solid_n=np.array(sn),
        hollow_f=np.stack([pad([f[:, None] for f in r], kh)[..., 0] for r in hf]),
        hollow_g=np.stack([pad(r, kh) for r in hg]), hollow_n=np.array(hn),
        mesh=np.array([SOLID_N, HOLLOW_N, HOLLOW_LAYERS]),
    )
    print("wrote", modal.TABLE_PATH, f"{modal.TABLE_PATH.stat().st_size / 1024:.0f} KiB", nz)


if __name__ == "__main__":
    main()

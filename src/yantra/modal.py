"""Finite-element modal analysis of free cubes, and the precomputed mode tables used at runtime.

Model: 3D isotropic linear elasticity, triquadratic 27-node hexahedra (scikit-fem ElementHex2),
free-free boundary conditions. Solve K u = w^2 M u with shift-invert Lanczos (scipy eigsh) and drop
the six rigid-body modes (w ~ 0).

Everything is solved once on a unit cube with E = rho = 1. For geometrically similar bodies of the
same Poisson ratio the eigenfrequencies scale as

    f = f_hat * sqrt(E / rho) / L,

so one solve per (nu, wall/edge) covers every size and material. A hollow cube is only similar to
another hollow cube at the same wall/edge ratio, so hollow tables are tabulated over that ratio.

Gains: a tap on the cube is an impulse J along the inward normal of the -x face at y = 0 and
z/h in [-0.4, 0.4] (h = half edge; see `world.object_scene`). For mass-normalised modes phi_i the
modal velocity just after the impulse is J * phi_i(p).n, so the gain of mode i is |phi_i(p).n|.
Degenerate modes (the cube's symmetry gives 2- and 3-fold multiplicities) have no unique basis,
so each degenerate cluster is merged into one mode with gain sqrt(sum_k (phi_k(p).n)^2), which is
invariant to the basis the eigensolver happens to return. Acoustic radiation efficiency is ignored.

The runtime (`yantra.sound.modes_for`) only reads `data/modes.npz`; it never runs FEM.
Build the table with `uv run python scripts/build_modes.py`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

TABLE_PATH = Path(__file__).parent / "data" / "modes.npz"
STRIKE_Z = np.linspace(-0.4, 0.4, 9)  # strike heights on the -x face, as a fraction of the half edge
NU_GRID = np.array([0.20, 0.25, 0.30, 0.35, 0.40])
WALL_GRID = np.array([0.018, 0.022, 0.026, 0.030, 0.034, 0.038, 0.042])  # wall/edge ratios
N_MODES = 40  # non-rigid modes kept per shape (before merging degenerate ones)


# ------------------------------------------------------------------ FEM (build time only)
def _axis(n: int, wall: float | None, layers: int) -> np.ndarray:
    """Grid lines on [0, 1]: uniform, or `layers` thin cells of total width `wall` at each end."""
    if wall is None:
        return np.linspace(0.0, 1.0, n + 1)
    skin = np.linspace(0.0, wall, layers + 1)
    core = np.linspace(wall, 1.0 - wall, n + 1)
    return np.unique(np.concatenate([skin, core, 1.0 - skin]))


def cube_mesh(n: int = 8, wall: float | None = None, layers: int = 1):
    """Hex mesh of the unit cube [0,1]^3; with `wall`, a closed box with that wall thickness."""
    from skfem import MeshHex

    ax = _axis(n, wall, layers)
    mesh = MeshHex.init_tensor(ax, ax, ax)
    if wall is None:
        return mesh
    mid = mesh.p[:, mesh.t].mean(axis=1)  # element centroids
    inside = np.all((mid > wall) & (mid < 1.0 - wall), axis=0)
    return mesh.remove_elements(np.nonzero(inside)[0])


def assemble(mesh, nu: float, E: float = 1.0, rho: float = 1.0):
    """Stiffness K, mass M (scipy sparse) and the vector basis for linear elasticity."""
    from skfem import Basis, BilinearForm, ElementHex2, ElementVector, asm
    from skfem.helpers import dot
    from skfem.models.elasticity import lame_parameters, linear_elasticity

    basis = Basis(mesh, ElementVector(ElementHex2()), intorder=4)
    K = asm(linear_elasticity(*lame_parameters(E, nu)), basis)

    @BilinearForm
    def mass(u, v, w):
        return rho * dot(u, v)

    M = asm(mass, basis)
    return K.tocsc(), M.tocsc(), basis


def solve_modes(K, M, n_modes: int = N_MODES, n_rigid: int = 6):
    """Lowest `n_modes` elastic modes of a free body: (omega^2, mass-normalised vectors, rigid omega^2)."""
    from scipy.sparse.linalg import eigsh

    # K is singular (6 rigid-body modes), so shift slightly below zero to keep K - sigma*M definite.
    scale = K.diagonal().mean() / M.diagonal().mean()
    lam, vec = eigsh(K, k=n_modes + n_rigid, M=M, sigma=-1e-6 * scale, which="LM")
    order = np.argsort(lam)
    lam, vec = lam[order], vec[:, order]
    return lam[n_rigid:], vec[:, n_rigid:], lam[:n_rigid]


def strike_points(z_frac=STRIKE_Z) -> np.ndarray:
    """Points on the -x face (x = 0 of the unit cube), y at the centre, varying height."""
    return np.array([[0.0] * len(z_frac), [0.5] * len(z_frac), 0.5 + 0.5 * np.asarray(z_frac)])


def _x_displacement(basis, vec, pts):
    """x-component (face normal) of each mode shape at the strike points: (n_modes, n_points)."""
    xb, xdofs = basis.split_bases()[0], basis.split_indices()[0]
    return (xb.probes(pts) @ vec[xdofs]).T


def merge_degenerate(freq: np.ndarray, disp: np.ndarray, rtol: float = 1e-4):
    """Merge clusters of (numerically) equal frequencies; gain = rms-sum of the cluster's displacements."""
    out_f, out_g = [], []
    i = 0
    while i < len(freq):
        j = i + 1
        while j < len(freq) and freq[j] - freq[i] < rtol * freq[i]:
            j += 1
        out_f.append(freq[i:j].mean())
        out_g.append(np.sqrt((disp[i:j] ** 2).sum(axis=0)))
        i = j
    return np.array(out_f), np.array(out_g)


def cube_modes(nu: float, wall: float | None = None, n: int = 8, layers: int = 1, n_modes: int = N_MODES):
    """Normalised modes of a unit free cube (E = rho = 1).

    Returns f_hat (Hz * L / sqrt(E/rho)) and gains (n_merged_modes, len(STRIKE_Z)) for unit mass,
    plus the rigid-body eigenvalues (for checking).
    """
    mesh = cube_mesh(n, wall, layers)
    K, M, basis = assemble(mesh, nu)
    lam, vec, rigid = solve_modes(K, M, n_modes)
    f_hat = np.sqrt(np.maximum(lam, 0)) / (2 * np.pi)
    disp = np.abs(_x_displacement(basis, vec, strike_points()))
    mass = float(np.ones(M.shape[0]) @ (M @ np.ones(M.shape[0]))) / 3  # total mass (=volume)
    disp *= np.sqrt(mass)  # gains per unit total mass: invariant to the body's absolute mass
    f, g = merge_degenerate(f_hat, disp)
    return f, g, rigid


# ------------------------------------------------------------------ Ritz reference (validation)
def ritz_cube(nu: float, order: int = 10, n_modes: int = 20) -> np.ndarray:
    """Free-cube eigenfrequencies (f_hat, E = rho = L = 1) by the Rayleigh-Ritz method of
    Visscher et al., "On the normal modes of free vibration of inhomogeneous and anisotropic
    elastic objects", J. Acoust. Soc. Am. 90, 2154 (1991): displacement basis x^l y^m z^n,
    l+m+n <= order, integrated exactly over the cube. Independent of the FEM mesh; used in tests.
    """
    from scipy.linalg import eigh

    lam_, mu_ = nu / ((1 + nu) * (1 - 2 * nu)), 1 / (2 * (1 + nu))
    C = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C[i, j, k, l] = lam_ * (i == j) * (k == l) + mu_ * ((i == k) * (j == l) + (i == l) * (j == k))
    pw = [(a, b, c) for a in range(order + 1) for b in range(order + 1 - a) for c in range(order + 1 - a - b)]
    P = np.array(pw)  # (nb, 3)
    nb = len(pw)

    def I(e):  # integral of x^e0 y^e1 z^e2 over [-1/2, 1/2]^3
        e = np.asarray(e)
        val = np.where(e % 2 == 0, 2 * 0.5 ** (e + 1) / (e + 1), 0.0)
        return val.prod(axis=-1)

    S = P[:, None, :] + P[None, :, :]
    Mb = I(S)
    G = np.zeros((nb, nb, 3, 3))  # G[p,q,j,l] = int d_j phi_p d_l phi_q
    for j in range(3):
        for l in range(3):
            e = S.copy()
            e[..., j] -= 1
            e[..., l] -= 1
            coef = P[:, None, j] * P[None, :, l]
            ok = np.all(e >= 0, axis=-1) & (coef != 0)
            G[:, :, j, l] = np.where(ok, coef * I(np.maximum(e, 0)), 0.0)
    K = np.einsum("ijkl,pqjl->ipkq", C, G).reshape(3 * nb, 3 * nb)
    M = np.kron(np.eye(3), Mb).reshape(3, nb, 3, nb).reshape(3 * nb, 3 * nb)
    w2 = eigh(K, M, eigvals_only=True, subset_by_index=[0, n_modes + 5])
    return np.sqrt(np.maximum(w2[6:], 0)) / (2 * np.pi)


# ------------------------------------------------------------------ runtime tables
@lru_cache(maxsize=1)
def load_table():
    """Precomputed tables: dict with nu, wall grids and padded (freq, gain) arrays."""
    with np.load(TABLE_PATH) as z:
        return {k: z[k] for k in z.files}


def _interp_axis(grid: np.ndarray, x: float):
    x = float(np.clip(x, grid[0], grid[-1]))
    i = int(np.clip(np.searchsorted(grid, x) - 1, 0, len(grid) - 2))
    t = (x - grid[i]) / (grid[i + 1] - grid[i])
    return i, t


def normalised_modes(nu: float, wall_ratio: float | None):
    """(f_hat, gains[mode, strike_z]) interpolated in nu (and wall/edge for hollow cubes).

    Modes are matched by rank, which is exact at the grid points and smooth in between for the
    small grid spacing used (frequencies change by a few percent between neighbours).
    """
    tab = load_table()
    i, t = _interp_axis(tab["nu"], nu)
    if wall_ratio is None:
        F, G, cnt = tab["solid_f"], tab["solid_g"], tab["solid_n"]
        pick = lambda a: a[i] * (1 - t) + a[i + 1] * t
        k = int(min(cnt[i], cnt[i + 1]))
        return pick(F)[:k], pick(G)[:k]
    j, s = _interp_axis(tab["wall"], wall_ratio)
    F, G, cnt = tab["hollow_f"], tab["hollow_g"], tab["hollow_n"]
    w = {(i, j): (1 - t) * (1 - s), (i + 1, j): t * (1 - s), (i, j + 1): (1 - t) * s, (i + 1, j + 1): t * s}
    k = int(min(cnt[a, b] for a, b in w))
    f = sum(c * F[a, b, :k] for (a, b), c in w.items())
    g = sum(c * G[a, b, :k] for (a, b), c in w.items())
    return f, g

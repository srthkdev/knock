"""FEM modal analysis: rigid modes, accuracy against an independent Ritz solution, scaling, tables."""
import numpy as np
import pytest

from yantra import modal, sound
from yantra.materials import MATERIALS, WALL


def test_six_rigid_modes_are_found_and_removed():
    K, M, _ = modal.assemble(modal.cube_mesh(3), nu=0.3)
    lam, _, rigid = modal.solve_modes(K, M, n_modes=4)
    assert len(rigid) == 6
    assert np.abs(rigid).max() < 1e-8 * lam[0]  # rigid: w^2 ~ 0 relative to the first elastic mode
    assert lam.min() > 0


def _raw_fem(nu, n, k=20):
    K, M, _ = modal.assemble(modal.cube_mesh(n), nu)
    lam, _, _ = modal.solve_modes(K, M, n_modes=k)
    return np.sqrt(lam) / (2 * np.pi)


@pytest.mark.parametrize("nu", [0.23, 0.37])
def test_solid_cube_matches_ritz_reference(nu):
    """First 20 eigenfrequencies (with multiplicity) of a free cube: FEM (27-node hexes, 6^3) vs
    Rayleigh-Ritz with a degree-10 polynomial basis (Visscher et al. 1991), independent of the mesh."""
    fem = _raw_fem(nu, 6)
    ref = modal.ritz_cube(nu, order=10, n_modes=20)
    assert fem == pytest.approx(ref, rel=0.005)


def test_solid_cube_fem_converges():
    ref = modal.ritz_cube(0.3, order=10, n_modes=20)
    e4 = np.abs(_raw_fem(0.3, 4) / ref - 1).max()
    e6 = np.abs(_raw_fem(0.3, 6) / ref - 1).max()
    assert e6 < e4 < 0.02
    tab_f, _ = modal.normalised_modes(0.3, None)  # n = 8 table, degenerate modes merged
    assert tab_f[0] == pytest.approx(ref[0], rel=1e-3)


def test_frequency_scaling_law():
    """A cube of edge L, modulus E, density rho rings at f_hat * sqrt(E/rho) / L."""
    L, E, rho = 0.1, 2.0e11, 7800.0
    mesh = modal.cube_mesh(4).scaled([L, L, L])
    K, M, _ = modal.assemble(mesh, nu=0.3, E=E, rho=rho)
    lam, _, _ = modal.solve_modes(K, M, n_modes=6)
    f_phys = np.sqrt(lam) / (2 * np.pi)
    f_hat, _, _ = modal.cube_modes(0.3, None, n=4, n_modes=6)
    assert f_phys[0] == pytest.approx(f_hat[0] * np.sqrt(E / rho) / L, rel=1e-6)


def test_modes_for_scales_with_size_and_wave_speed():
    a = sound.modes_for("plastic", 0.08, False, np.random.default_rng(0), WALL)[0]
    b = sound.modes_for("plastic", 0.12, False, np.random.default_rng(0), WALL)[0]
    k = min(len(a), len(b))
    assert k > 3
    assert b[:k] * 0.12 == pytest.approx(a[:k] * 0.08, rel=1e-9)
    p = MATERIALS["plastic"]
    f_hat, _ = modal.normalised_modes(p["nu"], None)
    nearest = np.abs(a[:, None] / (f_hat[None, :] * p["c"] / 0.08) - 1).min(axis=1)
    assert nearest.max() < 0.011  # every mode is a table mode scaled by sqrt(E/rho)/L (+-1% jitter)


def test_hollow_fundamental_between_plate_bounds():
    """Lowest hollow-cube mode lies between a simply supported and a clamped square plate of the wall
    (Leissa, Vibration of Plates, NASA SP-160: w a^2 sqrt(rho h / D) = 19.74 SS, 35.99 clamped)."""
    for nu in (0.25, 0.35):
        for r in (0.022, 0.03, 0.038):
            f_hat, _ = modal.normalised_modes(nu, r)
            a = 1 - r  # mid-plane span of a wall
            plate = np.sqrt(r ** 2 / (12 * (1 - nu ** 2))) / a ** 2 / (2 * np.pi)
            assert 19.74 * plate < f_hat[0] < 35.99 * plate


def test_hollow_table_mesh_converged():
    f, _, _ = modal.cube_modes(0.3, 0.03, n=10)
    tab_f, _ = modal.normalised_modes(0.3, 0.03)  # n = 16 table
    assert np.abs(f[:8] / tab_f[:8] - 1).max() < 0.015


def test_hollow_table_interpolation_is_smooth():
    f0, _ = modal.normalised_modes(0.30, 0.026)
    f1, _ = modal.normalised_modes(0.30, 0.028)
    f2, _ = modal.normalised_modes(0.30, 0.030)
    assert np.all(f0[:5] < f1[:5]) and np.all(f1[:5] < f2[:5])  # thicker walls ring higher


@pytest.mark.parametrize("material", list(MATERIALS))
@pytest.mark.parametrize("hollow", [False, True])
def test_modes_for_is_physical(material, hollow):
    f, d, g = sound.modes_for(material, 0.1, hollow, np.random.default_rng(1), WALL)
    p = MATERIALS[material]
    if hollow:
        assert len(f) >= 10  # a few modes are nodal on the struck line
    if not len(f):  # solid steel/glass: all modes below 20 kHz are nodal on the struck line
        return
    assert np.all(np.diff(f) > 0) and f.max() < 20000
    assert d == pytest.approx(0.5 * (p["alpha"] + p["beta"] * (2 * np.pi * f) ** 2))
    assert g.max() == pytest.approx(1.0) and g.min() > 0


def test_hollow_rings_lower_and_longer_than_solid():
    for m, p in MATERIALS.items():
        fh, dh, _ = sound.modes_for(m, 0.1, True, np.random.default_rng(0), WALL)
        f_solid = modal.normalised_modes(p["nu"], None)[0][0] * p["c"] / 0.1
        assert fh[0] < 0.5 * f_solid
        assert dh[0] < 0.5 * (p["alpha"] + p["beta"] * (2 * np.pi * f_solid) ** 2)

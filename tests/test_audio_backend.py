"""The Rust audio engine (knock_audio) must agree with the numpy reference implementation.

Random draws differ between backends (PCG64 seeded from the numpy rng vs numpy's own stream),
so dry/wet audio is compared statistically; `features` is deterministic and must match exactly.
"""
import contextlib
import os

import numpy as np
import pytest

from yantra import sound, world
from yantra.agents.baselines import sounds_hollow
from yantra.level import Hidden, Obj
from yantra.materials import MATERIALS, WALL

ka = pytest.importorskip("knock_audio")
CLIP = world.CLIP_S


@contextlib.contextmanager
def backend(name):
    old = sound._ka
    sound._ka = ka if name == "rust" else None
    try:
        yield
    finally:
        sound._ka = old


def cube(material="wood", hollow=False, half=0.05):
    return Obj("A", half, "#c8553d", 0.0, Hidden(material, hollow, "empty", 0.3))


def dry(name, events, modes, seed):
    with backend(name):
        return sound.synthesize(events, *modes, np.random.default_rng(seed), CLIP)


@pytest.mark.skipif(os.environ.get("KNOCK_AUDIO", "").lower() == "python", reason="numpy backend forced")
def test_rust_backend_is_active_by_default():
    assert sound.BACKEND == "rust" and sound._ka is ka


def _peaks_agree(freqs, decay, gains, tau, isolated):
    events = [(0.05, 0.02, tau)]
    f = np.fft.rfftfreq(int(CLIP * sound.FS), 1 / sound.FS)
    spectra = {}
    for name in ("rust", "python"):
        # average a few strikes so strike-position gains do not hide any mode
        spectra[name] = np.mean([np.abs(np.fft.rfft(dry(name, events, (freqs, decay, gains), s))) for s in range(8)], 0)
    for k in isolated:
        # peak frequency = power-weighted centroid over the mode's half-power band (decay/pi Hz
        # wide), which is robust to the broadband contact click riding on wide, damped modes
        half = max(0.005 * freqs[k], decay[k] / np.pi)
        win = (f > freqs[k] - half) & (f < freqs[k] + half)
        peaks = {n: np.sum(f[win] * s[win] ** 2) / np.sum(s[win] ** 2) for n, s in spectra.items()}
        tol = max(2.0, 0.002 * freqs[k], 0.1 * decay[k] / np.pi)
        assert abs(peaks["rust"] - peaks["python"]) <= tol, (k, freqs[k], peaks)
        assert abs(peaks["rust"] - freqs[k]) <= 2 * tol, (k, freqs[k], peaks)
    # and the overall spectral shape matches
    band = (f > 60) & (f < 12000)
    assert np.corrcoef(np.log10(spectra["rust"][band]), np.log10(spectra["python"][band]))[0, 1] > 0.97


@pytest.mark.parametrize("material", list(MATERIALS))
def test_mode_frequencies_give_the_same_spectral_peaks(material):
    p = MATERIALS[material]
    freqs = np.array([310.0, 745.0, 1630.0, 3270.0, 6120.0])
    decay = 0.5 * (p["alpha"] + p["beta"] * (2 * np.pi * freqs) ** 2)  # Rayleigh damping (materials.py)
    _peaks_agree(freqs, decay, 1 / np.arange(1, 6), p["tau"], range(5))


def test_hollow_metal_modes_give_the_same_spectral_peaks():
    freqs, decay, gains = sound.modes_for("metal", 0.1, True, np.random.default_rng(1), wall=WALL)
    gaps = np.diff(freqs)
    isolated = [k for k in range(len(freqs))
                if (k == 0 or gaps[k - 1] > 0.04 * freqs[k]) and (k == len(freqs) - 1 or gaps[k] > 0.04 * freqs[k])]
    assert len(isolated) >= 2
    _peaks_agree(freqs, decay, gains, MATERIALS["metal"]["tau"], isolated)


def test_dry_and_wet_levels_match_statistically():
    modes = sound.modes_for("wood", 0.1, False, np.random.default_rng(0), wall=WALL)
    events = [(0.1, 0.02, 0.0004), (0.6, 0.01, 0.0004)]
    db = lambda xs: np.array([10 * np.log10(np.sum(x ** 2)) for x in xs])
    d_rs = db([dry("rust", events, modes, s) for s in range(100)])
    d_py = db([dry("python", events, modes, s) for s in range(100)])
    assert abs(d_rs.mean() - d_py.mean()) < 0.5
    x0 = dry("python", events, modes, 0)
    w_rs = db([ka.room_and_mic(x0, s) for s in range(100)])
    w_py = db([sound._room_and_mic_py(x0, np.random.default_rng(s)) for s in range(100)])
    assert abs(w_rs.mean() - w_py.mean()) < 1.5
    assert abs(w_rs.std() - w_py.std()) < 1.0
    # absolute noise floor: same level range, with and without rustle
    z = np.zeros(int(CLIP * sound.FS))
    for rustle in (False, True):
        f_rs = db([ka.room_and_mic(z, s, rustle) for s in range(60)])
        f_py = db([sound._room_and_mic_py(z, np.random.default_rng(s), rustle) for s in range(60)])
        assert abs(f_rs.mean() - f_py.mean()) < 1.5, rustle


def test_features_are_identical_on_the_same_audio():
    for i, (mat, hollow) in enumerate([(m, h) for m in MATERIALS for h in (False, True)]):
        with backend("python"):
            x, _ = world.simulate_tap(cube(mat, hollow), seed=i)
        assert ka.features(x) == sound._features_py(x)
    with backend("python"):
        x, _ = world.simulate_shake(Obj("A", 0.05, "#c8553d", 0.0, Hidden("plastic", True, "marble", 0.3)), seed=2)
    assert ka.features(x) == sound._features_py(x)


@pytest.mark.parametrize("material,hollow", [("metal", True), ("glass", False), ("wood", False), ("plastic", True)])
def test_tap_features_agree_statistically(material, hollow):
    o = cube(material, hollow)
    stats = {}
    for name in ("rust", "python"):
        with backend(name):
            fs = [world.simulate_tap(o, seed=s)[1] for s in range(30)]
        stats[name] = {k: np.mean([f[k] for f in fs]) for k in ("ring_time_s", "loudness_db_over_floor")}
    r, p = stats["rust"], stats["python"]
    assert abs(r["ring_time_s"] - p["ring_time_s"]) < 0.25 * p["ring_time_s"] + 0.02, stats
    assert abs(r["loudness_db_over_floor"] - p["loudness_db_over_floor"]) < 3.0, stats


@pytest.mark.parametrize("name", ["rust", "python"])
def test_tap_separates_hollow_from_solid(name):
    # the FEM-based tap rule (baselines.sounds_hollow) is right ~97% of the time on either backend
    rng = np.random.default_rng(0)
    wrong = []
    with backend(name):
        for i in range(80):
            hollow = bool(i % 2)
            o = cube(str(rng.choice(list(MATERIALS))), hollow, half=float(rng.uniform(0.04, 0.07)))
            _, f = world.simulate_tap(o, seed=i)
            if sounds_hollow(f) != hollow:
                wrong.append((i, o.hidden.material, hollow, f["dominant_pitch_hz"], f["loudness_db_over_floor"]))
    assert len(wrong) <= 6, wrong


def test_rust_backend_is_deterministic():
    o = cube("glass", True)
    with backend("rust"):
        a, fa = world.simulate_tap(o, seed=5)
        b, fb = world.simulate_tap(o, seed=5)
        c, _ = world.simulate_tap(o, seed=6)
        s1, _ = world.simulate_shake(o, seed=1)
        s2, _ = world.simulate_shake(o, seed=1)
    assert np.array_equal(a, b) and fa == fb
    assert not np.array_equal(a, c)
    assert np.array_equal(s1, s2)
    ev = np.array([[0.1, 0.01, 0.0005]])
    m = np.array([400.0, 900.0]), np.array([10.0, 20.0]), np.array([1.0, 0.5])
    assert np.array_equal(ka.synthesize(ev, *m, 42, 1.0), ka.synthesize(ev, *m, 42, 1.0))
    assert not np.array_equal(ka.synthesize(ev, *m, 42, 1.0), ka.synthesize(ev, *m, 43, 1.0))


def test_rust_accepts_python_call_shapes():
    rng = np.random.default_rng(0)
    modes = sound.modes_for("metal", 0.1, True, rng, wall=WALL)
    with backend("rust"):
        assert sound.synthesize([], *modes, rng, 0.5).shape == (22050,)
        x = sound.synthesize([(0.1, 0.01, 1e-4)], *modes, rng, 0.5)
        y = sound.room_and_mic(x, rng, rustle=True)
        f = sound.features(y)
    assert y.dtype == np.float64 and y.shape == x.shape
    assert isinstance(f["distinct_hits"], int) and isinstance(f["dominant_pitch_hz"], int)
    with pytest.raises(ValueError):
        ka.synthesize(np.zeros((1, 3)), np.ones(2), np.ones(3), np.ones(2), 0, 0.5)

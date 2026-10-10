"""Benchmark the Rust (knock_audio) vs numpy audio backends: time to render 400 clips.

    uv run python scripts/bench_audio.py [--n 400]
"""
import argparse
import time

import numpy as np

from yantra import sound, world
from yantra.level import Hidden, Obj
from yantra.materials import MATERIALS, WALL


def clips(n):
    """Real contact events (taps and shakes) and modes for n varied objects."""
    rng = np.random.default_rng(0)
    out = []
    captured = []
    orig = sound.synthesize
    sound.synthesize = lambda ev, *a, **k: (captured.append(ev), orig(ev, *a, **k))[1]
    try:
        for i in range(16):  # simulate a pool of contact patterns, then reuse them
            o = Obj("A", 0.05, "#c8553d", 0.0, Hidden(str(rng.choice(list(MATERIALS))), bool(i % 2),
                                                      "marble" if i % 4 == 1 else "empty", 0.3))
            (world.simulate_shake if i % 4 == 1 else world.simulate_tap)(o, seed=i)
    finally:
        sound.synthesize = orig
    for i in range(n):
        mat, hollow = str(rng.choice(list(MATERIALS))), bool(i % 2)
        modes = sound.modes_for(mat, float(rng.uniform(0.08, 0.14)), hollow, rng, wall=WALL)
        out.append((captured[i % len(captured)], modes, i % 4 == 1))
    return out


def run(work, backend, stage):
    ka = sound._ka
    sound._ka = ka if backend == "rust" else None
    try:
        t0 = time.perf_counter()
        for i, (ev, modes, rustle) in enumerate(work):
            rng = np.random.default_rng(i)
            x = sound.synthesize(ev, *modes, rng, world.CLIP_S)
            if stage == "full":
                sound.features(sound.room_and_mic(x, rng, rustle=rustle))
        return time.perf_counter() - t0
    finally:
        sound._ka = ka


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    n = ap.parse_args().n
    if sound._ka is None:
        raise SystemExit("knock_audio is not importable; run `uv sync` with cargo on PATH")
    work = clips(n)
    print(f"{n} clips of {world.CLIP_S} s, {np.mean([len(w[0]) for w in work]):.1f} contact events/clip")
    for stage in ("synthesize", "full"):
        t_py, t_rs = run(work, "python", stage), run(work, "rust", stage)
        label = "synthesize" if stage == "synthesize" else "synthesize+room_and_mic+features"
        print(f"{label:34s} numpy {t_py:7.2f} s   rust {t_rs:6.2f} s   speedup {t_py / t_rs:5.1f}x")


if __name__ == "__main__":
    main()

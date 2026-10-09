"""Contact events -> audio via modal synthesis, plus features and spectrograms.

`synthesize`, `room_and_mic` and `features` run on the Rust engine (`knock_audio`, built from
crates/knock-audio) when it is importable; set KNOCK_AUDIO=python to force the numpy reference
implementations below. The Rust engine draws its randomness from a u64 seed taken from the
caller's numpy generator, so results stay deterministic per probe seed (but differ sample-wise
from the numpy path).
"""
import os

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

from .materials import MATERIALS

FS = 44100
REF_LEVEL = 1e-3  # typical rms of a tap at the mic; sets the absolute noise floor
# Room reverb: exponential noise tail with this time constant (s) behind a direct path. Short,
# so the room does not cap modal decay (a 0.05 s tail limited every decay to ~20/s and
# swamped wood/plastic damping vs real recordings). The IR is normalised to a fixed L2 gain,
# so a shorter tail carries proportionally less energy relative to the direct sound while the
# overall wet level (and hence loudness over the absolute noise floor) stays calibrated.
REVERB_TAIL_S = 0.01
REVERB_IR_GAIN = 0.14  # L2 norm of the room IR (~ that of the original 0.05 s tail)

_ka = None
if os.environ.get("KNOCK_AUDIO", "rust").lower() != "python":
    try:
        import knock_audio as _ka
    except ImportError:  # extension not built: fall back to numpy
        _ka = None
BACKEND = "rust" if _ka is not None else "python"


def _seed(rng) -> int:
    return int(rng.integers(2**63))


def modes_for(material, size, hollow, rng, wall, n=16):
    """(freqs, decay_rates, gains) for a cube of edge `size` metres."""
    p = MATERIALS[material]
    freqs, gains = [], []
    if hollow:
        # thin-plate bending modes of the walls
        k = (np.pi / 2) * wall * p["c"] / np.sqrt(12)
        for m in range(1, 6):
            for n_ in range(1, 6):
                aspect = rng.uniform(0.85, 1.15)
                freqs.append(k * ((m / size) ** 2 + (n_ / (size * aspect)) ** 2))
                gains.append(1.0 / (m * n_))
    else:
        # solid block: stiff, high, fast-damped modes plus a low thump into the table
        f0 = p["c"] / (2 * size)
        for i in range(1, 8):
            freqs.append(f0 * i * rng.uniform(0.97, 1.03))
            gains.append(0.6 / i)
        freqs.append(rng.uniform(120, 220))
        gains.append(1.5)
    freqs = np.array(freqs) * rng.uniform(0.98, 1.02, len(freqs))
    gains = np.array(gains)
    keep = (freqs > 60) & (freqs < 16000)
    freqs, gains = freqs[keep], gains[keep]
    order = np.argsort(freqs)[:n]
    freqs, gains = freqs[order], gains[order]
    decay = p["a"] + p["b"] * freqs
    if not hollow:
        decay = decay * 4.0
    return freqs, decay, gains


def synthesize(events, freqs, decay, gains, rng, seconds):
    """events: list of (time_s, impulse_Ns, contact_time_s)."""
    if _ka is not None:
        ev = np.asarray(events, dtype=np.float64).reshape(-1, 3)
        f, d, g = (np.ascontiguousarray(a, dtype=np.float64) for a in (freqs, decay, gains))
        return _ka.synthesize(ev, f, d, g, _seed(rng), float(seconds))
    return _synthesize_py(events, freqs, decay, gains, rng, seconds)


def _synthesize_py(events, freqs, decay, gains, rng, seconds):
    n = int(seconds * FS)
    nm = len(freqs)
    exc = np.zeros((nm, n))
    click = np.zeros(n)
    idx = np.arange(1, nm + 1)
    for t0, J, tau in events:
        w = max(2, int(tau * FS))
        i = int(t0 * FS)
        if i + w >= n:
            continue
        pulse = np.sin(np.pi * np.arange(w) / w)
        pulse /= pulse.sum()
        u, v = rng.uniform(0.1, 0.9, 2)
        pos_gain = np.abs(np.sin(np.pi * u * idx) * np.sin(np.pi * v * idx * 0.7 + 0.3))
        exc[:, i:i + w] += np.outer(gains * (0.3 + pos_gain) * J, pulse)
        click[i:i + w] += J * rng.normal(0, 1, w) * 0.15
    out = click
    for k in range(nm):
        r = np.exp(-decay[k] / FS)
        wk = 2 * np.pi * freqs[k] / FS
        out = out + lfilter([np.sin(wk)], [1, -2 * r * np.cos(wk), r * r], exc[k])
    return out


def room_and_mic(x, rng, rustle=False, tail_s=REVERB_TAIL_S):
    """Short room reverb, mic gain jitter, a quiet fixed noise floor, optional grip rustle."""
    if _ka is not None:
        return _ka.room_and_mic(np.ascontiguousarray(x, dtype=np.float64), _seed(rng), bool(rustle), float(tail_s))
    return _room_and_mic_py(x, rng, rustle, tail_s)


def _room_and_mic_py(x, rng, rustle=False, tail_s=REVERB_TAIL_S):
    L = max(1, int(min(0.25, 10 * tail_s) * FS))  # 10 time constants (-87 dB), at most 0.25 s
    ir = rng.normal(0, 1, L) * np.exp(-np.arange(L) / (tail_s * FS))
    ir[0] = 25.0
    ir *= REVERB_IR_GAIN / np.sqrt(np.sum(ir ** 2))
    y = np.convolve(x, ir)[: len(x)]
    y *= 10 ** (rng.uniform(-4, 4) / 20)
    noise = sosfilt(butter(2, 2000, "low", fs=FS, output="sos"), rng.normal(0, 1, len(y)))
    noise *= REF_LEVEL * 10 ** (-rng.uniform(32, 42) / 20) / np.sqrt(np.mean(noise ** 2))
    if rustle:
        t = np.arange(len(y)) / FS
        r = sosfilt(butter(2, [200, 3000], "band", fs=FS, output="sos"), rng.normal(0, 1, len(y)))
        r *= 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * rng.uniform(3, 5) * t))
        r *= REF_LEVEL * 10 ** (-rng.uniform(34, 44) / 20) / np.sqrt(np.mean(r ** 2))
        noise = noise + r
    return y + noise


def features(x):
    """Compact acoustic summary given to agents and used by baselines."""
    if _ka is not None:
        return _ka.features(np.ascontiguousarray(x, dtype=np.float64))
    return _features_py(x)


def _features_py(x):
    hop = int(0.005 * FS)
    frames = x[: len(x) // hop * hop].reshape(-1, hop)
    env = 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-12)
    floor = float(np.percentile(env, 10))
    peak_i = int(np.argmax(env))
    loudness = float(env[peak_i] - floor)
    target = max(env[peak_i] - 25, floor + 3)
    after = np.where(env[peak_i:] < target)[0]
    ring = (after[0] if len(after) else len(env) - peak_i) * 0.005
    rise = env[3:] - env[:-3]
    on = np.where((rise > 8) & (env[3:] > floor + 12))[0]
    hits = int(np.sum(np.diff(np.concatenate([[-99], on])) > 4))
    seg = x[peak_i * hop: peak_i * hop + int(0.2 * FS)]
    # one-sided (decaying half-Hann) taper keeps the onset, where fast-decaying solid modes live;
    # a symmetric Hann starting at the hit would erase them now that the room tail is short
    spec = np.abs(np.fft.rfft(seg * np.hanning(2 * len(seg))[len(seg):])) + 1e-12
    f = np.fft.rfftfreq(len(seg), 1 / FS)
    centroid = float((f * spec).sum() / spec.sum())
    band = f > 80
    peak_hz = float(f[band][np.argmax(spec[band])])
    return {
        "ring_time_s": round(float(ring), 3),
        "distinct_hits": hits,
        "brightness_hz": round(centroid),
        "dominant_pitch_hz": round(peak_hz),
        "loudness_db_over_floor": round(loudness, 1),
    }


def save_spectrogram(x, path, title=""):
    """Agent-facing spectrogram: dB-normalized, fixed range so clips are comparable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
    spec, freqs, times, _ = ax.specgram(x, NFFT=1024, Fs=FS, noverlap=896, cmap="magma",
                                        scale="dB", mode="magnitude")
    ax.cla()
    db = 20 * np.log10(spec + 1e-12) - 20 * np.log10(REF_LEVEL)
    ax.pcolormesh(times, freqs, db, vmin=-50, vmax=10, cmap="magma", shading="auto")
    ax.set_yscale("symlog", linthresh=500)
    ax.set_ylim(50, 12000)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Hz")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_wav(x, path, gain=1.0):
    """16-bit wav; `gain` is shared across clips so loudness stays comparable."""
    import wave

    y = np.clip(x * gain, -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes((y * 32767).astype(np.int16).tobytes())

"""Build a listening page of probe sounds, all at one shared gain (no per-clip normalisation)."""
import base64
import io
import sys
import wave

import numpy as np

from knock import world
from knock.level import Hidden, Obj
from knock.materials import MATERIALS

GAIN = None  # set from the loudest clip
out = sys.argv[1] if len(sys.argv) > 1 else "listen.html"
rng = np.random.default_rng(11)


def wav_b64(x):
    y = np.clip(x * GAIN, -1, 1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100)
        w.writeframes((y * 32767).astype(np.int16).tobytes())
    return base64.b64encode(buf.getvalue()).decode()


CLIPS = []


def clip(label, x):
    CLIPS.append((label, x))
    return len(CLIPS) - 1


def render_clip(label, x):
    rms = 20 * np.log10(np.sqrt(np.mean((x * GAIN) ** 2)) + 1e-12)
    return (f'<div class="clip"><span>{label}</span><audio controls preload="none" '
            f'src="data:audio/wav;base64,{wav_b64(x)}"></audio><small>{rms:.0f} dBFS rms</small></div>')


sections = []
rows = []
for mat in MATERIALS:
    for hollow in (True, False):
        o = Obj("A", float(rng.uniform(0.045, 0.065)), "#ffffff", 0, Hidden(mat, hollow, "empty", 0.3))
        x, _ = world.simulate_tap(o, seed=int(rng.integers(1 << 30)))
        rows.append(clip(f"tap: {mat}, {'hollow' if hollow else 'solid'}", x))
sections.append(("Tap", rows))
rows = []
for cont in ["empty", "weight", "foam", "marble", "beads"]:
    for k in range(2):
        o = Obj("A", float(rng.uniform(0.045, 0.065)), "#ffffff", 0,
                Hidden(str(rng.choice(["wood", "plastic", "metal"])), True, cont, 0.3))
        x, _ = world.simulate_shake(o, seed=int(rng.integers(1 << 30)))
        rows.append(clip(f"shake: {cont} ({k + 1})", x))
sections.append(("Shake", rows))

GAIN = 0.9 / max(np.abs(x).max() for _, x in CLIPS)
print(f"shared gain {GAIN:.0f}x")
body = "".join(f"<section><h2>{t}</h2><div class='grid'>{''.join(render_clip(*CLIPS[i]) for i in r)}</div></section>"
               for t, r in sections)
html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Knock Sounds v2</title><style>
:root{{--bg:#fafaf7;--fg:#1b1b1b;--muted:#666;--card:#fff;--line:#e3e2dc}}
@media (prefers-color-scheme:dark){{:root{{--bg:#141414;--fg:#eee;--muted:#9a9a9a;--card:#1d1d1d;--line:#2c2c2c}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif}}
main{{max-width:900px;margin:auto;padding:24px 16px}} .lead{{color:var(--muted)}}
section{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:16px 0}}
h2{{margin:0 0 10px;font-size:18px}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}}
.clip{{display:flex;flex-direction:column;gap:4px}} .clip span{{font-weight:600;font-size:13px}} small{{color:var(--muted)}} audio{{width:100%}}
</style></head><body><main><h1>Knock sounds v2</h1>
<p class="lead">Same engine the benchmark uses. All clips share one volume level, so an empty box now sounds as quiet as it really is.</p>
{body}</main></body></html>"""
open(out, "w").write(html)
print("wrote", out)

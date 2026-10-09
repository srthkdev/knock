"""Replay episode traces as narrated MP4 video with the real probe sounds.

Every motion is re-simulated with the same physics code that scored the episode, and every
sound is re-synthesised with the same seed, so the video shows exactly what the agent got.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import sound, world
from .env import PUSH_SECONDS, probe_seed
from .level import Level
from .render import _project

FPS = 30
W, H = 1280, 720
VIEW_W = 860
PANEL_W = W - VIEW_W
AUDIO_GAIN = 73.0

BG = (250, 249, 245)
INK = (28, 28, 28)
MUTED = (110, 110, 105)
ACCENT = (47, 111, 94)
WARN = (190, 70, 50)
CARD = (255, 255, 255)
LINE = (226, 224, 216)


def _font(size, bold=False):
    path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)


F_TITLE, F_H, F_BODY, F_SMALL, F_BIG = _font(22, True), _font(17, True), _font(15), _font(13), _font(44, True)


# ------------------------------------------------------------------ 3D scene
def _rgba(hex_color):
    h = hex_color.lstrip("#")
    return " ".join(f"{int(h[i:i + 2], 16) / 255:.3f}" for i in (0, 2, 4)) + " 1"


def scene_xml(level: Level) -> str:
    t = level.task
    span = (len(level.objects) - 1) * 0.35 / 2 + 0.18
    bodies, lanes = [], []
    for o in level.objects:
        bodies.append(f'<body name="o_{o.id}" mocap="true" pos="0 {o.y} {o.half}">'
                      f'<geom type="box" size="{o.half} {o.half} {o.half}" rgba="{_rgba(o.color)}" material="obj"/></body>')
        lanes.append(f'<geom type="box" pos="0.65 {o.y} 0.0006" size="0.8 0.003 0.0005" rgba="0.72 0.70 0.66 1"/>')
    return f"""<mujoco>
      <visual><headlight ambient="0.38 0.38 0.38" diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05"/>
        <quality shadowsize="8192" offsamples="8"/><global offwidth="{VIEW_W}" offheight="{H}" fovy="38"/></visual>
      <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.97 0.97 0.95" rgb2="0.86 0.88 0.90" width="256" height="256"/>
        <texture name="t" type="2d" builtin="checker" rgb1="0.88 0.85 0.79" rgb2="0.86 0.83 0.77" width="512" height="512"/>
        <material name="table" texture="t" texrepeat="10 10" reflectance="0.04"/>
        <material name="obj" specular="0.35" shininess="0.5" reflectance="0.03"/>
      </asset>
      <worldbody>
        <light pos="0.3 -0.8 2.4" dir="0 0.3 -1" diffuse="0.75 0.75 0.75" castshadow="true"/>
        <geom type="plane" size="3 3 .1" material="table"/>
        {''.join(lanes)}
        <geom type="box" pos="{t.target_x} 0 0.0016" size="{t.tol} {span} 0.0004" rgba="0.30 0.72 0.52 0.5"/>
        <geom type="box" pos="0 0 0.0008" size="0.003 {span} 0.0005" rgba="0.3 0.3 0.3 1"/>
        {''.join(bodies)}
        <body name="striker" mocap="true" pos="0 0 -1"><geom type="sphere" size="0.012" rgba="0.75 0.76 0.8 1" material="obj"/></body>
        <camera name="cam" pos="{_cam_pos(level)}" xyaxes="{_cam_axes(level)}"/>
      </worldbody></mujoco>"""


def _cam_frame(level: Level):
    """Camera that frames the start line, every lane and the target band."""
    far = max(level.task.target_x + level.task.tol, 0.6)
    look = np.array([far * 0.5, 0.0, 0.0])
    dist = 1.25 + 0.75 * far
    eye = look + dist * np.array([-0.52, -0.62, 0.58])
    fwd = look - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return eye, right, up


def _cam_pos(level):
    return " ".join(f"{v:.4f}" for v in _cam_frame(level)[0])


def _cam_axes(level):
    _, r, u = _cam_frame(level)
    return " ".join(f"{v:.4f}" for v in (*r, *u))


class Stage:
    def __init__(self, level: Level):
        self.level = level
        self.m = mujoco.MjModel.from_xml_string(scene_xml(level))
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, height=H, width=VIEW_W)
        self.cam = self.m.camera("cam").id
        self.mid = {o.id: self.m.body_mocapid[self.m.body(f"o_{o.id}").id] for o in level.objects}
        self.sid = self.m.body_mocapid[self.m.body("striker").id]
        self.reset()

    def reset(self):
        for o in self.level.objects:
            self.place(o.id, (0.0, o.y, o.half))
        self.striker(None)

    def place(self, oid, pos, quat=(1, 0, 0, 0)):
        self.d.mocap_pos[self.mid[oid]] = pos
        self.d.mocap_quat[self.mid[oid]] = quat

    def striker(self, pos):
        self.d.mocap_pos[self.sid] = pos if pos is not None else (0, 0, -1)

    def frame(self) -> Image.Image:
        mujoco.mj_forward(self.m, self.d)
        self.r.update_scene(self.d, camera="cam")
        img = Image.fromarray(self.r.render())
        draw = ImageDraw.Draw(img)
        for o in self.level.objects:
            p = self.d.mocap_pos[self.mid[o.id]] + [0, 0, o.half + 0.035]
            uv = _project(self.m, self.d, self.cam, p, img.size)
            if uv:
                draw.text(uv, o.id, fill=INK, font=_font(24, True), anchor="ms")
        t = self.level.task
        uv = _project(self.m, self.d, self.cam, np.array([t.target_x, -((len(self.level.objects) - 1) * 0.175 + 0.24), 0]), img.size)
        if uv:
            draw.text(uv, "target", fill=ACCENT, font=_font(20, True), anchor="mm")
        return img


# ------------------------------------------------------------------ panel
@dataclass
class PanelState:
    steps: list[str]
    current: int  # index into steps being shown, -1 = none
    reasoning: str
    probes_left: int
    spectrogram: str | None = None
    banner: tuple[str, tuple] | None = None


def draw_panel(title: str, level: Level, st: PanelState) -> Image.Image:
    img = Image.new("RGB", (PANEL_W, H), BG)
    d = ImageDraw.Draw(img)
    x, y = 22, 20
    d.text((x, y), title, fill=INK, font=F_TITLE)
    y += 32
    t = level.task
    goal = (f"Flick {t.object_id} into the green band" if t.kind == "slide"
            else f"Find the object that {_prop_text(t.property)}, flick it into the green band")
    d.text((x, y), f"{level.id}  ·  probes left: {st.probes_left}", fill=MUTED, font=F_SMALL)
    y += 22
    for line in textwrap.wrap(goal, 46):
        d.text((x, y), line, fill=INK, font=F_BODY)
        y += 20
    y += 10
    d.text((x, y), "Experiments", fill=INK, font=F_H)
    y += 26
    for i, s in enumerate(st.steps):
        col = ACCENT if i == st.current else (INK if i < st.current or st.current < 0 else MUTED)
        for j, line in enumerate(textwrap.wrap(s, 48)):
            d.text((x + (0 if j == 0 else 14), y), line, fill=col, font=F_BODY if i == st.current else F_SMALL)
            y += 19
        y += 3
    y += 8
    # reasoning card
    card_h = 150
    d.rounded_rectangle((x - 6, y, PANEL_W - 16, y + card_h), 10, fill=CARD, outline=LINE)
    d.text((x + 6, y + 10), "Agent reasoning", fill=MUTED, font=F_SMALL)
    yy = y + 30
    for line in textwrap.wrap(st.reasoning or "…", 50)[:6]:
        d.text((x + 6, yy), line, fill=INK, font=F_SMALL)
        yy += 18
    y += card_h + 12
    if st.spectrogram == "silence":
        d.text((x, y + 4), "(silence: nothing audible)", fill=MUTED, font=F_BODY)
    elif st.spectrogram and Path(st.spectrogram).exists():
        sp = Image.open(st.spectrogram).convert("RGB")
        sp = sp.resize((PANEL_W - 38, int((PANEL_W - 38) * sp.height / sp.width)))
        if y + sp.height < H - 10:
            img.paste(sp, (x - 4, y))
    if st.banner:
        text, col = st.banner
        d.rounded_rectangle((x - 6, H - 92, PANEL_W - 16, H - 18), 12, fill=col)
        d.text(((PANEL_W - 10) // 2, H - 55), text, fill=(255, 255, 255), font=_font(26, True), anchor="mm")
    return img


def _prop_text(p):
    return {"hollow": "is hollow", "solid": "is solid", "rattles": "rattles inside",
            "heaviest": "is heaviest", "lightest": "is lightest"}[p]


def _step_text(action, obs):
    if action.get("type") == "flick":
        return f"flick {action['object']} with {float(action['impulse_Ns']):.2f} N·s"
    p, o = action["probe"], action["object"]
    if "error" in obs:
        return f"{p} {o}: {obs['error']}"
    if p == "push":
        return f"push {o} with {obs['force_N']:.2f} N → moved {obs['moved_mm']:.0f} mm"
    if p == "lift":
        return f"lift {o} → {obs['weight_g']:.0f} g"
    s = obs["sound"]
    if p == "tap":
        return f"tap {o} → pitch {s['dominant_pitch_hz']} Hz, rings {s['ring_time_s']:.2f} s"
    return f"shake {o} → {s['distinct_hits']} hits, {s['loudness_db_over_floor']:.0f} dB"


# ------------------------------------------------------------------ episode timeline
class Recorder:
    def __init__(self, title):
        self.title = title
        self.frames: list[Image.Image] = []
        self.audio_events: list[tuple[int, np.ndarray]] = []  # (frame index, samples)

    def add(self, view: Image.Image, panel: Image.Image, n=1):
        canvas = Image.new("RGB", (W, H), BG)
        canvas.paste(view, (0, 0))
        canvas.paste(panel, (VIEW_W, 0))
        for _ in range(n):
            self.frames.append(canvas)

    def sound(self, samples, at_frame=None):
        self.audio_events.append((len(self.frames) if at_frame is None else at_frame, samples))


def _ease(u):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(u, 0, 1))


def replay_episode(trace: dict, rec: Recorder, model_label: str):
    level = Level.from_json(json.dumps(trace["level"]))
    stage = Stage(level)
    replies = [c["reply"] for c in trace.get("calls", []) if c.get("reply")]
    steps = [h for h in trace["history"] if h["event"] == "step"]
    texts = [_step_text(h["action"], h["obs"]) for h in steps]
    budget = level.task.budget
    title = f"Knock · {model_label}"

    def panel(i, reasoning, left, spec=None, banner=None, shown=None):
        shown = texts[: (shown if shown is not None else i + 1)]
        return draw_panel(title, level, PanelState(shown, i, reasoning, left, spec, banner))

    left = budget
    rec.add(stage.frame(), panel(-1, "Reading the level…", left, shown=0), FPS * 2)
    for i, h in enumerate(steps):
        a, obs = h["action"], h["obs"]
        why = replies[i]["reasoning"] if i < len(replies) else ""
        stage.reset()
        rec.add(stage.frame(), panel(i, why, left, shown=i), int(FPS * 1.2))  # thinking
        o = level.obj(a["object"]) if str(a.get("object", "")).upper() in {x.id for x in level.objects} else None
        if "error" in obs or o is None:
            rec.add(stage.frame(), panel(i, why, left), FPS)
            continue
        if a["type"] == "flick":
            _anim_flick(stage, rec, level, o, a, trace["result"], lambda b=None: panel(i, why, left, banner=b))
            break
        left -= 1
        P = lambda spec=None: panel(i, why, left, spec)
        if a["probe"] == "push":
            _anim_push(stage, rec, o, obs["force_N"], P)
        elif a["probe"] == "lift":
            _anim_lift(stage, rec, o, P)
        else:
            n = trace["history"].index(h)
            spec = Path(trace.get("_dir", ".")) / f"{n:02d}_{a['probe']}_{o.id}.png"
            prior = sum(1 for g in steps[:i] if g["action"].get("probe") == a["probe"] and g["action"]["object"] == o.id)
            seed = probe_seed(level, o.id, a["probe"], prior)
            sim = world.simulate_tap if a["probe"] == "tap" else world.simulate_shake
            x, _ = sim(o, seed=seed)
            if obs["sound"]["loudness_db_over_floor"] < 10:
                spec = "silence"
            if a["probe"] == "tap":
                _anim_tap(stage, rec, o, x, lambda: P(str(spec)))
            else:
                _anim_shake(stage, rec, o, x, lambda: P(str(spec)))


def _anim_push(stage, rec, o, force, P):
    traj = []
    world.simulate_push(o, force, PUSH_SECONDS, record=traj, fps=FPS)
    x0 = traj[0][1][0] if traj else 0
    p = P()
    step = max(1, len(traj) // (FPS * 4))  # play long slides faster
    for _, pos, q in traj[::step]:
        stage.place(o.id, (pos[0] - x0, o.y, pos[2]), q)
        rec.add(stage.frame(), p)
    rec.add(stage.frame(), p, int(FPS * 0.8))
    xe = traj[-1][1][0] - x0 if traj else 0
    for u in np.linspace(0, 1, int(FPS * 0.5)):
        stage.place(o.id, (xe * (1 - _ease(u)), o.y, o.half))
        rec.add(stage.frame(), p)


def _anim_lift(stage, rec, o, P):
    p = P()
    for u in np.linspace(0, 1, int(FPS * 0.6)):
        stage.place(o.id, (0, o.y, o.half + 0.12 * _ease(u)))
        rec.add(stage.frame(), p)
    rec.add(stage.frame(), p, int(FPS * 1.0))
    for u in np.linspace(1, 0, int(FPS * 0.5)):
        stage.place(o.id, (0, o.y, o.half + 0.12 * _ease(u)))
        rec.add(stage.frame(), p)


def _anim_tap(stage, rec, o, x, P):
    p = P()
    z = o.half
    for u in np.linspace(0, 1, int(FPS * 0.4)):
        stage.striker((-o.half - 0.18 * (1 - u) - 0.012, o.y, z))
        rec.add(stage.frame(), p)
    rec.sound(x[int(0.02 * sound.FS):])
    for u in np.linspace(0, 1, int(FPS * 0.3)):
        stage.striker((-o.half - 0.012 - 0.06 * u, o.y, z))
        rec.add(stage.frame(), p)
    stage.striker(None)
    rec.add(stage.frame(), p, int(FPS * 1.3))


def _anim_shake(stage, rec, o, x, P):
    p = P()
    lift = 0.14
    for u in np.linspace(0, 1, int(FPS * 0.4)):
        stage.place(o.id, (0, o.y, o.half + lift * _ease(u)))
        rec.add(stage.frame(), p)
    rec.sound(x)
    n = int(len(x) / sound.FS * FPS)
    for k in range(n):
        t = k / FPS
        stage.place(o.id, (0.03 * np.sin(2 * np.pi * 4 * t), o.y, o.half + lift + 0.012 * np.sin(4 * np.pi * 4 * t)))
        rec.add(stage.frame(), p)
    for u in np.linspace(1, 0, int(FPS * 0.4)):
        stage.place(o.id, (0, o.y, o.half + lift * _ease(u)))
        rec.add(stage.frame(), p)


def _anim_flick(stage, rec, level, o, a, result, P):
    traj = []
    world.simulate_flick(o, float(np.clip(a["impulse_Ns"], 0.02, 6.0)), record=traj, fps=FPS)
    x0 = traj[0][1][0] if traj else 0
    p = P()
    for u in np.linspace(0, 1, int(FPS * 0.35)):
        stage.striker((-o.half - 0.012 - 0.12 * (1 - u), o.y, o.half))
        rec.add(stage.frame(), p)
    stage.striker(None)
    rec.sound(_flick_click(float(a["impulse_Ns"])))
    step = max(1, len(traj) // (FPS * 5))
    for _, pos, q in traj[::step]:
        stage.place(o.id, (pos[0] - x0, o.y, pos[2]), q)
        rec.add(stage.frame(), p)
    ok = result and result.get("success")
    if ok:
        banner = ("SOLVED", ACCENT)
    elif result and not result.get("correct_object"):
        banner = ("WRONG OBJECT", WARN)
    else:
        banner = (f"MISSED by {abs(result['error_m']) * 100:.1f} cm", WARN)
    rec.add(stage.frame(), P(banner), int(FPS * 2.5))


def _flick_click(J):
    rng = np.random.default_rng(0)
    n = int(0.06 * sound.FS)
    env = np.exp(-np.arange(n) / (0.004 * sound.FS))
    return rng.normal(0, 1, n) * env * 0.004 * min(1.0, 0.3 + J / 3)


# ------------------------------------------------------------------ cards and encoding
def card(lines: list[tuple[str, ImageFont.FreeTypeFont, tuple]]) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    total = sum(f.size + 14 for _, f, _ in lines)
    y = (H - total) // 2
    for text, f, col in lines:
        d.text((W // 2, y), text, fill=col, font=f, anchor="mt")
        y += f.size + 14
    return img


def encode(rec: Recorder, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    n = len(rec.frames)
    audio = np.zeros(int(n / FPS * sound.FS) + sound.FS)
    for fi, x in rec.audio_events:
        i = int(fi / FPS * sound.FS)
        seg = x[: len(audio) - i]
        audio[i:i + len(seg)] += seg
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "a.wav"
        sound.write_wav(audio, wav, gain=AUDIO_GAIN)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", str(FPS), "-i", "-", "-i", str(wav), "-c:v", "libx264", "-preset", "medium", "-crf", "20",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(out)]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for f in rec.frames:
            p.stdin.write(f.tobytes())
        p.stdin.close()
        if p.wait() != 0:
            raise RuntimeError("ffmpeg failed")
    return out


def render_run(run_dir: Path, out: Path, model_label: str, levels: list[str] | None = None, cards: bool = True):
    traces = []
    for t in sorted(Path(run_dir).glob("*/trace.json")):
        d = json.loads(t.read_text())
        d["_dir"] = str(t.parent)
        if levels and d["level"]["id"] not in levels:
            continue
        traces.append(d)
    rec = Recorder(model_label)
    solved = sum(bool(t["result"] and t["result"].get("success")) for t in traces)
    tainted = sum(bool(t.get("tainted")) for t in traces)
    intro = card([("Knock", F_BIG, INK),
                  (f"{model_label} plays {len(traces)} levels", F_TITLE, INK),
                  ("Identical-looking objects. Hidden physics. A few experiments, then one flick.", F_BODY, MUTED),
                  (f"Solved {solved}/{len(traces)}", F_H, ACCENT)])
    if cards:
        rec.frames += [intro] * (FPS * 3)
    for t in traces:
        replay_episode(t, rec, model_label)
    rows = [(f"{t['level']['id']}  {'solved' if t['result'] and t['result'].get('success') else 'missed'}", F_BODY,
             ACCENT if t["result"] and t["result"].get("success") else WARN) for t in traces]
    if cards:
        rec.frames += [card([("Results", F_BIG, INK), (f"{model_label}: {solved}/{len(traces)} solved", F_TITLE, INK)])] * (FPS * 3)
    return encode(rec, out)

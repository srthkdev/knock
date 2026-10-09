"""Export levels, play bundles and agent replays as static files for the web app.

    uv run python -m knock.webexport --levels levels/dev --runs runs/sol runs/luna runs/scripted \
        --out web/public/data

Layout:
    index.json                       levels + run summaries
    levels/<id>.json                 public level + play bundle (MJCF per object, probe sounds)
    replays/<run>/<id>.json          steps, readings, reasoning, trajectories, result
    audio/<id>_<obj>_<probe><k>.mp3  probe sounds (same seeds the benchmark uses)

The play bundle necessarily contains each object's simulation model (so the browser can run
the same MuJoCo physics); it is a game build, not a held-out benchmark split.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import sound, world
from .env import PUSH_SECONDS, probe_seed
from .level import FLICK_RANGE, PROPERTIES, PUSH_RANGE, Level, public_view
from .materials import CONTENTS

FPS = 30
AUDIO_GAIN = 73.0
RUN_LABELS = {
    "sol": "GPT-6.1 Sol",
    "luna": "GPT-6 Luna",
    "astra": "GPT-6 Astra",
    "scripted": "Scripted baseline",
    "random": "Random probes",
    "none": "No probes",
    "sbi": "SBI neural baseline",
}


def _mp3(x, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "a.wav"
        sound.write_wav(x, wav, gain=AUDIO_GAIN)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-ac", "1", "-b:a", "80k", str(path)],
                       check=True)


def _probe_audio(level: Level, oid: str, probe: str, repeat: int, out: Path) -> dict:
    o = level.obj(oid)
    sim = world.simulate_tap if probe == "tap" else world.simulate_shake
    x, feats = sim(o, seed=probe_seed(level, oid, probe, repeat))
    name = f"{level.id}_{oid}_{probe}{repeat}.mp3"
    f = out / "audio" / name
    if not f.exists():
        _mp3(x, f)
    return {"audio": f"audio/{name}", "features": feats}


def _traj(fn, *args):
    rec = []
    fn(*args, record=rec, fps=FPS)
    if not rec:
        return []
    x0 = rec[0][1][0]
    return [[round(t, 4), round(float(p[0] - x0), 5), round(float(p[2]), 5), *[round(float(v), 5) for v in q]]
            for t, p, q in rec]


def export_level(level: Level, out: Path) -> dict:
    pub = public_view(level)
    objects = []
    for o in level.objects:
        rng = np.random.default_rng(0)
        objects.append({
            "id": o.id,
            "edge_m": round(2 * o.half, 4),
            "half": o.half,
            "color": o.color,
            "y": o.y,
            "mjcf": world.object_scene(o, rng),
            "lift_g": world.weigh(o, np.random.default_rng(level.seed * 7 + ord(o.id))),
            "tap": _probe_audio(level, o.id, "tap", 0, out),
            "shake": _probe_audio(level, o.id, "shake", 0, out),
        })
    t = level.task
    bundle = {
        "id": level.id,
        "public": pub,
        "task": {"kind": t.kind, "property": t.property, "property_text": PROPERTIES.get(t.property or "", ""),
                 "object_id": t.object_id, "target_x": t.target_x, "tol": t.tol, "budget": t.budget},
        "objects": objects,
        "answer": level.answer_id(),
        "oracle": level.notes.get("oracle"),
        "physics": {"push_seconds": PUSH_SECONDS, "push_range": PUSH_RANGE, "flick_range": FLICK_RANGE,
                    "dt": world.DT, "gravity": world.G},
    }
    (out / "levels").mkdir(parents=True, exist_ok=True)
    (out / "levels" / f"{level.id}.json").write_text(json.dumps(bundle))
    return {"id": level.id, "kind": t.kind, "property": t.property, "budget": t.budget,
            "goal": pub["goal"], "color": level.objects[0].color, "n_objects": len(level.objects)}


def export_replay(trace: dict, run: str, out: Path) -> dict:
    level = Level.from_json(json.dumps(trace["level"]))
    replies = [c["reply"] for c in trace.get("calls", []) if c.get("reply")]
    steps = []
    counts: dict = {}
    for i, h in enumerate([h for h in trace["history"] if h["event"] == "step"]):
        a, obs = h["action"], h["obs"]
        step = {"action": a, "reading": {k: v for k, v in obs.items()
                                         if k in ("force_N", "moved_mm", "weight_g", "sound", "error",
                                                  "final_x", "error_m", "success", "correct_object")},
                "reasoning": replies[i]["reasoning"] if i < len(replies) else None}
        oid = str(a.get("object", "")).upper()
        if "error" not in obs and oid in {o.id for o in level.objects}:
            o = level.obj(oid)
            if a["type"] == "flick":
                J = float(np.clip(float(a["impulse_Ns"]), *FLICK_RANGE))
                step["traj"] = _traj(world.simulate_flick, o, J)
            elif a["probe"] == "push":
                step["traj"] = _traj(world.simulate_push, o, float(obs["force_N"]), PUSH_SECONDS)
            elif a["probe"] in ("tap", "shake"):
                k = counts.get((oid, a["probe"]), 0)
                counts[(oid, a["probe"])] = k + 1
                step["audio"] = _probe_audio(level, oid, a["probe"], k, out)["audio"]
        steps.append(step)
    oracle = level.notes.get("oracle") or {}
    ideal = None
    if oracle:
        ideal = _traj(world.simulate_flick, level.obj(oracle["object"]), oracle["impulse_Ns"])
    data = {"run": run, "label": RUN_LABELS.get(run, run), "level": level.id, "steps": steps,
            "result": trace.get("result"), "tainted": trace.get("tainted", False),
            "model": trace.get("model"), "oracle": oracle, "oracle_traj": ideal,
            "seconds": trace.get("seconds")}
    d = out / "replays" / run
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{level.id}.json").write_text(json.dumps(data))
    r = trace.get("result") or {}
    return {"success": bool(r.get("success")), "correct_object": bool(r.get("correct_object")),
            "error_m": r.get("error_m"), "probes": r.get("probes_used")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="levels/dev")
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--out", default="web/public/data")
    args = ap.parse_args()
    out = Path(args.out)
    levels = [Level.from_json(f.read_text()) for f in sorted(Path(args.levels).glob("*.json"))]
    index = {"levels": [], "runs": []}
    for lv in levels:
        index["levels"].append(export_level(lv, out))
        print("level", lv.id, flush=True)
    for run_dir in args.runs:
        run = Path(run_dir).name
        eps = {}
        for t in sorted(Path(run_dir).glob("*/trace.json")):
            eps[t.parent.name] = export_replay(json.loads(t.read_text()), run, out)
        n = len(eps)
        s = sum(e["success"] for e in eps.values())
        index["runs"].append({"run": run, "label": RUN_LABELS.get(run, run), "n": n, "solved": s, "episodes": eps})
        print("run", run, f"{s}/{n}", flush=True)
    (out / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()

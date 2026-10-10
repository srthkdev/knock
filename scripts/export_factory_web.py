"""Export the factory suite and agent episodes for the web.

    uv run python scripts/export_factory_web.py levels/factory runs/factory_v1 web/public/data/factory

Held-out maps live in <levels>_heldout/ (git-ignored): they are scored but never published.

- play/<task>.json   playable maps (public split only; ships the hidden physics, so never test maps)
- <run>/<task>.json  one episode per model and map: every turn with the layout after it, reasoning,
                     prediction vs result, and its replay (public split only; test maps keep scores only)
- index.json         tasks (tier, split) with each model's result
"""
import json
import shutil
import sys
from pathlib import Path

import yantra.packs  # noqa: F401
from yantra.core import Scenario
from yantra.packs import factory_sim as fs
from yantra.packs.factory import PRODUCTION_S, apply_ops

LABELS = {"sol": "GPT-6.1 Sol", "luna": "GPT-6 Luna", "terra": "GPT-5.6 Terra", "sonnet": "Claude Sonnet 5.5", "astra": "GPT-6 Astra"}


def compact_replay(src: Path, dst: Path, every: int = 2) -> None:
    """Every `every`-th frame, positions in mm and quaternions x1000 as integers (the viewer interpolates)."""
    d = json.loads(src.read_text())
    frames = d["frames"][::every]
    for f in frames:
        f["items"] = [[it[0], int(it[1]), *(round(v * 1000) for v in it[2:])] for it in f["items"]]
        f["push"] = [round(v * 1000) for v in f["push"]]
    d["frames"] = frames
    d["fps"] = 12 / every
    d["scale"] = 1000
    dst.write_text(json.dumps(d, separators=(",", ":")))


def task_id(sc_id: str) -> str:
    return sc_id.removeprefix("F-")


def trim_run(o: dict) -> dict:
    keep = {k: o.get(k) for k in ("seconds", "shipped", "spilled", "bins", "spills_by_cell", "jams_by_cell",
                                 "blocked_at_inlet", "in_transit", "pusher_fires", "prediction_error", "emitted")}
    keep["scales"] = {c: {"n": len(v), "min": min(v) if v else None, "max": max(v) if v else None, "readings": v}
                      for c, v in (o.get("scale_readings_g") or {}).items()}
    return keep


def export_episode(trace_path: Path, sc: Scenario, out: Path, run: str) -> dict:
    t = json.loads(trace_path.read_text())
    tid = task_id(sc.id)
    lay = fs.Layout.from_dict(sc.task.params["_start"])
    money = sc.world.settings["money"]
    dst = out / run / tid
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    turns = []
    for i, c in enumerate(t["calls"], 1):
        call, obs = c["call"], c["observation"]
        turn = {"i": i, "tool": call["tool"], "args": call["args"], "reasoning": call["reasoning"],
                "prediction": call.get("prediction"), "replay": None}
        if call["tool"] == "build" and "ops" in call["args"]:
            log, spent = apply_ops(lay, call["args"]["ops"], money)
            money -= spent
            turn["obs"] = {"results": obs.get("results", log), "money_left": obs.get("money_left", round(money))}
        elif call["tool"] == "run":
            turn["obs"] = trim_run(obs)
        else:
            turn["obs"] = obs
        turn["layout"] = lay.to_dict()
        for a in c.get("attachments", []):
            if a["kind"] == "trajectory":
                name = Path(a["path"]).name
                compact_replay(trace_path.parent / name, dst / name)
                turn["replay"] = f"{run}/{tid}/{name}"
        turns.append(turn)
    res = t["result"] or {}
    final = None
    if res.get("replay") and (trace_path.parent / "production.json").exists():
        compact_replay(trace_path.parent / "production.json", dst / "production.json")
        final = f"{run}/{tid}/production.json"
    llm = t.get("llm_calls", [])
    ep = {"id": tid, "run": run, "label": LABELS.get(run, run), "model": t.get("model"), "title": sc.task.params["title"],
          "goal": sc.task.goal, "params": {k: v for k, v in sc.task.params.items() if k != "rules" and not k.startswith("_")},
          "budgets": sc.task.budgets, "turns": turns,
          "result": {k: res.get(k) for k in ("success", "score", "checks", "production", "end", "money_left")},
          "final_layout": res.get("layout"), "final_replay": final,
          "think_seconds": round(sum(x.get("seconds", 0) for x in llm)), "tainted": t.get("tainted", False),
          "item": sc.world.settings.get("item", "box")}
    (out / run / f"{tid}.json").write_text(json.dumps(ep, separators=(",", ":")))
    return ep


def play_level(sc: Scenario) -> dict:
    p = sc.world.get("plant").hidden
    return {"id": task_id(sc.id), "title": sc.task.params["title"], "goal": sc.task.goal, "start": sc.task.params["_start"],
            "hidden": {**p, "mixes": sc.world.settings["_mixes"], "item": sc.world.settings.get("item", "box")}, "money": sc.world.settings["money"],
            "target": sc.task.params["target_good_per_min"], "max_bad": sc.task.params["max_bad_shipped"],
            "budgets": sc.task.budgets, "production_seed": (sc.world.seed * 7919 + 17) & 0xFFFFFFFF,
            "production_s": PRODUCTION_S}


def main(levels: Path, runs: Path, out: Path):
    for d in out.iterdir() if out.exists() else []:
        if d.is_dir():
            shutil.rmtree(d)
    (out / "play").mkdir(parents=True, exist_ok=True)
    files = sorted(levels.glob("*.json")) + sorted((levels.parent / f"{levels.name}_heldout").glob("*.json"))
    scen = {sc.id: sc for sc in (Scenario.from_json(f.read_text()) for f in files)}
    tasks = {}
    for sc in scen.values():
        split = sc.task.params.get("_split", "public")
        tid = task_id(sc.id)
        if split == "public":
            (out / "play" / f"{tid}.json").write_text(json.dumps(play_level(sc), separators=(",", ":")))
        lay = fs.Layout.from_dict(sc.task.params["_start"])
        tasks[sc.id] = {"id": tid, "title": sc.task.params["title"], "detail": sc.task.goal.split(" Score:")[0],
                        "tier": sc.task.params.get("_tier", 0), "split": split, "size": f"{lay.w}x{lay.h}",
                        "target": sc.task.params["target_good_per_min"], "item": sc.world.settings.get("item", "box"), "runs": []}
    for tp in sorted(runs.glob("*/*/trace.json")):
        run, sid = tp.parent.parent.name, tp.parent.name
        if sid not in scen:
            continue
        t = json.loads(tp.read_text())
        res = t["result"] or {}
        entry = {"run": run, "label": LABELS.get(run, run), "success": bool(res.get("success")), "score": res.get("score"),
                 "turns": len(t["calls"]), "tainted": bool(t.get("tainted"))}
        if tasks[sid]["split"] == "public":
            export_episode(tp, scen[sid], out, run)
        tasks[sid]["runs"].append(entry)
        print("exported", run, sid)
    order = sorted(tasks.values(), key=lambda t: (t["tier"], t["split"] != "public", t["id"]))
    order.append({"id": "bottling", "title": "Bottling line (physics replay)", "tier": 0, "split": "demo", "size": "",
                  "detail": "A recorded MuJoCo run of a bottling line: checkweigher and reject pusher. No agent.",
                  "runs": [{"run": "demo", "label": "Watch replay", "success": True, "score": None, "turns": 0}]})
    (out / "index.json").write_text(json.dumps({"benchmark": "factory v1", "tasks": order}, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))

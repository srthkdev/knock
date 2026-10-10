"""Export factory episodes for the web viewer.

    uv run python scripts/export_factory_web.py runs/factory web/public/data/factory

For every run/<model>/<scenario>/trace.json writes <out>/<model>/<map>.json (turns with the layout after
each turn, reasoning, predictions vs results) and copies each turn's replay; then writes index.json.
"""
import json
import shutil
import sys
from pathlib import Path

import knock.packs  # noqa: F401
from knock.packs import FactoryPack
from knock.packs import factory_sim as fs
from knock.packs.factory import MAPS, apply_ops

LABELS = {"sol": "GPT-6.1 Sol", "luna": "GPT-6 Luna", "astra": "GPT-6 Astra", "reference": "Reference build"}


def trim_run(o: dict) -> dict:
    keep = {k: o.get(k) for k in ("seconds", "shipped", "spilled", "bins", "spills_by_cell", "jams_by_cell",
                                 "blocked_at_inlet", "in_transit", "pusher_fires", "prediction_error", "emitted")}
    keep["scales"] = {c: {"n": len(v), "min": min(v) if v else None, "max": max(v) if v else None, "readings": v}
                      for c, v in (o.get("scale_readings_g") or {}).items()}
    return keep


def export(trace_path: Path, out: Path, run: str) -> dict:
    t = json.loads(trace_path.read_text())
    params = t["task"]["params"]
    map_ = params["map"]
    sc = FactoryPack().generate(t["world"]["seed"], map=map_)
    lay = fs.Layout.from_dict(sc.world.settings["_layout"])
    money = sc.world.settings["money"]
    dst = out / run / map_
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
                shutil.copy(trace_path.parent / name, dst / name)
                turn["replay"] = f"{run}/{map_}/{name}"
        turns.append(turn)
    res = t["result"] or {}
    final = None
    if res.get("replay"):
        shutil.copy(trace_path.parent / "production.json", dst / "production.json")
        final = f"{run}/{map_}/production.json"
    llm = t.get("llm_calls", [])
    ep = {"id": map_, "run": run, "label": LABELS.get(run, run), "model": t.get("model"), "title": params["title"],
          "goal": t["task"]["goal"], "params": {k: v for k, v in params.items() if k != "rules"},
          "budgets": t["task"]["budget"], "turns": turns,
          "result": {k: res.get(k) for k in ("success", "score", "checks", "production", "end", "money_left")},
          "final_layout": res.get("layout"), "final_replay": final,
          "think_seconds": round(sum(x.get("seconds", 0) for x in llm)), "tainted": t.get("tainted", False)}
    (out / run / f"{map_}.json").write_text(json.dumps(ep, separators=(",", ":")))
    return ep


def main(src: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    eps = []
    for tp in sorted(src.glob("*/*/trace.json")):
        run = tp.parent.parent.name
        eps.append(export(tp, out, run))
        print("exported", run, tp.parent.name)
    tasks = []
    for map_, spec in MAPS.items():
        runs = [{"run": e["run"], "label": e["label"], "success": e["result"]["success"], "score": e["result"]["score"],
                 "turns": len(e["turns"])} for e in eps if e["id"] == map_]
        tasks.append({"id": map_, "title": spec["title"], "detail": spec["brief"], "runs": runs})
    tasks.append({"id": "bottling", "title": "Bottling line (physics replay)",
                  "detail": "A recorded MuJoCo run of a bottling line: checkweigher and reject pusher. No agent.",
                  "runs": [{"run": "demo", "label": "Watch replay", "success": True}]})
    (out / "index.json").write_text(json.dumps({"tasks": tasks}, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))

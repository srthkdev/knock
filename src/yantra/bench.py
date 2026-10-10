"""Benchmark runner on yantra.core: any agent, any pack, traces and summaries.

    from yantra.bench import load_scenarios, run_suite
    results = run_suite(load_scenarios("levels/dev"), agent, "runs/v03/dev-scripted")
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from .core import Episode, Scenario, run_episode


def load_scenarios(path: str | Path) -> list[Scenario]:
    """Scenario JSON files, or Knock v0.2 level files (converted to the tabletop pack)."""
    from .level import Level
    from .packs.tabletop import scenario_from_level

    out = []
    for f in sorted(Path(path).glob("*.json")):
        if f.name.endswith(".report.json"):
            continue
        d = json.loads(f.read_text())
        out.append(Scenario.from_dict(d) if "pack" in d else scenario_from_level(Level.from_json(f.read_text())))
    return out


def run_suite(scenarios: list[Scenario], agent, out_dir: str | Path | None = None, overwrite: bool = False,
              log=print) -> list[dict]:
    results = []
    for sc in scenarios:
        ep_dir = Path(out_dir) / sc.id if out_dir else None
        if ep_dir and (ep_dir / "trace.json").exists() and not overwrite:
            results.append(json.loads((ep_dir / "trace.json").read_text())["result"])
            continue
        t0 = time.time()
        ep = Episode(sc.world, sc.task, ep_dir)
        r = run_episode(agent, ep)
        meta = getattr(agent, "episode_meta", lambda: {})()
        ep.save({"scenario": sc.id, "pack": sc.pack, "agent": agent.name, "seconds": round(time.time() - t0, 1), **meta})
        results.append(r)
        if log:
            pe = (r.get("prediction") or {}).get("stop_x_m", {}).get("error")
            log(f"{sc.id} {'OK ' if r.get('success') else 'x  '} target={'y' if r.get('correct_target') else 'n'} "
                f"err={r.get('error_m', float('nan')):+.3f} pred_err={'-' if pe is None else f'{pe:.3f}'} "
                f"probes={r.get('probes_used')} {time.time() - t0:.0f}s")
    return results


def summarize(results: list[dict]) -> dict:
    n = len(results)
    out = {"n": n, "solved": sum(bool(r.get("success")) for r in results),
           "correct_target": sum(bool(r.get("correct_target")) for r in results)}
    act_err = [r["prediction"]["stop_x_m"]["error"] for r in results if (r.get("prediction") or {}).get("stop_x_m")]
    if act_err:
        out["act_prediction_error_median_m"] = round(statistics.median(act_err), 4)
    first, last = [], []
    for r in results:
        errs = [p["scores"]["moved_mm"]["error"] for p in r.get("probe_predictions") or [] if "moved_mm" in p["scores"]]
        if errs:
            first.append(errs[0])
        if len(errs) >= 2:
            last.append(errs[-1])
    if first:
        out["push_prediction_error_first_median_mm"] = round(statistics.median(first), 1)
    if last:
        out["push_prediction_error_last_median_mm"] = round(statistics.median(last), 1)
    return out

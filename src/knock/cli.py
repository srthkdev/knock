"""Command line: generate levels, run agents, summarise results.

    knock gen --n 30 --out levels/dev
    knock run --agent scripted --levels levels/dev --out runs/scripted
    knock run --agent codex --model gpt-6-luna --levels levels/dev --out runs/luna
    knock summary runs/*
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .env import Episode
from .level import Level
from .oracle import valid_level


def cmd_gen(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        lv = valid_level(args.start + i)
        (out / f"{lv.id}.json").write_text(lv.to_json())
    print(f"wrote {args.n} levels to {out}")


def make_agent(args):
    if args.agent in ("none", "random", "scripted"):
        from .agents.baselines import BaselineAgent
        return BaselineAgent(args.agent)
    if args.agent == "codex":
        from .agents.codex_agent import CodexAgent
        return CodexAgent(model=args.model, effort=args.effort)
    raise SystemExit(f"unknown agent {args.agent}")


def cmd_run(args):
    agent = make_agent(args)
    out = Path(args.out)
    files = sorted(Path(args.levels).glob("*.json"))[: args.limit or None]
    results = []
    for f in files:
        lv = Level.from_json(f.read_text())
        ep_dir = out / lv.id
        if (ep_dir / "trace.json").exists() and not args.overwrite:
            results.append(json.loads((ep_dir / "trace.json").read_text())["result"])
            continue
        t0 = time.time()
        ep = Episode(lv, ep_dir, render=args.render)
        agent.run(ep)
        meta = getattr(agent, "episode_meta", lambda: {})()
        ep.save_trace({"agent": agent.name, "seconds": round(time.time() - t0, 1), **meta})
        r = ep.result or {"success": False, "error": "no flick"}
        results.append(r)
        print(f"{lv.id} {lv.task.kind:15} {str(lv.task.property or ''):9} "
              f"{'OK ' if r.get('success') else 'x  '} obj={'y' if r.get('correct_object') else 'n'} "
              f"err={r.get('error_m', float('nan')):+.3f} probes={r.get('probes_used')} {time.time() - t0:.0f}s", flush=True)
    summarise(out, results)


def summarise(out, results):
    n = len(results)
    if not n:
        return
    s = sum(bool(r.get("success")) for r in results)
    c = sum(bool(r.get("correct_object")) for r in results)
    print(f"{out}: success {s}/{n} = {s / n:.0%} | correct object {c}/{n}")


def cmd_summary(args):
    for d in args.dirs:
        traces = sorted(Path(d).glob("*/trace.json"))
        summarise(d, [json.loads(t.read_text())["result"] or {} for t in traces])


def main():
    p = argparse.ArgumentParser(prog="knock")
    sub = p.add_subparsers(required=True)
    g = sub.add_parser("gen")
    g.add_argument("--n", type=int, default=30)
    g.add_argument("--start", type=int, default=0)
    g.add_argument("--out", default="levels/dev")
    g.set_defaults(fn=cmd_gen)
    r = sub.add_parser("run")
    r.add_argument("--agent", default="scripted")
    r.add_argument("--model", default="gpt-6-luna")
    r.add_argument("--effort", default="high")
    r.add_argument("--levels", default="levels/dev")
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    r.add_argument("--overwrite", action="store_true")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("summary")
    s.add_argument("dirs", nargs="+")
    s.set_defaults(fn=cmd_summary)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

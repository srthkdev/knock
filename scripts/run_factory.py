"""Run a Codex (GPT) agent on factory maps; one trace per episode with per-turn replays.

    uv run python scripts/run_factory.py gpt-6.1-sol runs/factory/sol first_line:1 quality_gate:1 two_lines:1
"""
import json
import sys
import time
from pathlib import Path

import knock.packs  # noqa: F401
from knock.agents.llm import CodexLLMAgent
from knock.core import Episode, run_episode
from knock.packs import FactoryPack

model, out = sys.argv[1], Path(sys.argv[2])
effort = "high"
for spec in sys.argv[3:]:
    map_, seed = spec.split(":")
    sc = FactoryPack().generate(int(seed), map=map_)
    d = out / sc.id
    if (d / "trace.json").exists():
        print("skip", sc.id); continue
    agent = CodexLLMAgent(model, effort, log_dir=d / "_codex")
    ep = Episode(sc.world, sc.task, out_dir=d)
    t = time.time()
    res = run_episode(agent, ep)
    ep.save({"agent": agent.name, "scenario": sc.id, "seconds": round(time.time() - t), **agent.episode_meta()})
    print(sc.id, model, json.dumps({k: res.get(k) for k in ("success", "score", "checks", "production", "end")}), flush=True)

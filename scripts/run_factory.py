"""Run a Codex (GPT) agent on factory scenarios; one trace per episode with per-turn replays.

    uv run python scripts/run_factory.py gpt-6.1-sol runs/factory_v1/sol levels/factory/F-first_line-1.json ...
"""
import json
import sys
import time
from pathlib import Path

import yantra.packs  # noqa: F401
from yantra.agents.llm import CodexLLMAgent
from yantra.core import Episode, Scenario, run_episode

model, out = sys.argv[1], Path(sys.argv[2])
for f in sys.argv[3:]:
    sc = Scenario.from_json(Path(f).read_text())
    d = out / sc.id
    if (d / "trace.json").exists():
        print("skip", sc.id); continue
    agent = CodexLLMAgent(model, "high", log_dir=d / "_codex")
    ep = Episode(sc.world, sc.task, out_dir=d)
    t = time.time()
    res = run_episode(agent, ep)
    ep.save({"agent": agent.name, "scenario": sc.id, "seconds": round(time.time() - t), **agent.episode_meta()})
    print(sc.id, model, json.dumps({k: res.get(k) for k in ("success", "score", "checks", "production", "end")}), flush=True)

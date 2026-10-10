"""Controlled test of the v0.3 harness drop: same model, same levels, three prompt variants.

    A  current generic prompt
    B  A + explicit instruction to compute the decisive act from the readings
    C  B with predictions optional
"""
import json
import sys
from pathlib import Path

import yantra.packs  # noqa: F401
from yantra.agents.llm import CodexLLMAgent
from yantra.bench import load_scenarios, summarize
from yantra.core import Episode, run_episode

COMPUTE = ("When you use the decisive act, choose its arguments by explicit calculation from your readings "
           "(write the formula and numbers in your reasoning): it is your only attempt and it is scored by its result.")
variant, model, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
n = int(sys.argv[4]) if len(sys.argv) > 4 else 10
extra = "" if variant == "A" else COMPUTE
results = []
for sc in load_scenarios("levels/dev")[:n]:
    d = out / sc.id
    if (d / "trace.json").exists():
        results.append(json.loads((d / "trace.json").read_text())["result"])
        continue
    agent = CodexLLMAgent(model, "high", log_dir=d / "_codex", extra_instructions=extra)
    ep = Episode(sc.world, sc.task, d, require_predictions=(variant != "C"))
    r = run_episode(agent, ep)
    ep.save({"scenario": sc.id, "agent": agent.name, "variant": variant, **agent.episode_meta()})
    results.append(r)
    print(sc.id, r.get("success"), r.get("error_m"), flush=True)
print(json.dumps({"variant": variant, **summarize(results)}))

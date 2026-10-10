"""Run a Codex-backed LLM agent on a scenario set with the core runner.

    uv run python scripts/run_llm.py gpt-6.1-sol levels/dev runs/v03/dev-sol
"""
import json
import sys

import knock.packs  # noqa: F401
from knock.agents.llm import CodexLLMAgent
from knock.bench import load_scenarios, run_suite, summarize

model, levels, out = sys.argv[1], sys.argv[2], sys.argv[3]
effort = sys.argv[4] if len(sys.argv) > 4 else "high"


results = []
for sc in load_scenarios(levels):
    agent = CodexLLMAgent(model, effort, log_dir=f"{out}/{sc.id}/_codex")
    results += run_suite([sc], agent, out)
print(json.dumps(summarize(results)))

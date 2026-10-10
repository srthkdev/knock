"""Write web/scripts/parity_cases.json: Python flick/push results for the browser parity test.

    uv run python scripts/parity_cases.py && (cd web && npx tsx scripts/parity.ts)
"""
import json
from pathlib import Path

import numpy as np

from yantra import world
from yantra.level import Level

cases = []
for f in sorted(Path("levels/dev").glob("*.json"))[:8]:
    lv = Level.from_json(f.read_text())
    for o in lv.objects:
        xml = world.object_scene(o, np.random.default_rng(0))
        J = float(o.mass * np.sqrt(2 * o.hidden.friction * 9.81 * 0.5))
        F = float(1.5 * o.hidden.friction * o.mass * 9.81)
        cases.append(dict(id=f"{lv.id}{o.id}", xml=xml, J=J, F=F,
                          flick=world.simulate_flick(o, J), push=world.simulate_push(o, F, 0.4)))
Path("web/scripts/parity_cases.json").write_text(json.dumps(cases))
print(len(cases), "cases")

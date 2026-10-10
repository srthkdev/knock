"""Write the hand-made factory maps (two seeds each) as scenario files next to the generated ones."""
import sys
from pathlib import Path

from yantra.packs import FactoryPack
from yantra.packs.factory import MAPS

out = Path(sys.argv[1] if len(sys.argv) > 1 else "levels/factory")
out.mkdir(parents=True, exist_ok=True)
seeds = {"tight_spec": (2, 3), "mixed_merge": (2, 3)}
for name in MAPS:
    for seed in seeds.get(name, (1, 2)):
        sc = FactoryPack().generate(seed, map=name)
        sc.task.params["_tier"] = {"first_line": 1, "quality_gate": 2, "two_lines": 2}.get(name, 3)
        (out / f"{sc.id}.json").write_text(sc.to_json())
        print(sc.id)

"""Generate the factory suite: random maps per tier, kept only if the reference build passes.

    uv run python scripts/gen_factory.py levels/factory 24
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from yantra.packs import FactoryPack
from yantra.packs.factory_gen import random_spec, solve_spec


def attempt(args):
    seed, tier = args
    spec = random_spec(seed, tier)
    if spec is None:
        return seed, tier, None
    sol = solve_spec(spec, seed)
    if sol is None:
        return seed, tier, None
    ref, prod = sol
    from yantra.packs.factory_gen import MONEY_SLACK, TARGET
    spec["target_abs"] = float(int(TARGET * prod["good_per_min"]))
    spec["money"] = int(round(ref["cost"] * MONEY_SLACK / 50.0) * 50)
    sc = FactoryPack().generate(seed, map=f"gen{seed}", spec=spec)
    sc.notes = {"reference": ref, "reference_production": prod}
    return seed, tier, sc.to_json()


if __name__ == "__main__":
    out, want = Path(sys.argv[1]), int(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    per_tier = {1: want // 3, 2: want // 3, 3: want - 2 * (want // 3)}
    got = {1: 0, 2: 0, 3: 0}
    seed = 1000
    with ProcessPoolExecutor(8) as ex:
        while any(got[t] < per_tier[t] for t in got):
            jobs = [(seed + i, t) for i, t in enumerate(t for t in got for _ in range(max(0, per_tier[t] - got[t]) * 2))]
            seed += len(jobs) + 1
            for s, t, js in ex.map(attempt, jobs):
                if js is None or got[t] >= per_tier[t]:
                    continue
                got[t] += 1
                d = json.loads(js)
                d["task"]["params"]["_split"] = "test" if got[t] % 3 == 0 else "public"
                d["task"]["params"]["title"] = f"Plant {s}"
                (out / f"{d['id']}.json").write_text(json.dumps(d, indent=1, default=str))
                print("kept", d["id"], "tier", t, "target", d["task"]["params"]["target_good_per_min"], d["task"]["params"]["_split"], flush=True)

"""Generate bottling levels; the production target is 80% of a hidden-info physics reference.

    uv run python scripts/gen_bottling.py levels/bottling 1 2 3 4 5 6
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scratch"))
from bottling_oracle import reference  # noqa: E402

from yantra.packs import BottlingPack  # noqa: E402


def make(seed):
    sc = BottlingPack().generate(seed, target_good_per_min=0)
    best = reference(sc)
    if best is None:
        return seed, None
    s, o = best
    target = float(int(0.8 * o["good_per_min"]))
    sc = BottlingPack().generate(seed, target_good_per_min=target)
    sc.notes = {"reference_settings": s.model_dump(), "reference_outcome": o}
    return seed, sc.to_json()


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(6) as ex:
        for seed, js in ex.map(make, map(int, sys.argv[2:])):
            if js is None:
                print(seed, "no feasible reference, skipped"); continue
            sc = json.loads(js)
            (out / f"{sc['id']}.json").write_text(js)
            print(seed, sc["task"]["params"]["target_good_per_min"], sc["notes"]["reference_outcome"]["good_per_min"])

"""Solvability check: search pusher offset/stroke (and threshold) with the true physics on short runs."""
import sys, itertools
from knock.core import Episode, ToolCall
from knock.packs import FactoryPack
from knock.packs import factory_sim as fs
from knock.packs.factory import apply_ops, _hidden
sys.path.insert(0, "scratch")
import factory_ref as R

def score(t): return t["good_shipped"] - 5 * t["bad_shipped"] - 2 * t["spilled"] - 3 * t["jammed"]

def solve(map, seed, v, thr):
    sc = FactoryPack().generate(seed, map=map)
    R.THR = thr
    best = None
    for off, st in itertools.product(range(50, 701, 50), (200, 350, 500)):
        ops = R.ops_hard(map, sc, v).replace("offset_mm=", "offset_mm=X").split("\n")
        ops = "\n".join(l.split(" offset_mm=")[0] + f" offset_mm={off} stroke_ms={st}" if l.startswith("pusher") else l for l in ops)
        lay = fs.Layout.from_dict(sc.world.settings["_layout"]); apply_ops(lay, ops, 1e9)
        t = fs.run_factory(lay, _hidden(sc.world), 45, seed=99).truth()
        if best is None or score(t) > best[0]: best = (score(t), off, st, ops)
    ep = Episode(sc.world, sc.task); ep.call(ToolCall("build", args={"ops": best[3]})); ep.call(ToolCall("finish"))
    print(map, seed, v, thr, "off", best[1], "stroke", best[2], ep.result["success"], ep.result["production"], "target", sc.task.params["target_good_per_min"], flush=True)

if __name__ == "__main__":
    map = sys.argv[1]
    for seed in (1, 2, 3):
        solve(map, seed, float(sys.argv[2]), float(sys.argv[3]))

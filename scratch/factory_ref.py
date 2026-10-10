"""Hand-written reference factories (hidden-info tuned) to check each map is solvable."""
import sys, time
from knock.core import Episode, ToolCall
from knock.packs import FactoryPack

def tuned(sc, v, d_cells=0.0):
    h = sc.world.get("plant").hidden
    # paddle must travel ~0.19+0.07 m to meet the box; box moves v * (lag + travel/speed)
    off = v * (h["pusher_lag"] + 0.26 / h["pusher_speed"]) * 1000
    stroke = (0.6 / h["pusher_speed"]) * 1000
    return round(off), round(stroke)

def ops_for(map, sc):
    v = 1.0
    if map == "first_line":
        return f"""belt 1..2 1 E speed={v}
belt 3 1..5 N speed={v}
belt 3..5 6 E speed={v}
belt 6 6..2 S speed={v}
belt 6..9 1 E speed={v}
belt 10 1..5 N speed={v}
belt 10 6 E speed={v}"""
    if map == "quality_gate":
        off, st = tuned(sc, v)
        return f"""belt 1 2 E speed={v}
scale 2 2 E speed={v} reject_below_g=950
belt 3..4 2 E speed={v}
pusher 4 3 S offset_mm={off} stroke_ms={st}
bin 4..5 1
belt 5 2..4 N speed={v}
belt 5..10 5 E speed={v}"""
    if map == "two_lines":
        off, st = tuned(sc, v)
        return f"""belt 1..5 1 E speed={v}
belt 6 1..3 N speed={v}
scale 1 7 E speed={v} reject_above_g=1400
belt 2..5 7 E speed={v}
pusher 3 8 S offset_mm={off} stroke_ms={st}
bin 3..4 6
belt 6 7..5 S speed={v}
belt 6..12 4 E speed={v}"""

for map in sys.argv[1:]:
    for seed in (1, 2, 3):
        sc = FactoryPack().generate(seed, map=map)
        ep = Episode(sc.world, sc.task)
        o = ep.call(ToolCall("build", args={"ops": ops_for(map, sc)}))
        bad = [r for r in o.data.get("results", [o.data]) if not str(r).startswith("ok")]
        print(map, seed, "money", o.data.get("money_left"), bad)
        if seed == 1: print(o.data["map"])
        t = time.time(); ep.call(ToolCall("finish")); r = ep.result
        print("  ", r["success"], r["score"], r["production"], r["checks"], f"{time.time()-t:.1f}s", "target", sc.task.params["target_good_per_min"])

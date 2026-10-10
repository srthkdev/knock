"""Hand-written reference factories (hidden-info tuned) to check each map is solvable."""
import sys, time
from yantra.core import Episode, ToolCall
from yantra.packs import FactoryPack

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

THR = 978
def ops_hard(map, sc, v):
    off, st = tuned(sc, v)
    if map == "rush_hour":
        return f"""belt 1..2 1 E speed={v}
belt 4 6 E speed={v}
belt 3 1..5 N speed={v}
belt 3 6 E speed={v}
belt 5..7 6 E speed={v}
belt 8 6..2 S speed={v}
belt 8..9 1 E speed={v}
belt 10 1..6 N speed={v}
belt 10 7 E speed={v}
belt 11..12 7 E speed={v}"""
    if map == "tight_spec":
        return f"""belt 1 2 E speed={v}
scale 2 2 E speed={v} reject_below_g={THR}
belt 3..4 2 E speed={v}
pusher 4 3 S offset_mm={off} stroke_ms={st}
bin 4..5 1
belt 5 2..4 N speed={v}
belt 5..10 5 E speed={v}"""
    if map == "mixed_merge":
        return f"""belt 1 1 E speed={v}
scale 2 1 E speed={v} reject_below_g=965
belt 3..5 1 E speed={v}
pusher 4 2 S offset_mm={off} stroke_ms={st}
bin 4..5 0
belt 6 1..3 N speed={v}
scale 1 7 E speed={v} reject_above_g=1400
belt 2..5 7 E speed={v}
pusher 3 8 S offset_mm={off} stroke_ms={st}
bin 3..4 6
belt 6 7..5 S speed={v}
belt 6..12 4 E speed={v}"""

if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "hard":
    for map in sys.argv[2:]:
      for THR in ((972, 976) if map == "tight_spec" else (978,)):
        for v, SEED in [(0.8, 1), (0.8, 2), (0.8, 3)] if map == "tight_spec" else [(1.4, 1), (1.4, 2), (1.0, 2)]:
            sc = FactoryPack().generate(SEED, map=map)
            ep = Episode(sc.world, sc.task)
            o = ep.call(ToolCall("build", args={"ops": ops_hard(map, sc, v)}))
            bad = [r for r in o.data.get("results", [o.data]) if not str(r).startswith("ok")]
            if bad: print(bad)
            ep.call(ToolCall("finish")); r = ep.result
            print(map, THR, v, r["success"], r["production"], "target", sc.task.params["target_good_per_min"])
    sys.exit()

for map in (sys.argv[1:] if __name__ == "__main__" else []):
    for seed in (1, 2, 3):
        sc = FactoryPack().generate(seed, map=map)
        ep = Episode(sc.world, sc.task)
        o = ep.call(ToolCall("build", args={"ops": ops_for(map, sc)}))
        bad = [r for r in o.data.get("results", [o.data]) if not str(r).startswith("ok")]
        print(map, seed, "money", o.data.get("money_left"), bad)
        if seed == 1: print(o.data["map"])
        t = time.time(); ep.call(ToolCall("finish")); r = ep.result
        print("  ", r["success"], r["score"], r["production"], r["checks"], f"{time.time()-t:.1f}s", "target", sc.task.params["target_good_per_min"])

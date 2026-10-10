"""Hidden-info reference: physics-derived settings swept over belt speed; best successful production."""
import sys, time
from knock.packs import BottlingPack
from knock.packs.bottling import StartProduction, Settings
from knock.packs import bottling_sim as sim

def reference(sc, speeds=(0.6, 0.8, 1.0, 1.2, 1.5, 1.8), pitch=0.5):
    line, lot = sc.world.get("line").hidden, sc.world.get("lot").hidden
    good = sim.GLASS_G + sim.FULL_G
    bad = sim.GLASS_G + sim.FULL_G * lot["underfill_level"]
    thr = (good + bad) / 2
    best = None
    for v in speeds:
        off = v * (line["pusher_lag"] + 0.068 / line["pusher_speed"]) * 1000
        stroke = (0.20 / line["pusher_speed"]) * 1000
        s = Settings(belt_speed=v, feed_per_min=min(150, v * 60 / pitch), reject_below_g=thr, pusher_offset_mm=min(300, off), stroke_ms=max(40, stroke))
        o = StartProduction().outcome(sc.world, None, s)
        ok = sc.task.success(sc.world, None, o)
        print(f"  v={v} feed={s.feed_per_min:.0f} thr={thr:.1f} off={off:.0f} stroke={stroke:.0f} -> good/min {o['good_per_min']} def {o['defects_shipped']} goodrej {o['good_rejected']} tip {o['tipped']} {'OK' if ok else ''}")
        if o["defects_shipped"] == 0 and o["good_rejected"] <= 1 and o["tipped"] == 0 and (best is None or o["good_per_min"] > best[1]["good_per_min"]):
            best = (s, o)
    return best

if __name__ == "__main__":
    for seed in map(int, sys.argv[1:]):
        sc = BottlingPack().generate(seed, target_good_per_min=0)
        print(seed, sc.world.get("line").hidden, sc.world.get("lot").hidden)
        t = time.time(); b = reference(sc); print("  best", b and b[1]["good_per_min"], f"{time.time()-t:.0f}s")

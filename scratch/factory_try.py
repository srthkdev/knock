import time, sys
from knock.packs.factory_sim import *
def lay(sp, off, stroke, below=940):
    parts=[Part("inlet",0,2,"E",{"speed":sp,"feed_per_min":40},True,"A")]
    parts+=[Part("belt",x,2,"E",{"speed":sp}) for x in (1,2,4)]
    parts+=[Part("scale",3,2,"E",{"speed":sp,"reject_below_g":below})]
    parts+=[Part("belt",5,y,"N",{"speed":sp}) for y in (2,3,4)]
    parts+=[Part("belt",x,5,"E",{"speed":sp}) for x in (5,6,7,8)]
    parts+=[Part("bin",9,5,"E",{},True,"ship"),Part("pusher",4,3,"S",{"offset_mm":off,"stroke_ms":stroke}),Part("bin",4,1,"E",{},False,"reject")]
    return Layout(12,8,parts)
hid=Hidden(mixes={"A":[["good",0.75,1.0,0.01],["light",0.25,0.88,0.01]]})
for sp,off,st in [(0.8,60,250),(0.8,180,250),(0.8,180,400),(1.6,300,250),(2.0,400,250),(2.5,400,250)]:
    r=run_factory(lay(sp,off,st),hid,40,seed=1)
    print(sp,off,st, r.truth(), 'fires',r.pusher_fires,'spills',r.spills,'jams',r.jams)
for sp in (0.8,1.6):
    r=run_factory(lay(sp,180,400),hid,40,seed=1)
    print(sp, [round(x) for x in r.scales['3,2']])

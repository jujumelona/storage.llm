#!/usr/bin/env python3
import cueta_traceelephant_eval as e
import ueta_causal_generalization as cu

def fit_u_fast(core,val,horizon):
 cr=e.flatten_traces(core);vr=e.flatten_traces(val);g=e.goldmap(core);best=(-1,None)
 for depth in (1,2):
  for ms in (2,5):
   routes=cu.discover_routes(cr,{t['tid'] for t in core},depth,horizon,ms)
   for p in (.06,.15):
    cs=cu.map_routes(cr,{t['tid'] for t in core},g,routes,depth,horizon,2,p)
    if not cs:continue
    a=e.cal_top(cu.score(vr,{t['tid'] for t in val},cs,depth,horizon),val)
    if a>best[0]:best=(a,{'depth':depth,'min_support':ms,'min_precision':p,'routes':len(routes),'contracts':len(cs)})
 return best[1] or {'depth':1,'min_support':2,'min_precision':.06}

e.fit_u=fit_u_fast
if __name__=='__main__':e.main()

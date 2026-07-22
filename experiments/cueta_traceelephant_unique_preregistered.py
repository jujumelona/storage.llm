#!/usr/bin/env python3
from collections import Counter
import cueta_traceelephant_eval as e

_original_load=e.load
def unique_load(root):
 ts,skip=_original_load(root);seen=Counter()
 for t in ts:
  key=(t['system'],t['tid']);seen[key]+=1;new=f"{t['system']}::{t['tid']}::{seen[key]}"
  t['tid']=new
  for r in t['records']:r['tid']=new
 return ts,skip

def fixed_u(core,val,horizon):
 return {'depth':2,'min_support':3,'min_precision':.06,'preregistered':True}
def fixed_b(core,val):
 return {'min_positive':3,'preregistered':True}

e.load=unique_load
e.fit_u=fixed_u
e.fit_b=fixed_b
if __name__=='__main__':e.main()

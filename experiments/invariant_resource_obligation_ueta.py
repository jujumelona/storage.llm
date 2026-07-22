#!/usr/bin/env python3
import json
from pathlib import Path
import resource_obligation_ueta as r
import boundary_ueta_traceelephant as b

def fixed_tune(core,val):
 return {'min_support':2,'topk':10,'mode':'raw','alpha':0.0,'invariant_priority':True}
b.tune=fixed_tune

def proof_rank(a):
 if ('COMMIT_AFTER_UNRESOLVED_FAIL',) in a:return 7
 if any(x[0]=='COMMIT_WITH_OPEN_OBLIGATION' for x in a):return 6
 if any(x[0]=='CAUSAL_MUTATION_TO_FAIL' for x in a):return 5
 if ('COMMIT',) in a:return 4
 if ('VERIFY','fail') in a:return 3
 if any(x[0] in ('LAST_MUTATION_BEFORE_FAIL','LAST_MUTATION_BEFORE_COMMIT') for x in a):return 2
 if any(x[0] in ('RECOVERY_MUTATION_AFTER_FAIL','MUTATION') for x in a):return 1
 return 0

def invariant_scores(t,w,topk=10):
 aa=r.annotate(t);out=[]
 for i in range(len(t['events'])):
  vals=sorted((w[k] for k in r.keys(t,i) if k in w),reverse=True);learn=sum(vals[:topk]);rank=proof_rank(aa[i]);
  # Lexicographic encoding: no learned score can overtake one proof level.
  out.append(rank*10000.0 + min(9999.0,max(-9999.0,learn)) - (1e-6*i))
 return out
b.raw_scores=invariant_scores

def main():
 b.main()
 import argparse
 ap=argparse.ArgumentParser(add_help=False);ap.add_argument('--out');a,_=ap.parse_known_args();p=Path(a.out)/'summary.json';d=json.load(open(p));d['architecture']='Invariant Resource-Obligation UETA';d['system_holdout_min_step_gain']=min(x['step_gain'] for x in d['system_holdout']);d['strict_verdict']['all_system_holdouts_nonnegative']=d['system_holdout_min_step_gain']>=0;d['strict_verdict']['external_supported']=all(d['strict_verdict'][k] for k in ('oof_gain_3pp','bootstrap_positive','joint_not_worse','all_system_holdouts_nonnegative'));json.dump(d,open(p,'w'),indent=2);print(json.dumps({'revised_strict_verdict':d['strict_verdict'],'system_holdout_min_step_gain':d['system_holdout_min_step_gain']},indent=2))
if __name__=='__main__':main()

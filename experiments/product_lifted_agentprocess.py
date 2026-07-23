#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, gzip, hashlib, json, math, random, re, statistics, time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

SEED=20260723
random.seed(SEED)
DATASETS=('hotpotqa','gaia_dev','bfcl','tau2')
FIRST_ERROR='-1'
READ_WORDS=('get','read','list','show','fetch','search','find','lookup','inspect','view','cat','ls','pwd','stat','query','retrieve','open')
MUTATE_WORDS=('create','write','update','modify','delete','remove','cancel','book','return','exchange','move','mv','copy','cp','mkdir','touch','echo','submit','send','post','patch','replace','edit','set')
VERIFY_WORDS=('verify','check','test','validate','diff','compare','calculate','count','wc','assert','confirm')
AUTH_WORDS=('auth','login','identify','user_id_by','verify_user')
EXEC_WORDS=('run','execute','bash','python','shell','terminal')
COMM_WORDS=('transfer','handoff','notify','email','message')
ERROR_RE=re.compile(r'\b(error|failed|failure|invalid|denied|forbidden|not found|cannot|unable|exception|timeout|blocked|unavailable)\b',re.I)
PARTIAL_RE=re.compile(r'\b(partial|truncated|only\s+\d+%|\d+%\s+of|incomplete|limited set|not all)\b',re.I)
YES_RE=re.compile(r'\b(yes|confirm|confirmed|proceed|go ahead|please do|correct|okay|ok)\b',re.I)
FINAL_RE=re.compile(r'<answer>|final answer|the answer is|request (?:is )?satisfied|successfully|completed',re.I)
ASK_RE=re.compile(r'\?|could you|please provide|need (?:your|the)|would you',re.I)
PLAN_RE=re.compile(r'\b(plan|next step|first|then|we need|should)\b',re.I)

def stable_int(s):return int(hashlib.sha256(s.encode()).hexdigest()[:16],16)
def bucket(n):
    if n<=0:return '0'
    if n==1:return '1'
    if n<=3:return '2-3'
    if n<=7:return '4-7'
    return '8+'
def family(name):
    n=re.sub(r'[^a-z0-9_]+','_',str(name).lower())
    if any(w in n for w in AUTH_WORDS):return 'AUTH'
    if any(w in n for w in VERIFY_WORDS):return 'VERIFY'
    if any(w in n for w in MUTATE_WORDS):return 'MUTATE'
    if any(w in n for w in EXEC_WORDS):return 'EXECUTE'
    if any(w in n for w in COMM_WORDS):return 'COMMUNICATE'
    if any(w in n for w in READ_WORDS):return 'READ'
    return 'TOOL_OTHER'
def resource_types(args):
    if isinstance(args,str):
        try:args=json.loads(args)
        except Exception:return ('text',) if args.strip() else ('none',)
    if not isinstance(args,dict):return ('none',)
    out=[]
    for k,v in args.items():
        lk=str(k).lower()
        if lk.endswith('_id') or lk in {'id','user_id','order_id','reservation_id','product_id','item_id'}:out.append(lk)
        elif any(x in lk for x in ('file','path','folder','dir','url','query','email','address','payment','date','name')):out.append(lk)
        elif isinstance(v,(list,dict)):out.append(lk+':struct')
    return tuple(sorted(set(out))[:4]) or ('none',)
def declared_tool_names(row):
    out=set()
    for t in row.get('tools') or []:
        if not isinstance(t,dict):continue
        n=(t.get('function') or {}).get('name') if isinstance(t.get('function'),dict) else t.get('name')
        if n:out.add(str(n))
    return out
def policy_flags(messages):
    sys='\n'.join(str(m.get('content') or '') for m in messages if m.get('role')=='system').lower()
    return {'confirm':bool(('confirm' in sys or 'confirmation' in sys) and ('before' in sys or 'prior' in sys)),'auth':bool('authenticate' in sys or 'authentication' in sys),'listed_only':bool('only use functions' in sys or 'only use tools' in sys or 'explicitly listed' in sys),'one_tool':bool('one tool call at a time' in sys or 'at most make one tool call' in sys)}
def message_text(m):
    x=m.get('content') if isinstance(m,dict) else ''
    return '' if x is None else str(x)
def tool_calls(m):
    out=[]
    for tc in (m.get('tool_calls') or []) if isinstance(m,dict) else []:
        f=tc.get('function') or {};out.append((str(f.get('name') or ''),f.get('arguments') or '{}'))
    fc=m.get('function_call') if isinstance(m,dict) else None
    if isinstance(fc,dict):out.append((str(fc.get('name') or ''),fc.get('arguments') or '{}'))
    return out
@dataclass
class Event:
    tid:str;dataset:str;pos:int;msg_idx:int;label:str;op:str;status:str;resources:tuple;atoms:tuple;hard:int;mask:int;text_len:int;tool_count:int;first_error:bool=False
UNRESOLVED_ERROR=1;NEED_VERIFY=2;PARTIAL_EVIDENCE=4;NEED_AUTH=8;NEED_CONFIRM=16;UNSUPPORTED=32;REPEAT_NO_PROGRESS=64

def compile_row(row,dataset):
    tid=f"{dataset}:{row.get('total_index',row.get('query_index','?'))}:{row.get('sample_index','0')}"
    msgs=row.get('messages') or [];labels={int(k):str(v) for k,v in (row.get('step_labels') or {}).items()};cand=sorted(i for i in labels if 0<=i<len(msgs))
    declared=declared_tool_names(row);policy=policy_flags(msgs);first_bad=next((i for i in cand if labels[i]==FIRST_ERROR),None)
    events=[];mask=NEED_AUTH if policy['auth'] else 0;last_op='START';last_res=('none',);last_status='NONE';repeats=0
    for pos,mi in enumerate(cand):
        m=msgs[mi];calls=tool_calls(m);content=message_text(m);prev_user=''
        for j in range(mi-1,-1,-1):
            if msgs[j].get('role')=='user':prev_user=message_text(msgs[j]);break
        result_text='';nxt=cand[pos+1] if pos+1<len(cand) else len(msgs)
        for j in range(mi+1,nxt):
            if msgs[j].get('role')=='tool':result_text+=' '+message_text(msgs[j])
        if calls:
            fams=[family(n) for n,_ in calls];op=fams[0] if len(set(fams))==1 else 'MULTI_TOOL';res=tuple(sorted(set(x for _,a in calls for x in resource_types(a))))[:5] or ('none',)
        else:
            res=('none',);op='ASK' if ASK_RE.search(content) else 'PLAN' if PLAN_RE.search(content) and not FINAL_RE.search(content) else 'FINAL'
        status='ERROR' if result_text and ERROR_RE.search(result_text) else 'PARTIAL' if result_text and PARTIAL_RE.search(result_text) else 'SUCCESS' if result_text else 'NO_RESULT' if calls else 'TEXT'
        hard=0
        if status=='SUCCESS':mask&=~UNRESOLVED_ERROR
        if status=='ERROR':mask|=UNRESOLVED_ERROR
        if status=='PARTIAL':mask|=PARTIAL_EVIDENCE
        if op in {'READ','VERIFY','AUTH'} and status=='SUCCESS':
            mask&=~PARTIAL_EVIDENCE
            if op=='VERIFY':mask&=~NEED_VERIFY
            if op=='AUTH':mask&=~NEED_AUTH
        if calls and policy['listed_only'] and any(n not in declared for n,_ in calls):mask|=UNSUPPORTED;hard+=4
        if policy['one_tool'] and len(calls)>1:hard+=3
        if op=='MUTATE':
            if policy['auth'] and mask&NEED_AUTH:hard+=4
            if policy['confirm'] and not YES_RE.search(prev_user):mask|=NEED_CONFIRM;hard+=4
            else:mask&=~NEED_CONFIRM
            mask|=NEED_VERIFY
        if op==last_op and res==last_res and status in {'ERROR','NO_RESULT'} and last_status in {'ERROR','NO_RESULT'}:repeats+=1;mask|=REPEAT_NO_PROGRESS;hard+=2+min(2,repeats)
        else:repeats=0;mask&=~REPEAT_NO_PROGRESS
        if op=='FINAL':
            if mask&UNRESOLVED_ERROR:hard+=5
            if mask&PARTIAL_EVIDENCE:hard+=3
            if mask&NEED_CONFIRM:hard+=4
            if mask&UNSUPPORTED:hard+=3
            if mask&NEED_VERIFY:hard+=1
        atoms=[f'OP:{op}',f'ST:{status}',f'MASK:{mask}',f'RES:{"+".join(res)}',f'CALLS:{bucket(len(calls))}',f'TLEN:{bucket(len(content))}',f'PREV:{last_op}',f'PREVST:{last_status}',f'HARD:{min(hard,6)}']
        if policy['confirm']:atoms.append('POLICY:CONFIRM')
        if policy['auth']:atoms.append('POLICY:AUTH')
        if policy['listed_only']:atoms.append('POLICY:LISTED')
        if YES_RE.search(prev_user):atoms.append('USER:CONFIRM')
        if ERROR_RE.search(content):atoms.append('TEXT:ERROR')
        if PARTIAL_RE.search(content):atoms.append('TEXT:PARTIAL')
        events.append(Event(tid,dataset,pos,mi,labels[mi],op,status,res,tuple(atoms),hard,mask,len(content),len(calls),mi==first_bad));last_op,last_res,last_status=op,res,status
    return {'tid':tid,'dataset':dataset,'events':events,'has_error':first_bad is not None,'first_error_msg':first_bad}
def load_all(root):
    out=[]
    for ds in DATASETS:
        with (Path(root)/ds/'test.jsonl').open(encoding='utf-8') as f:
            for line in f:
                if line.strip():out.append(compile_row(json.loads(line),ds))
    return out
def empirical(events,keyfn):
    n=Counter();p=Counter()
    for tr in events:
        for e in tr['events']:
            for k in keyfn(tr,e):n[k]+=1;p[k]+=int(e.first_error)
    return n,p
def loglift(k,n,p,base,alpha=.7):
    nn=n.get(k,0);pp=p.get(k,0);q=(pp+alpha)/(nn+2*alpha)
    return math.log(max(1e-9,q)/max(1e-9,base))+.08*math.log1p(nn)
def route_keys(tr,e,h=2,f=1):
    es=tr['events'];i=e.pos;hist=tuple(x.op for x in es[max(0,i-h):i]);fut=tuple(x.op for x in es[i+1:i+1+f])
    return [('R',hist,e.op,e.status,e.mask,fut),('R2',hist,e.op,e.mask,fut)]
def current_keys(tr,e):return [('A',a) for a in e.atoms]+[('C',e.op,e.status,e.mask),('C2',e.op,e.status)]
def node(e):return (e.op,e.status,e.resources[0] if e.resources else 'none')
def pstate(e):return (node(e),e.mask,min(e.hard,6))
def build_graph(traces,product=False):
    g=defaultdict(Counter);states=set()
    for tr in traces:
        es=tr['events']
        for e in es:states.add(pstate(e) if product else node(e))
        for a,b in zip(es,es[1:]):g[pstate(a) if product else node(a)][pstate(b) if product else node(b)]+=1
    return g,states
def unavoidable_states(g,states,bad):
    u=set(bad);changed=True
    while changed:
        changed=False
        for s in states-u:
            succ=set(g.get(s,{}))
            if succ and succ<=u:u.add(s);changed=True
    return u
def distance_to_bad(g,states,bad):
    rev=defaultdict(set)
    for a,vs in g.items():
        for b in vs:rev[b].add(a)
    d={s:0 for s in bad};q=deque(bad)
    while q:
        x=q.popleft()
        for y in rev.get(x,()):
            if y not in d:d[y]=d[x]+1;q.append(y)
    return d
def train_model(traces,cfg):
    all_events=[e for tr in traces for e in tr['events']];base=(sum(e.first_error for e in all_events)+1)/(len(all_events)+2)
    cn,cp=empirical(traces,current_keys);rn,rp=empirical(traces,lambda tr,e:route_keys(tr,e,cfg['hist'],cfg['future']));nn,np_=empirical(traces,lambda tr,e:[('N',node(e))]);pn,pp=empirical(traces,lambda tr,e:[('P',pstate(e)),('PM',(e.op,e.mask,min(e.hard,6)))])
    tn,tp=Counter(),Counter()
    for tr in traces:
        for a,b in zip(tr['events'],tr['events'][1:]):k=('T',pstate(a),pstate(b));tn[k]+=1;tp[k]+=int(b.first_error)
    ng,ns=build_graph(traces,False);pg,ps=build_graph(traces,True)
    badn={s for s in ns if (np_[('N',s)]+.7)/(nn[('N',s)]+1.4)>=cfg['bad_precision'] and nn[('N',s)]>=cfg['support']}
    badp={s for s in ps if s[2]>=cfg['hard_bad'] or ((pp[('P',s)]+.7)/(pn[('P',s)]+1.4)>=cfg['bad_precision'] and pn[('P',s)]>=cfg['support'])}
    return {'cfg':cfg,'base':base,'cn':cn,'cp':cp,'rn':rn,'rp':rp,'nn':nn,'np':np_,'pn':pn,'pp':pp,'tn':tn,'tp':tp,'ng':ng,'pg':pg,'un':unavoidable_states(ng,ns,badn),'up':unavoidable_states(pg,ps,badp),'dn':distance_to_bad(ng,ns,badn),'dp':distance_to_bad(pg,ps,badp)}
def scores(tr,m,kind):
    out=[];base=m['base'];cfg=m['cfg']
    for i,e in enumerate(tr['events']):
        if kind=='current':
            vals=[loglift(k,m['cn'],m['cp'],base) for k in current_keys(tr,e)];sc=sum(sorted(vals,reverse=True)[:cfg['topk']])
        elif kind=='ueta':sc=sum(loglift(k,m['rn'],m['rp'],base) for k in route_keys(tr,e,cfg['hist'],cfg['future']))+.15*e.hard
        elif kind=='graph':
            s=node(e);sc=loglift(('N',s),m['nn'],m['np'],base)+(cfg['unavoidable_bonus'] if s in m['un'] else 0)+(cfg['distance_bonus']/(1+m['dn'][s]) if s in m['dn'] else 0)
        elif kind in {'product','product_no_topology'}:
            s=pstate(e);sc=sum(loglift(k,m['pn'],m['pp'],base) for k in [('P',s),('PM',(e.op,e.mask,min(e.hard,6)))])+cfg['hard_weight']*e.hard
            if i:sc+=cfg['transition_weight']*loglift(('T',pstate(tr['events'][i-1]),s),m['tn'],m['tp'],base)
            if kind=='product':sc+=(cfg['unavoidable_bonus'] if s in m['up'] else 0)+(cfg['distance_bonus']/(1+m['dp'][s]) if s in m['dp'] else 0)
        out.append(sc)
    return out
def predict(tr,m,kind):
    ss=scores(tr,m,kind)
    if not ss:return -1
    if kind=='product':
        ps=[pstate(e) for e in tr['events']]
        for i,s in enumerate(ps):
            if s in m['up'] and (i==0 or ps[i-1] not in m['up']):return i
    return max(range(len(ss)),key=lambda i:(ss[i],-i))
def acc(pred,gold):return sum(a==b for a,b in zip(pred,gold))/len(gold) if gold else 0.
def near(pred,gold,r=1):return sum(abs(a-b)<=r for a,b in zip(pred,gold))/len(gold) if gold else 0.
def evaluate(traces,m,kind):
    xs=[tr for tr in traces if tr['has_error'] and tr['events']];gold=[next(i for i,e in enumerate(tr['events']) if e.first_error) for tr in xs];pred=[predict(tr,m,kind) for tr in xs]
    return {'n':len(xs),'exact':acc(pred,gold),'near1':near(pred,gold,1),'pred':pred,'gold':gold,'tids':[tr['tid'] for tr in xs],'datasets':[tr['dataset'] for tr in xs]}
CONFIGS=[{'hist':h,'future':f,'bad_precision':bp,'support':2,'hard_bad':4,'topk':4,'hard_weight':.55,'transition_weight':.45,'unavoidable_bonus':1.3,'distance_bonus':.7} for h in (1,2) for f in (0,1) for bp in (.35,.5)]
def choose(train,val):
    best=None
    for cfg in CONFIGS:
        ev=evaluate(val,train_model(train,cfg),'product');key=(ev['exact'],ev['near1'],-cfg['future'],cfg['hist'])
        if best is None or key>best[0]:best=(key,cfg)
    return best[1]
def hashfold(tid,k=5):return stable_int(tid)%k
def fold_eval(traces):
    kinds=('current','ueta','graph','product_no_topology','product');aggregate={k:{'pred':[],'gold':[],'tids':[],'datasets':[]} for k in kinds};folds=[]
    for f in range(5):
        test=[t for t in traces if hashfold(t['tid'])==f];rest=[t for t in traces if hashfold(t['tid'])!=f];val=[t for t in rest if stable_int('v:'+t['tid'])%5==0];train=[t for t in rest if t not in val];cfg=choose(train,val);m=train_model(rest,cfg);fr={'fold':f,'train':len(rest),'test':len(test),'cfg':cfg}
        for kind in kinds:
            e=evaluate(test,m,kind);fr[kind]={'n':e['n'],'exact':e['exact'],'near1':e['near1']}
            for z in ('pred','gold','tids','datasets'):aggregate[kind][z]+=e[z]
        folds.append(fr)
    return {k:{'n':len(v['gold']),'exact':acc(v['pred'],v['gold']),'near1':near(v['pred'],v['gold'],1)} for k,v in aggregate.items()},aggregate,folds
def dataset_holdout(traces):
    out=[]
    for ds in DATASETS:
        test=[t for t in traces if t['dataset']==ds];rest=[t for t in traces if t['dataset']!=ds];val=[t for t in rest if stable_int('h:'+t['tid'])%5==0];train=[t for t in rest if t not in val];cfg=choose(train,val);m=train_model(rest,cfg);row={'heldout':ds,'train':len(rest),'test':len(test),'cfg':cfg}
        for kind in ('current','ueta','graph','product_no_topology','product'):
            e=evaluate(test,m,kind);row[kind]={'n':e['n'],'exact':e['exact'],'near1':e['near1']}
        out.append(row)
    return out
def bootstrap(agg,a='product',b='current',n=4000):
    A=agg[a];B=agg[b];vals=[];N=len(A['gold']);rng=random.Random(SEED)
    for _ in range(n):
        idx=[rng.randrange(N) for _ in range(N)];vals.append(sum(A['pred'][i]==A['gold'][i] for i in idx)/N-sum(B['pred'][i]==B['gold'][i] for i in idx)/N)
    vals.sort();return {'mean':statistics.mean(vals),'lo':vals[int(.025*n)],'hi':vals[int(.975*n)]}
def counterfactual_auc(traces,m):
    ys=[];ss=[]
    for tr in traces:
        if len(tr['events'])<3:continue
        ys.append(1);ss.append(max(0.,10.-sum(e.hard for e in tr['events'])));es=list(tr['events']);j=next((i for i,e in enumerate(es[:-1]) if e.op in {'READ','VERIFY','AUTH'} and e.status=='SUCCESS'),1);mut=es[:j]+es[j+1:];novelty=sum(1 for a,b in zip(mut,mut[1:]) if pstate(b) not in m['pg'].get(pstate(a),{}));ys.append(0);ss.append(max(0.,10.-sum(e.hard for e in mut)-novelty))
    order=sorted(range(len(ss)),key=lambda i:ss[i]);rank=[0]*len(ss)
    for r,i in enumerate(order,1):rank[i]=r
    pos=[rank[i] for i,y in enumerate(ys) if y==1];neg=sum(1 for y in ys if y==0);P=len(pos);auc=(sum(pos)-P*(P+1)/2)/(P*neg) if P and neg else .5
    return {'n_pairs':P,'auc':auc}
def checkpoint_summary(traces,cfg):
    m=train_model(traces,cfg)
    def pack(c):return [[repr(k),v] for k,v in c.items()]
    return {'architecture':'Product-Lifted Obligation Graph','seed':SEED,'config':cfg,'base':m['base'],'workflow_nodes':len(m['ng']),'workflow_edges':sum(len(v) for v in m['ng'].values()),'product_states':len(m['pg']),'product_edges':sum(len(v) for v in m['pg'].values()),'unavoidable_product_states':len(m['up']),'obligation_bits':7,'no_dense_or_embedding':True,'counters':{'product_total':pack(m['pn']),'product_positive':pack(m['pp']),'transition_total':pack(m['tn']),'transition_positive':pack(m['tp'])}}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True);t0=time.perf_counter();traces=load_all(args.data);load_s=time.perf_counter()-t0
    metrics,agg,folds=fold_eval(traces);holds=dataset_holdout(traces);boot=bootstrap(agg);train=[t for t in traces if stable_int('all:'+t['tid'])%5];val=[t for t in traces if stable_int('all:'+t['tid'])%5==0];cfg=choose(train,val);full=train_model(traces,cfg);cf=counterfactual_auc(traces,full);checkpoint=checkpoint_summary(traces,cfg);cp_path=out/'PRODUCT_LIFTED_SINGLE_CHECKPOINT.json.gz'
    with gzip.open(cp_path,'wb') as f:f.write(json.dumps(checkpoint,separators=(',',':')).encode())
    rows=[]
    for kind,v in agg.items():
        for tid,ds,p,g in zip(v['tids'],v['datasets'],v['pred'],v['gold']):rows.append({'model':kind,'tid':tid,'dataset':ds,'pred_position':p,'gold_position':g,'correct':p==g,'near1':abs(p-g)<=1})
    with (out/'predictions.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    hold_deltas=[r['product']['exact']-r['current']['exact'] for r in holds]
    strict={'oof_plus_5pp':metrics['product']['exact']>=metrics['current']['exact']+.05,'bootstrap_lower_positive':boot['lo']>0,'all_dataset_holdouts_nonnegative':all(x>=-1e-12 for x in hold_deltas),'mean_dataset_holdout_positive':statistics.mean(hold_deltas)>0,'product_beats_trace_ueta':metrics['product']['exact']>metrics['ueta']['exact'],'topology_ablation_positive':metrics['product']['exact']>metrics['product_no_topology']['exact'],'counterfactual_auc_075':cf['auc']>=.75}
    summary={'benchmark':'AgentProcessBench','trajectories':len(traces),'error_trajectories':metrics['product']['n'],'events':sum(len(t['events']) for t in traces),'datasets':dict(Counter(t['dataset'] for t in traces)),'load_seconds':load_s,'oof':metrics,'folds':folds,'dataset_holdout':holds,'bootstrap_product_minus_current':boot,'counterfactual':cf,'checkpoint_bytes':cp_path.stat().st_size,'checkpoint_graph':{k:checkpoint[k] for k in ('workflow_nodes','workflow_edges','product_states','product_edges','unavoidable_product_states','obligation_bits')},'strict':strict,'overall_pass':all(strict.values())}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8');(out/'README.txt').write_text('Product-Lifted Obligation Graph strict AgentProcessBench falsification.\nNo embeddings, dense latent vectors, pretrained encoders, LLM calls, TF-IDF/SVD, nearest-vector retrieval, code generation, or external solvers.\n',encoding='utf-8');print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=='__main__':main()

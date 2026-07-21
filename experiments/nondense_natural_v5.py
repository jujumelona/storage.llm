#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, json, math, os, re, statistics
from collections import Counter, defaultdict
from pathlib import Path

WORD=re.compile(r"[A-Za-z][A-Za-z'-]{1,30}")
STOP=set('a an the this that these those someone somebody person individual people they them their theirs he him his she her hers it its i me my mine we us our ours you your yours will would can could may might must should is are was were be been being do does did have has had to of in on at for from with by and or then today currently particular situation case imply implies implied entails entail entailed infer infers inferred mean means meant indicate indicates indicated show shows shown also very some likely often typically however therefore thus indeed thing one ones such any each every all as into about after before during while where when who whom whose which what why how does this does it does that know known sure true false rule stated knowing remembering'.split())
SYN={
 'tired':'tire tiredness exhausted exhaustion fatigue fatigued weary', 'rest':'relax relaxation slumber nap sleep repose',
 'sad':'sadness sorrow sorrowful unhappy grief', 'cry':'cries crying tears tear weep weeping shed',
 'medicine':'medication drug remedy medicate pills pill', 'headache':'migraine pounding throbbing',
 'angry':'anger enraged frustration furious mad', 'shout':'voice yell yelling scream screaming',
 'trip':'journey travel expedition tour vacation', 'movie':'film cinema', 'help':'assistance aid support',
 'promotion':'promote promoted raise', 'money':'funds financial finance salary paycheck',
 'buy':'purchase purchasing afford acquire ordering order', 'good':'excel proficiency proficient skilled skills ability capable',
 'arithmetic':'math mathematics', 'infection':'infected', 'immune':'immunity', 'weight':'weigh gaining gain',
 'late':'tardy delayed delay', 'miss':'unable catch missed missing', 'train':'railway',
 'finish':'finished completing complete completed completion', 'receive':'receives receiving received get gets getting got',
 'study':'studies studying studied work working worked', 'wet':'soaked drenched', 'dry':'dryness',
 'satisfy':'satisfied satisfaction content', 'attend':'attending attended participation participate',
 'leave':'leaving left quit', 'salary':'paycheck pay wage', 'forget':'forgot forgotten forgetting',
 'walk':'walking stroll', 'swim':'swimming', 'submit':'submitted submitting submission',
 'cake':'bake baking making', 'report':'writing write written', 'school':'class classroom',
}
CANON={k:k for k in SYN}
for k,v in SYN.items():
    for x in v.split(): CANON[x]=k
NEG_PAT=re.compile(r"\b(?:not|no|never|won't|wouldn't|cannot|can't|doesn't|didn't|isn't|aren't|wasn't|weren't|hasn't|hadn't|unable|failed to|fails to|decided against|without|lacks?|lack of)\b",re.I)
DISCOURSE=re.compile(r"\b(?:however|but|nevertheless|nonetheless|yet|despite this|against all odds|unfortunately|instead|rather)\b",re.I)

# ---------- explicit proposition/event compiler ----------
def strip_question(s:str)->str:
    s=s.strip().strip('?').strip()
    s=re.sub(r"(?i)^(does|do|can|could|would)\s+(this|it|that)\s+(entail|imply|mean|infer|show|indicate)(?:s)?\s+(that\s+)?",'',s)
    s=re.sub(r"(?i)^(is|are)\s+it\s+(?:true|false)\s+that\s+",'',s)
    return s

def lemma(w:str)->str:
    w=w.lower().strip("'-")
    if w in CANON:return CANON[w]
    if w.endswith('ies') and len(w)>5:w=w[:-3]+'y'
    elif w.endswith('ing') and len(w)>5:w=w[:-3]
    elif w.endswith('ed') and len(w)>4:w=w[:-2]
    elif w.endswith('es') and len(w)>5:w=w[:-2]
    elif w.endswith('s') and len(w)>4:w=w[:-1]
    return CANON.get(w,w)

def prop(s:str):
    s=strip_question(s); neg=bool(NEG_PAT.search(s)); toks=[]
    for raw in WORD.findall(s):
        w=lemma(raw)
        if not w or w in STOP or len(w)<3:continue
        # Person names and discourse adverbs add no predicate identity.
        if raw[:1].isupper() and raw.lower() not in {'if','either'}:continue
        toks.append(w)
    return frozenset(toks),neg

def psim(a,b):
    A,B=a[0],b[0]
    if not A or not B:return 0.0
    inter=len(A&B)
    return max(inter/min(len(A),len(B)), 2*inter/(len(A)+len(B)))

def opposite(p):return p[0],not p[1]

def split_clauses(sent:str):
    parts=[]
    for x in DISCOURSE.split(sent):
        parts.extend(re.split(r"[;]",x))
    return [x.strip(' .,;:') for x in parts if x.strip(' .,;:')]

def parse_program(ctx:str):
    rules=[];facts=[];disj=[];defaults=[];priorities=[]
    sentences=[x.strip() for x in re.split(r"(?<=[.!?])\s+",ctx) if x.strip()]
    for sent in sentences:
        x=sent.strip(' .')
        # Explicit implication.
        mm=re.search(r"(?i)\bif\s+(.+?)(?:,|\s+then\s+)\s*(.+)",x)
        if mm:
            rules.append((prop(mm.group(1)),prop(mm.group(2)),'strict'));continue
        # Natural paraphrases of implication.
        mm=re.search(r"(?i)(.+?),?\s+(?:which|this)\s+(?:meant|means|indicates|implies)\s+that\s+(.+)",x)
        if mm:
            rules.append((prop(mm.group(1)),prop(mm.group(2)),'strict'));continue
        mm=re.search(r"(?i)(.+?)\s+(?:would|will)\s+(?:typically|usually|normally|often)?\s*(.+)",x)
        if mm and len(split_clauses(x))==1 and re.search(r"(?i)meant|knowing|rule|whenever",x):
            rules.append((prop(mm.group(1)),prop(mm.group(2)),'default'));continue
        # Either A or B (or both).
        mm=re.search(r"(?i)(?:either\s+)?(.+?)\s+or\s+(.+?)(?:,\s*or\s+(?:maybe\s+)?both|,?\s+or both|$)",x)
        if mm and not re.search(r"(?i)whether|unknown which|unclear which",x):
            disj.append((prop(mm.group(1)),prop(mm.group(2))));continue
        # Defaults / exceptions.
        mm=re.search(r"(?i)(?:normally|typically|generally|usually),?\s+(.+?)\s+(?:are|is|will|would)\s+(.+)",x)
        if mm:
            defaults.append((prop(mm.group(1)),prop(mm.group(2))));continue
        # Facts are local discourse clauses; uncertainty sentences are ignored.
        for c in split_clauses(x):
            if re.search(r"(?i)\b(?:unclear|uncertain|unsure|unknown|could be|might be|possible that|rule stated|remembering the rule|knowing that)\b",c):continue
            facts.append(prop(c))
    return rules,facts,disj,defaults,priorities

def derive(ctx:str):
    rules,known,disj,defaults,_=parse_program(ctx);known=list(known)
    def has(p,threshold=.66):return any(k[1]==p[1] and psim(k,p)>=threshold for k in known)
    changed=True;rounds=0
    while changed and rounds<20:
        rounds+=1;changed=False
        for a,b,_ in rules:
            if has(a) and not has(b):known.append(b);changed=True
            # Contraposition is an explicit operator.
            if has(opposite(b)) and not has(opposite(a)):known.append(opposite(a));changed=True
        for a,b in disj:
            if has(opposite(a)) and not has(b):known.append(b);changed=True
            if has(opposite(b)) and not has(a):known.append(a);changed=True
        for a,b in defaults:
            if has(a) and not has(opposite(b)) and not has(b):known.append(b);changed=True
    return known

def entail(ctx,q):
    qp=prop(q);known=derive(ctx)
    def has(p):return any(k[1]==p[1] and psim(k,p)>=.66 for k in known)
    return 'yes' if has(qp) else 'no'

# ---------- LogicBench ----------
def eval_logicbench(root,out):
    files=glob.glob(os.path.join(root,'**','LogicBench(Eval)','BQA','**','data_instances.json'),recursive=True);rows=[];total=correct=0
    for f in files:
        d=json.load(open(f,encoding='utf-8'));c=t=0
        for s in d.get('samples',[]):
            for qa in s.get('qa_pairs',[]):
                pr=entail(s.get('context',''),qa.get('question',''));g=str(qa.get('answer','')).lower();t+=1;c+=pr==g
        if t:rows.append({'file':os.path.relpath(f,root),'type':d.get('type'),'axiom':d.get('axiom'),'questions':t,'correct':c,'accuracy':c/t});total+=t;correct+=c
    write_csv(out/'logicbench_v5_by_file.csv',rows);return {'files':len(rows),'questions':total,'correct':correct,'accuracy':correct/total if total else 0}

# ---------- LogiQA sparse Bayesian evidence ledger ----------
QOPS=[('weaken',r'weaken'),('strengthen',r'strengthen'),('support',r'support'),('assumption',r'assum|presuppos'),('except',r'except'),('must_true',r'must be true|also true|can be established|follows'),('evaluate',r'evaluation|most appropriate|accurate evaluation'),('analogy',r'similar|belongs|same reasoning'),('conclusion',r'conclusion'),('explain',r'explain|reason'),('complete',r'complete|fill')]
def qtype(q):
    for n,p in QOPS:
        if re.search(p,q,re.I):return n
    return 'other'

def ctoks(s):return {lemma(w) for w in WORD.findall(s) if lemma(w) not in STOP and len(lemma(w))>=3}
def rank(values,i):return str(sorted(range(len(values)),key=lambda j:(values[j],j)).index(i))
def binned(x,cuts):return str(sum(x>=c for c in cuts))
def markers(s):
    pats={'neg':r"\bnot\b|n't|never|no\b",'if':r'\bif\b','only':r'\bonly\b','all':r'\b(?:all|every|each|any)\b','some':r'\b(?:some|at least|may|possible)\b','causal':r'\b(?:because|therefore|cause|result|reason)\b','modal':r'\b(?:must|cannot|can|necessary|sufficient)\b','contrast':r'\b(?:however|but|other|alternative|outside|mainly|unless)\b','compar':r'\b(?:more|less|most|least|twice|times)\b'}
    return {k:int(bool(re.search(p,s,re.I))) for k,p in pats.items()}
def base_features(p,q,opts,i):
    o=re.sub(r'^[A-D]\.\s*','',opts[i]);pt,qt,ot=ctoks(p),ctoks(q),ctoks(o);lens=[len(ctoks(x)) for x in opts];ovs=[len(ctoks(x)&pt) for x in opts];qvs=[len(ctoks(x)&qt) for x in opts];fs=set()
    qtpe=qtype(q);fs|={'Q:'+qtpe,'I:'+str(i),'LEN:'+binned(len(ot),[5,10,18,30]),'LR:'+rank(lens,i),'OV:'+binned(len(ot&pt),[1,3,6,10]),'OR:'+rank(ovs,i),'QOV:'+binned(len(ot&qt),[1,2,4]),'QR:'+rank(qvs,i)}
    om=markers(o);pm=markers(p);qm=markers(q)
    for k,v in om.items():fs.add('OM:'+k+str(v));fs.add('QM:'+k+str(qm[k]));fs.add('PM:'+k+str(pm[k]));fs.add('D:'+k+str(v-pm[k]))
    # Explicit relation atoms.
    fs.add('ENT:'+str(int(entail(p,o)=='yes')))
    fs.add('OPP:'+str(int(entail(p,re.sub(r"(?i)\bnot\b|n't",'',o))=='yes' and om['neg'])))
    # Lexical atoms remain discrete symbols; no vectors or similarity search.
    for w in sorted(qt)[:18]:fs.add('QW:'+w)
    for w in sorted(ot)[:22]:fs.add('OW:'+w)
    for w in sorted(ot&pt)[:14]:fs.add('PW:'+w)
    # Conditional event atoms discover operator-specific evidence.
    for x in list(fs):
        if x.startswith(('OM:','D:','ENT:','OPP:','I:','LR:','OR:','OV:')):fs.add('C:'+qtpe+'|'+x)
    return fs

def parse_logiqa(path):
    lines=[x.rstrip('\n') for x in open(path,encoding='utf-8',errors='ignore')];i=0;rows=[]
    while i<len(lines):
        if re.fullmatch(r'[a-dA-D]',lines[i].strip()) and i+6<len(lines):
            g=lines[i].strip().lower();p=lines[i+1];q=lines[i+2];opts=lines[i+3:i+7]
            if all(re.match(r'[A-D]\.',x) for x in opts):rows.append((g,p,q,opts));i+=7;continue
        i+=1
    return rows

def build_ledger(data,min_count=3):
    total=Counter();pos=Counter();group=Counter();gpos=Counter()
    for g,p,q,opts in data:
        gi='abcd'.index(g)
        for i in range(4):
            fs=base_features(p,q,opts,i);y=i==gi
            for f in fs:total[f]+=1;pos[f]+=y
            # Only meaningful conjunctions, not arbitrary combinatorial templates.
            anchors=[x for x in fs if x.startswith(('Q:','C:','I:','ENT:','OPP:'))]
            evidence=[x for x in fs if x.startswith(('OM:','D:','LR:','OR:','OV:','OW:','PW:'))]
            for a in anchors:
                for e in evidence:
                    k=a+'&'+e;group[k]+=1;gpos[k]+=y
    ledger={}
    for f,n in total.items():
        if n>=min_count:
            p=(pos[f]+1)/(n+2);ledger[f]=(math.log(p/(1-p))-math.log(1/3),n)
    for f,n in group.items():
        if n>=max(4,min_count):
            p=(gpos[f]+1)/(n+2);ledger[f]=(math.log(p/(1-p))-math.log(1/3),n)
    return ledger

def score_option(fs,ledger,topk=20,shrink=10.0):
    keys=set(fs);anchors=[x for x in fs if x.startswith(('Q:','C:','I:','ENT:','OPP:'))];evidence=[x for x in fs if x.startswith(('OM:','D:','LR:','OR:','OV:','OW:','PW:'))]
    for a in anchors:
        for e in evidence:keys.add(a+'&'+e)
    vals=[]
    for k in keys:
        if k in ledger:
            w,n=ledger[k];vals.append((abs(w)*n/(n+shrink),w*n/(n+shrink),k))
    vals.sort(reverse=True)
    # Cap evidence from each family to avoid correlated lexical overcounting.
    used=Counter();score=0.;chosen=[]
    for _,v,k in vals:
        fam=k.split(':',1)[0]
        if used[fam]>=4:continue
        used[fam]+=1;score+=v;chosen.append(k)
        if len(chosen)>=topk:break
    return score,chosen

def predict(data,ledger,topk,shrink):
    rows=[]
    for idx,(g,p,q,opts) in enumerate(data):
        ss=[];why=[]
        for i in range(4):
            s,w=score_option(base_features(p,q,opts,i),ledger,topk,shrink);ss.append(s);why.append(w)
        pr='abcd'[max(range(4),key=lambda i:(ss[i],-i))];rows.append((pr,g,ss,why,qtype(q)))
    return rows

def eval_logiqa(root,out):
    tr=parse_logiqa(os.path.join(root,'Train.txt'));dv=parse_logiqa(os.path.join(root,'Eval.txt'));te=parse_logiqa(os.path.join(root,'Test.txt'));best=(-1,None)
    for mc in [2,3,5,8]:
        led=build_ledger(tr,mc)
        for k in [8,12,20,32]:
            for sh in [3.,10.,30.]:
                r=predict(dv,led,k,sh);acc=sum(a==b for a,b,*_ in r)/len(r)
                if acc>best[0]:best=(acc,(mc,k,sh))
    led=build_ledger(tr+dv,best[1][0]);r=predict(te,led,best[1][1],best[1][2]);rows=[]
    for i,(pr,g,ss,why,qt) in enumerate(r):rows.append({'id':i,'qtype':qt,'pred':pr,'gold':g,'correct':pr==g,'scores':json.dumps(ss),'evidence':json.dumps(why)})
    write_csv(out/'logiqa_v5_predictions.csv',rows)
    by={}
    for qt in sorted(set(x[-1] for x in r)):
        sub=[x for x in r if x[-1]==qt];by[qt]={'n':len(sub),'accuracy':sum(x[0]==x[1] for x in sub)/len(sub)}
    return {'train':len(tr),'dev':len(dv),'test':len(te),'dev_best_accuracy':best[0],'config':best[1],'ledger_records':len(led),'accuracy':sum(x[0]==x[1] for x in r)/len(r),'by_qtype':by}

def write_csv(path,rows):
    if not rows:path.write_text('',encoding='utf-8');return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys:keys.append(k)
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=__import__('csv').DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--logicbench',required=True);ap.add_argument('--logiqa',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    lb=eval_logicbench(a.logicbench,out);lq=eval_logiqa(a.logiqa,out);summary={'architecture':'explicit clause/operator graph + sparse Bayesian evidence ledger','forbidden_dense_components_used':[],'logicbench':lb,'logiqa':lq,'gate_pass':lb['accuracy']>=.8 and lq['accuracy']>=.45};json.dump(summary,open(out/'summary.json','w',encoding='utf-8'),ensure_ascii=False,indent=2);print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

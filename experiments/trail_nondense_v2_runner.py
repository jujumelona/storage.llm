#!/usr/bin/env python3
import json, os, re, runpy
from collections import defaultdict
from pathlib import Path
import trail_nondense_full_eval as m

ANCHORS={
'format.syntax':['syntaxerror','jsondecodeerror','unterminated string','invalid json','parse error','malformed'],
'format.arguments':['unexpected keyword argument','missing required','invalid argument','invalid parameter','takes no arguments','tool.parameters','kwargs','arguments'],
'format.required_marker':['missing','did not','required format','formatting error'],
'tool.unsupported':['unsupported','not supported','incompatible','wrong tool','cannot convert','tool selection'],
'tool.disallowed_operation':['not permitted','disallowed','tried to execute open'],
'tool.definition':['tool definition','description','schema','defined as'],
'env.setup':['modulenotfounderror','no module named','permissionerror','permission denied','dependency','environment setup','api key'],
'api.429':['429','rate limit','too many requests'],
'api.auth':['401','403','unauthorized','unauthorised','authentication','invalid token','access denied'],
'api.service':['service unavailable','connectionerror','connection refused','connection reset','internal server error'],
'resource.not_found':['404','filenotfounderror','no such file','resource not found','does not exist','missing file'],
'resource.exhaustion':['out of memory','memoryerror','resource exhausted','no space left','cuda out of memory','oom'],
'resource.timeout':['timeout','timed out','deadline exceeded','took too long'],
'output.misread':['misinterpret','incorrectly assumed','incorrectly concluded','incorrectly interpreted','no results','empty output'],
'retrieval.poor':['irrelevant search','irrelevant query','irrelevant result','broad search','broad web search','poor information retrieval','failed to search','failed to check','failed to verify'],
'problem.misidentified':['misunderstood','incorrect problem','wrong task','wrong question','wrong target'],
'instruction.explicit':['did not follow','failed to comply','failed to adhere','instruction non-compliance','instead of providing','instead of performing','instead of returning'],
'context.explicit':['did not learn','did not adapt','did not incorporate','did not remember','ignored the previous','ignored previous','forgot','context window','state tracking'],
'resource.abuse.explicit':['resource abuse','excessive tool calls','excessively called','repeatedly invoked','repeatedly called','repeatedly retried'],
'goal.explicit':['goal deviation','abandoned its plan','abandoned the plan','abandoned its task','prematurely stopped','prematurely terminated','premature final','deviated from','gave up'],
'orchestration.explicit':['task orchestration','failed to coordinate','duplicate subtask','progress monitoring','subtask coordination'],
'memory.explicit':['incorrect memory','stale memory','wrong memory','recalled incorrectly'],
'language.explicit':['typo','misspell','fabricated fact','fabricated name','fabricated value','hallucinated'],
'tool.fabrication':['fabricated tool','claimed the tool','nonexistent tool']}

STOP=set('a an the this that these those someone somebody person individual people they he she it their his her will would can could may might is are was were be been being do does did have has had to of in on at for from with by and or then today currently particular situation case imply implies implied entails entail entailed mean means meant also very some likely often typically does this does it does that'.split())
SYN={'tired':'exhaust exhausted fatigue fatigued weary tiredness','rest':'relax relaxation slumber nap sleep','sad':'sorrow sorrowful sadness unhappy','cry':'tears tear weep shed crying','medicine':'medication drug remedy medicate','headache':'migraine pounding throbbing','angry':'anger enraged frustration furious','shout':'voice yell scream shouting','trip':'journey travel expedition tour','movie':'film cinema','help':'assistance aid support','promotion':'promote promoted','money':'funds financial finance','buy':'purchase afford acquire','good':'excel proficiency proficient skilled skills ability','arithmetic':'math mathematics','infection':'infected','immune':'immunity','weight':'weigh gain','late':'tardy','miss':'unable catch','train':'railway','not':'no never cannot unable'}
CANON={k:k for k in SYN}
for k,v in SYN.items():
    for x in v.split(): CANON[x]=k
NEG_RE=re.compile(r"\b(not|no|never|won't|wouldn't|cannot|can't|doesn't|isn't|aren't|wasn't|weren't|unable|fails? to|lack(?:s|ing)?)\b",re.I)


def select_local(s):
    attrs=s.get('span_attributes') or {}; kind=str(attrs.get('openinference.span.kind') or s.get('span_kind') or '').upper(); selected={}
    if kind=='LLM':
        for k,v in attrs.items():
            lk=k.lower()
            if 'output_messages' in lk or lk=='output.value' or lk.startswith('llm.output'): selected[k]=v
    elif kind=='TOOL':
        for k,v in attrs.items():
            lk=k.lower()
            if lk in {'input.value','output.value','tool.name','tool.description','tool.parameters'} or lk.startswith('tool.'): selected[k]=v
    elif kind in {'AGENT','CHAIN'}:
        for k,v in attrs.items():
            if k.lower()=='output.value': selected[k]=v
    logs=[]
    for x in s.get('logs') or []:
        body=m.scrub(x.get('body')); raw=json.dumps(body,ensure_ascii=False,default=str)
        if isinstance(body,dict):
            name=body.get('function.name'); output=body.get('function.output'); b={'function.name':name,'function.output':output}
            if m.ERROR_WORDS.search(str(output or '')): b['function.arguments']=body.get('function.arguments')
            logs.append(b)
        elif m.ERROR_WORDS.search(raw): logs.append(body)
    return json.dumps({'name':s.get('span_name'),'kind':kind,'status_code':s.get('status_code'),'status_message':s.get('status_message'),'attributes':selected,'events':m.scrub(s.get('events')),'logs':logs},ensure_ascii=False,sort_keys=True,default=str)

def fast_load(path):
    d=json.load(open(path,encoding='utf-8')); tid=str(d.get('trace_id') or Path(path).stem); ev=[]
    for s in m.flatten_spans(d.get('spans') or []):
        text=select_local(s); low=text.lower(); name=str(s.get('span_name') or '')
        ev.append(m.Event(tid,str(s.get('span_id') or ''),str(s.get('parent_span_id') or ''),str(s.get('timestamp') or ''),name,str(s.get('span_kind') or ''),text,frozenset(),m.tool_name(text,name),bool(m.FINAL_WORDS.search(low) or 'final' in name.lower()),bool(m.PLAN_WORDS.search(low) or 'plan' in name.lower()),bool(m.ERROR_WORDS.search(low))))
    ev.sort(key=lambda x:(x.timestamp,x.span_id)); return tid,ev

def candidate_ids(e):
    low=e.text.lower(); ids=set()
    for i,(_,rid,_,_) in enumerate(m.COMPILED_RULES):
        if any(a in low for a in ANCHORS.get(rid,())): ids.add(i)
        if rid=='api.service' and re.search(r'\b50[0-9]\b',low): ids.add(i)
    return ids

def direct(events,indexed=True):
    out=[]
    for e in events:
        ids=candidate_ids(e) if indexed else range(len(m.COMPILED_RULES)); best=None
        for i in ids:
            cat,rid,conf,pats=m.COMPILED_RULES[i]
            if conf<0.88: continue
            if any(p.search(e.text) for p in pats):
                p=m.Prediction(cat,e.span_id,rid,m.excerpt(e.text),conf)
                if best is None or p.confidence>best.confidence: best=p
        if best: out.append(best)
    return out

def fast_sequence(events):
    out=[]; failures=[]; by_sig=defaultdict(list); query_seen={}
    for i,e in enumerate(events):
        sig=m.normalize_signature(e)
        if e.has_error: failures.append(i); by_sig[sig].append(i)
        if e.is_final and failures:
            j=failures[-1]; prior=events[j]
            if not any((x.tool and x.tool!=prior.tool and not x.has_error) for x in events[j+1:i]): out.append(m.Prediction('Goal Deviation',e.span_id,'sequence.final_after_failure',m.excerpt(e.text),0.91))
        if 'search' in e.tool:
            for q in re.findall(r'"(?:query|q)"\s*:\s*"([^"]+)"',e.text,re.I):
                nq=re.sub(r'\W+',' ',q.lower()).strip()
                if nq in query_seen: out.append(m.Prediction('Poor Information Retrieval',e.span_id,'sequence.repeated_query',m.excerpt(e.text),0.9))
                query_seen[nq]=i
    for sig,ix in by_sig.items():
        if len(ix)>=2:
            out.append(m.Prediction('Resource Abuse',events[ix[-1]].span_id,'sequence.repeated_failure',m.excerpt(events[ix[-1]].text),0.96))
            out.append(m.Prediction('Context Handling Failures',events[ix[1]].span_id,'sequence.not_adapted',m.excerpt(events[ix[1]].text),0.91))
    return out

def extract_claim(s):
    s=s.strip().strip('?')
    s=re.sub(r'^(does|do|can|could|would)\s+(this|it|that)\s+(entail|imply|mean|show|indicate)(?:s)?\s+(that\s+)?','',s,flags=re.I)
    s=re.sub(r'^(is|are)\s+it\s+(true|false)\s+that\s+','',s,flags=re.I)
    if re.search(r'\bthat\b',s,re.I) and any(x in s.lower() for x in ['entail','imply','mean']): s=re.split(r'\bthat\b',s,flags=re.I)[-1]
    return s

def prop(s):
    s=extract_claim(s); neg=bool(NEG_RE.search(s)); toks=[]
    for t in re.findall(r'[a-z]+',s.lower()):
        if t in STOP or len(t)<3: continue
        if t.endswith('ing') and len(t)>5:t=t[:-3]
        elif t.endswith('ed') and len(t)>4:t=t[:-2]
        elif t.endswith('es') and len(t)>5:t=t[:-2]
        elif t.endswith('s') and len(t)>4:t=t[:-1]
        toks.append(CANON.get(t,t))
    return frozenset(toks),neg

def sim(a,b):
    A=a[0];B=b[0]
    if not A or not B:return 0
    return len(A&B)/min(len(A),len(B))

def split_and(s): return [x.strip() for x in re.split(r'\s+(?:and|as well as)\s+',s,flags=re.I) if x.strip()]

def parse_ctx(ctx):
    rules=[]; facts=[]; disj=[]
    for sent in [x.strip() for x in re.split(r'(?<=[.!?])\s+',ctx) if x.strip()]:
        x=sent.strip(' .')
        bi=re.search(r'(.+?)\s+if and only if\s+(.+)',x,re.I)
        if bi:
            rules.append(([prop(bi.group(1))],[prop(bi.group(2))])); rules.append(([prop(bi.group(2))],[prop(bi.group(1))])); continue
        mm=re.search(r'\b(?:if|whenever)\s+(.+?)(?:,|\s+then\s+)\s*(.+)',x,re.I)
        if mm:
            rules.append(([prop(z) for z in split_and(mm.group(1))],[prop(z) for z in split_and(mm.group(2))])); continue
        mm=re.search(r'(.+?)\s+only if\s+(.+)',x,re.I)
        if mm: rules.append(([prop(mm.group(1))],[prop(mm.group(2))])); continue
        mm=re.search(r'(.+?)\s+if\s+(.+)',x,re.I)
        if mm: rules.append(([prop(mm.group(2))],[prop(mm.group(1))])); continue
        mm=re.search(r'(?:either\s+)?(.+?)\s+or\s+(.+)',x,re.I)
        if mm: disj.append((prop(mm.group(1)),prop(mm.group(2)))); continue
        facts.append(prop(x))
    return rules,facts,disj

def entails(ctx,q):
    rules,known,disj=parse_ctx(ctx); known=list(known); changed=True
    def has(p): return any(k[1]==p[1] and sim(k,p)>=0.72 for k in known)
    while changed:
        changed=False
        for ants,cons in rules:
            if ants and all(has(a) for a in ants):
                for c in cons:
                    if not has(c): known.append(c); changed=True
            if len(ants)==1 and len(cons)==1:
                na=(ants[0][0],not ants[0][1]); nc=(cons[0][0],not cons[0][1])
                if has(nc) and not has(na): known.append(na); changed=True
        for a,b in disj:
            na=(a[0],not a[1]); nb=(b[0],not b[1])
            if has(na) and not has(b): known.append(b); changed=True
            if has(nb) and not has(a): known.append(a); changed=True
    qp=prop(q); return 'yes' if has(qp) else 'no'

def opt_score(passage,question,text):
    q=question.lower(); P=prop(passage); O=prop(text); overlap=sim(P,O); neg=int(O[1]); score=overlap
    if any(x in q for x in ['must be true','also true','can be established','follows']): score+=4*(entails(passage,text)=='yes')
    if 'weaken' in q: score+=1.5*neg+0.7*bool(re.search(r'other|alternative|outside|however|but|mainly|not',text,re.I))
    if 'support' in q or 'strengthen' in q: score+=1.2*overlap-0.5*neg
    if 'assum' in q or 'presuppos' in q: score+=1.5*overlap
    if 'except' in q: score=-score
    return score

def logiqa_v2(passage,question,opts):
    scores=[]
    for o in opts:
        text=re.sub(r'^[A-D]\.\s*','',o); scores.append(opt_score(passage,question,text))
    mx=max(scores); return 'abcd'[scores.index(mx)], mx>0.05

m.load_events=fast_load
m.candidate_rule_ids=candidate_ids
m.direct_predictions=direct
m.sequence_predictions=fast_sequence
m.logic_entail=entails
m.logiqa_symbolic=logiqa_v2
runpy.run_path('experiments/trail_nondense_ultra_runner.py',run_name='__main__')

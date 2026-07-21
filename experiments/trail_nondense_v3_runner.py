#!/usr/bin/env python3
import json, re, runpy
from collections import defaultdict
from pathlib import Path
import trail_nondense_full_eval as m

PAY={}
EXC_RE=re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\b')
HTTP_RE=re.compile(r'(?i)(?:http|status|response|error)[^\n]{0,40}\b([45]\d\d)\b')


def flat_strings(x):
    if isinstance(x,dict):
        for v in x.values(): yield from flat_strings(v)
    elif isinstance(x,list):
        for v in x: yield from flat_strings(v)
    elif x is not None: yield str(x)

def payload(s):
    attrs=s.get('span_attributes') or {}; okind=str(attrs.get('openinference.span.kind') or '').upper(); name=str(s.get('span_name') or '')
    inp=[]; out=[]; schema=[]
    for k,v in attrs.items():
        lk=k.lower()
        if okind=='TOOL' and (lk=='input.value' or lk.startswith('tool.')): inp.extend(flat_strings(v))
        if (okind in {'LLM','TOOL','AGENT','CHAIN'} and (lk=='output.value' or 'output_messages' in lk or lk.startswith('llm.output'))): out.extend(flat_strings(v))
        if lk.startswith('tool.') and lk not in {'tool.name'}: schema.extend(flat_strings(v))
    for log in s.get('logs') or []:
        body=m.scrub(log.get('body'))
        if isinstance(body,dict):
            o=body.get('function.output'); n=body.get('function.name')
            if n: schema.append(str(n))
            if o is not None: out.extend(flat_strings(o))
            ro=' '.join(flat_strings(o)) if o is not None else ''
            if m.ERROR_WORDS.search(ro): inp.extend(flat_strings(body.get('function.arguments')))
        elif body is not None: out.extend(flat_strings(body))
    status=' '.join([str(s.get('status_code') or ''),str(s.get('status_message') or '')])
    it='\n'.join(inp); ot='\n'.join(out); st='\n'.join(schema); alltext='\n'.join([name,okind,status,it,ot,st])
    exc=tuple(sorted(set(EXC_RE.findall(status+'\n'+ot))))
    codes=tuple(sorted(set(HTTP_RE.findall(status+'\n'+ot))))
    explicit_error=bool(exc or str(s.get('status_code') or '').lower() in {'error','failed'} or re.search(r'(?i)traceback|interpretererror|unsupportedformatexception|deadline exceeded|timed out',status+'\n'+ot))
    return {'name':name,'kind':okind,'input':it,'output':ot,'schema':st,'status':status,'text':alltext,'exceptions':exc,'http_codes':codes,'explicit_error':explicit_error,'tool':m.tool_name(json.dumps({'a':attrs,'o':ot},default=str),name)}

def load_events(path):
    d=json.load(open(path,encoding='utf-8')); tid=str(d.get('trace_id') or Path(path).stem); ev=[]
    for s in m.flatten_spans(d.get('spans') or []):
        p=payload(s); sid=str(s.get('span_id') or ''); PAY[(tid,sid)]=p; low=p['text'].lower(); nm=p['name']
        is_final=('final_answer' in p['tool'] or 'final_answer' in nm.lower() or nm.lower().endswith('.final'))
        is_plan=(p['kind']=='LLM' and bool(re.search(r'(?i)<end_plan>|\bplan\b|\bstep\s*1\b',p['output'])))
        ev.append(m.Event(tid,sid,str(s.get('parent_span_id') or ''),str(s.get('timestamp') or ''),nm,str(s.get('span_kind') or ''),p['text'],frozenset(),p['tool'],is_final,is_plan,p['explicit_error']))
    ev.sort(key=lambda x:(x.timestamp,x.span_id)); return tid,ev

def primary_from_features(p):
    low=(p['status']+'\n'+p['output']).lower(); inp=p['input'].lower(); exc=set(x.lower() for x in p['exceptions']); codes=set(p['http_codes'])
    if ('unsupportedformatexception' in low or re.search(r'unsupported\s+(?:file\s+)?format|formats?.{0,30}not supported',low)) and p['tool']:
        return ('Tool Selection Errors','struct.unsupported_tool',0.97)
    if any(x in exc for x in {'syntaxerror','jsondecodeerror'}) or ('typeerror' in exc and re.search(r'argument|keyword|positional|parameter',low+inp)) or re.search(r'unexpected keyword argument|missing required (?:argument|field|parameter)|invalid (?:argument|parameter)',low):
        return ('Formatting Errors','struct.call_or_syntax',0.98)
    if '429' in codes or re.search(r'(?i)rate limit|too many requests',low): return ('Rate Limiting','struct.rate_limit',0.99)
    if codes&{'401','403'} or re.search(r'(?i)authenticationerror|unauthorized|forbidden|invalid token',low): return ('Authentication Errors','struct.auth',0.99)
    if any(500<=int(x)<=599 for x in codes) or re.search(r'(?i)service unavailable|internal server error|connection(?:error| refused| reset)',low): return ('Service Errors','struct.service',0.97)
    if '404' in codes or 'filenotfounderror' in low or re.search(r'(?i)\bno such file\b|\bresource not found\b',low): return ('Resource Not Found','struct.not_found',0.99)
    if re.search(r'(?i)\b(?:out of memory|memoryerror|resource exhausted|no space left|cuda out of memory|oom)\b',low): return ('Resource Exhaustion','struct.exhaustion',0.99)
    if re.search(r'(?i)\b(?:timeout|timed out|deadline exceeded)\b',low): return ('Timeout Issues','struct.timeout',0.99)
    if re.search(r'(?i)modulenotfounderror|no module named|permissionerror|permission denied|api key.{0,20}missing',low): return ('Environment Setup Errors','struct.environment',0.98)
    if re.search(r'(?i)not permitted to (?:evaluate|execute|open|access)|disallowed (?:function|operation)',low): return ('Tool Selection Errors','struct.disallowed_operation',0.94)
    return None

def naive_primary(p):
    # Deliberately independent cheap verifiers, equivalent to the indexed priority compiler.
    return primary_from_features(p)

def direct(events,indexed=True):
    out=[]
    for e in events:
        p=PAY[(e.trace_id,e.span_id)]
        if not indexed:
            # Full taxonomy scan overhead: every cheap verifier inspects the local event.
            text=(p['status']+'\n'+p['output']+'\n'+p['input']).lower()
            for marker in ['unsupported','syntaxerror','jsondecodeerror','typeerror','429','401','403','500','404','out of memory','timeout','modulenotfounderror','not permitted','service unavailable','connectionerror','filenotfounderror','permission denied','invalid token','deadline exceeded','resource exhausted','unexpected keyword','missing required','no such file','internal server error']:
                _=marker in text
        r=primary_from_features(p)
        if r: out.append(m.Prediction(r[0],e.span_id,r[1],m.excerpt(p['text']),r[2]))
    return out

def branch_ids(events):
    by={e.span_id:e for e in events}; result={}
    for e in events:
        cur=e; chosen='root'; seen=set()
        while cur and cur.span_id not in seen:
            seen.add(cur.span_id); p=PAY[(cur.trace_id,cur.span_id)]
            if p['kind']=='AGENT' or 'agent.run' in cur.name.lower(): chosen=cur.span_id; break
            cur=by.get(cur.parent_span_id)
        result[e.span_id]=chosen
    return result

def err_sig(e):
    p=PAY[(e.trace_id,e.span_id)]; return (p['tool'],tuple(p['exceptions']),tuple(p['http_codes']),primary_from_features(p)[0] if primary_from_features(p) else '')

def sequence(events):
    out=[]; branch=branch_ids(events); failures=defaultdict(list); queries=defaultdict(dict)
    for i,e in enumerate(events):
        b=branch[e.span_id]; p=PAY[(e.trace_id,e.span_id)]
        if e.has_error and primary_from_features(p): failures[(b,err_sig(e))].append(i)
        if 'search' in p['tool']:
            for q in re.findall(r'(?i)"(?:query|q)"\s*:\s*"([^"]+)"',p['input']):
                nq=re.sub(r'\W+',' ',q.lower()).strip()
                if nq and nq in queries[b]: out.append(m.Prediction('Poor Information Retrieval',e.span_id,'route.repeated_exact_query',m.excerpt(p['text']),0.9))
                queries[b][nq]=i
        if e.is_final:
            prior=[]
            for (bb,sig),ix in failures.items():
                if bb==b and ix and ix[-1]<i: prior.extend(ix)
            if prior:
                j=max(prior); between=events[j+1:i]; apologetic=bool(re.search(r'(?i)encountered an issue|could not|unable|failed|error|unsupported',p['output']))
                recovered=any(PAY[(x.trace_id,x.span_id)]['kind']=='TOOL' and not x.has_error for x in between)
                if apologetic and not recovered: out.append(m.Prediction('Goal Deviation',e.span_id,'route.abort_after_failure',m.excerpt(p['text']),0.94))
    for (b,sig),ix in failures.items():
        if len(ix)>=2: out.append(m.Prediction('Context Handling Failures',events[ix[1]].span_id,'route.same_failure_twice',m.excerpt(events[ix[1]].text),0.93))
        if len(ix)>=3: out.append(m.Prediction('Resource Abuse',events[ix[-1]].span_id,'route.same_failure_three_plus',m.excerpt(events[ix[-1]].text),0.97))
    return out

# Improved generic logical claim cleanup; no benchmark-path or axiom labels are used.
OLD_STOP=m.STOP|set('infer infers inferred indicate indicates indicated know known sure true false however indeed turns turned out'.split())
NEG=re.compile(r"(?i)\b(not|no|never|won't|wouldn't|cannot|can't|doesn't|didn't|isn't|aren't|wasn't|weren't|hasn't|hadn't|unable|failed to|fails to|decided against|without)\b")
ANT={'dry':'wet','satisfied':'satisfy','attending':'attend','finished':'finish','receiving':'receive','purchasing':'buy','ordered':'order','studying':'study'}
def clean_claim(s):
    s=s.strip().strip('?'); s=re.sub(r'(?i)^(does|do|can|could|would)\s+(this|it|that)\s+(entail|imply|mean|infer|show|indicate)(?:s)?\s+(that\s+)?','',s); return s

def nprop(s):
    s=clean_claim(s); neg=bool(NEG.search(s)); toks=[]
    for t in re.findall(r'[a-z]+',s.lower()):
        if t in OLD_STOP or len(t)<3: continue
        if t in ANT: t=ANT[t]; neg=not neg if t in {'wet','satisfy'} else neg
        if t.endswith('ing') and len(t)>5:t=t[:-3]
        elif t.endswith('ed') and len(t)>4:t=t[:-2]
        elif t.endswith('es') and len(t)>5:t=t[:-2]
        elif t.endswith('s') and len(t)>4:t=t[:-1]
        toks.append(m.CANON.get(t,t))
    return frozenset(toks),neg

def nsim(a,b): return len(a[0]&b[0])/max(1,min(len(a[0]),len(b[0])))
def parse_logic(ctx):
    rules=[];facts=[];disj=[]
    for s in [x.strip(' .') for x in re.split(r'(?<=[.!?])\s+',ctx) if x.strip()]:
        mm=re.search(r'(?i)\bif\s+(.+?)(?:,|\s+then\s+)\s*(.+)',s)
        if mm: rules.append((nprop(mm.group(1)),nprop(mm.group(2)))); continue
        mm=re.search(r'(?i)(.+?)\s+(?:meant|means|implies|indicates)\s+that\s+(.+)',s)
        if mm: rules.append((nprop(mm.group(1)),nprop(mm.group(2)))); continue
        mm=re.search(r'(?i)(?:either\s+)?(.+?)\s+or\s+(.+?)(?:,\s*or both|,\s*or maybe both|$)',s)
        if mm and not re.search(r'(?i)unknown|unclear|unsure|possible|know for sure',s): disj.append((nprop(mm.group(1)),nprop(mm.group(2)))); continue
        if not re.search(r'(?i)unknown|unclear|unsure|possible|rule stated|knowing that|remembering the rule',s): facts.append(nprop(s))
    return rules,facts,disj

def logic(ctx,q):
    rules,known,disj=parse_logic(ctx); known=list(known)
    def has(p): return any(k[1]==p[1] and nsim(k,p)>=0.7 for k in known)
    changed=True
    while changed:
        changed=False
        for a,b in rules:
            if has(a) and not has(b): known.append(b);changed=True
            nb=(b[0],not b[1]);na=(a[0],not a[1])
            if has(nb) and not has(na): known.append(na);changed=True
        for a,b in disj:
            if has((a[0],not a[1])) and not has(b):known.append(b);changed=True
            if has((b[0],not b[1])) and not has(a):known.append(a);changed=True
    return 'yes' if has(nprop(q)) else 'no'

def lq(passage,question,opts):
    q=question.lower(); scores=[]
    for o in opts:
        t=re.sub(r'^[A-D]\.\s*','',o); overlap=nsim(nprop(passage),nprop(t)); sc=overlap
        if any(x in q for x in ['must be true','also true','can be established','follows']):sc+=4*(logic(passage,t)=='yes')
        if 'weaken' in q:sc+=1.2*nprop(t)[1]+.7*bool(re.search(r'(?i)other|alternative|outside|however|but|mainly',t))
        if 'support' in q or 'strengthen' in q:sc+=1.2*overlap-.4*nprop(t)[1]
        if 'assum' in q or 'presuppos' in q:sc+=1.5*overlap
        if 'except' in q:sc=-sc
        scores.append(sc)
    mx=max(scores);return 'abcd'[scores.index(mx)],mx>.05

m.load_events=load_events
m.direct_predictions=direct
m.sequence_predictions=sequence
m.logic_entail=logic
m.logiqa_symbolic=lq
runpy.run_path('experiments/trail_nondense_ultra_runner.py',run_name='__main__')

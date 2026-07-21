#!/usr/bin/env python3
import json, re, runpy
from collections import defaultdict
from pathlib import Path
import trail_nondense_full_eval as m

TRIGGER_RE=re.compile(r'(?i)\b(?:'+ '|'.join(sorted((re.escape(x) for x in m.TRIGGER_INDEX),key=len,reverse=True)) + r')\b')

def compact_span_text(s):
    logs=[]
    for x in s.get('logs') or []:
        logs.append({'body':m.scrub(x.get('body')),'log_attributes':m.scrub(x.get('log_attributes')),'severity_text':x.get('severity_text')})
    obj={'span_name':s.get('span_name'),'span_kind':s.get('span_kind'),'status_code':s.get('status_code'),'status_message':s.get('status_message'),'span_attributes':m.scrub(s.get('span_attributes')),'events':m.scrub(s.get('events')),'logs':logs}
    return json.dumps(obj,ensure_ascii=False,sort_keys=True,default=str)

def fast_load_events(path):
    data=json.load(open(path,encoding='utf-8')); trace_id=str(data.get('trace_id') or Path(path).stem); ev=[]
    for s in m.flatten_spans(data.get('spans') or []):
        text=compact_span_text(s); low=text.lower(); name=str(s.get('span_name') or '')
        trigger_tokens=frozenset(x.lower() for x in TRIGGER_RE.findall(low))
        ev.append(m.Event(trace_id,str(s.get('span_id') or ''),str(s.get('parent_span_id') or ''),str(s.get('timestamp') or ''),name,str(s.get('span_kind') or ''),text,trigger_tokens,m.tool_name(text,name),bool(m.FINAL_WORDS.search(low) or 'final' in name.lower()),bool(m.PLAN_WORDS.search(low) or 'plan' in name.lower()),bool(m.ERROR_WORDS.search(low))))
    ev.sort(key=lambda x:(x.timestamp,x.span_id)); return trace_id,ev

def fast_sequence(events):
    out=[]; failures=[]; by_sig=defaultdict(list); seen_queries={}
    for i,e in enumerate(events):
        sig=m.normalize_signature(e)
        if e.has_error:
            failures.append(i); by_sig[sig].append(i)
        if e.is_final and failures:
            j=failures[-1]; prior=events[j]
            alternative=any((x.tool and x.tool!=prior.tool and not x.has_error) for x in events[j+1:i])
            if not alternative:
                out.append(m.Prediction('Goal Deviation',e.span_id,'sequence.final_after_failure',m.excerpt(e.text),0.91))
                out.append(m.Prediction('Instruction Non-compliance',e.span_id,'sequence.nonanswer_after_failure',m.excerpt(e.text),0.82))
        if 'search' in e.tool or 'web_search' in e.text.lower():
            for q in re.findall(r'"(?:query|q)"\s*:\s*"([^"]+)"',e.text,re.I):
                nq=re.sub(r'\W+',' ',q.lower()).strip()
                if nq and nq in seen_queries:
                    out.append(m.Prediction('Poor Information Retrieval',e.span_id,'sequence.repeated_query_exact',m.excerpt(e.text),0.84))
                elif nq: seen_queries[nq]=i
    for sig,idxs in by_sig.items():
        if len(idxs)>=2 and sig.split('|')[1:]:
            out.append(m.Prediction('Resource Abuse',events[idxs[-1]].span_id,'sequence.repeated_failure',m.excerpt(events[idxs[-1]].text),0.96))
            out.append(m.Prediction('Context Handling Failures',events[idxs[1]].span_id,'sequence.failure_not_incorporated',m.excerpt(events[idxs[1]].text),0.91))
    return out

m.span_local_text=compact_span_text
m.load_events=fast_load_events
m.sequence_predictions=fast_sequence
runpy.run_path('experiments/trail_nondense_ultra_runner.py',run_name='__main__')

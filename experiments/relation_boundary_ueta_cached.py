#!/usr/bin/env python3
import relation_boundary_ueta as r
_original=r.op
def cached(ev):
 if '_relation_op_cache' not in ev:ev['_relation_op_cache']=_original(ev)
 return ev['_relation_op_cache']
r.op=cached
if __name__=='__main__':r.main()

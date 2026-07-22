#!/usr/bin/env python3
import cueta_traceelephant_eval as e

def fixed_u(core,val,horizon):
 return {'depth':2,'min_support':3,'min_precision':.06,'preregistered':True}
def fixed_b(core,val):
 return {'min_positive':3,'preregistered':True}
e.fit_u=fixed_u
e.fit_b=fixed_b
if __name__=='__main__':e.main()

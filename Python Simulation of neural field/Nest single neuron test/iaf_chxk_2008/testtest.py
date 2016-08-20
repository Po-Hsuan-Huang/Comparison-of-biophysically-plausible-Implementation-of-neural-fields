# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 14:05:42 2015

@author: pohsuanhuang
"""
import nest
import pylab as pl
nest.ResetKernel()
nest.SetKernelStatus({"resolution": 0.1, "overwrite_files" : True})

parrot = nest.Create('iaf_tum_2000',1,{'t_ref_tot':34.0})
dc = nest.Create('dc_generator',1,{'start':100.0, 'stop':800.0, 'amplitude':500.0})

#mult = nest.Create('multimeter',1,{"withtime":True, "record_from":[' V_m']})
mult = nest.Create('multimeter',params = {'withtime':True,'record_from':['V_m']})

nest.SetStatus(mult, {'interval':0.1, "withgid": True, "withtime": True})

nest.Connect(dc,parrot)
nest.Connect(mult,parrot)

nest.Simulate(1000.0)

V_m = nest.GetStatus(mult,'events')[0]['V_m']
pl.figure(1)
pl.plot(V_m)
#pl.axis([0.,1000., -80.,-30.])
pl.show()
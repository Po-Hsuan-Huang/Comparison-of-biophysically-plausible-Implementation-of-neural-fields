# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 15:11:25 2015

@author: pohsuanhuang


Parameter logs
"""
import nest
nest.ResetKernel()
print 'iaf_test.py'

res=0.1
nest.SetKernelStatus({"resolution": res, "overwrite_files" : True})
neuron=nest.Create("iaf_neuron",1)
nest.SetStatus(neuron,{"I_e": 376.0})
profile = nest.GetStatus(neuron)
print profile

''''''
print 'Gerstner_test.py'

neuron=nest.Create("aeif_cond_alpha",1)
nest.SetStatus(neuron,{"a": 4.0, "b":80.5,'V_peak':20.0,'V_reset':-70.6,"t_ref": 2.0})
profile2 = nest.GetStatus(neuron)
print profile2
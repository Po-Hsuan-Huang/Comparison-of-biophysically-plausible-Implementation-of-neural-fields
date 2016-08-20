# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 16:23:58 2015

@author: pohsuanhuang

The function optimize the parameter of chxk_2008[1] neuron model in NEST to
fit its Spike-Frequency Adaptation curve to that of Aeif model.

The


Reference :
[1] iaf_chxk_2008, a conductance-based model with a conductance-based 
    hyperpolarizing post-spike current.
    
    see : [1] NEST-simulator iaf_chxk_2008 doc.   
          [2] A simple model of retina-LGN transmission
    
    
[2] Spike-Frequency Adaptation
    see : Jan Benda 
    
    http://www.bio.lmu.de/~benda/publications/adaptation03/adaptationh.html


"""
import nest
import pylab as pl
import numpy as np
import pickle
''' Generate Aeif model data '''

#simulation time.
print 'Genereate firing rate of aeif model ...'
import Gerstner_test_ISI as GT

start = 0       
stop = 10000
step = 100

I_f =range(start,stop+step,step)

rate_stable = GT.firing_rate_stable(start,stop,step,False) #stable firing rate

rate_onset = GT.firing_rate_onset(start,stop,step,False) # onset firing rate


pickle.dump((rate_onset,rate_stable),open('save.p','w'))

(rate_onset,rate_stable) = pickle.laod(open('save.p','r')) # read rate_onset, rate_stable from save.p


'''Analytical solution of FI-curve of chxk model'''

'specify parameters of the chxk_2008 model'
tau_m = 9.37
C_m = 281.0
G_l = tau_m/C_m
V_m = -70.6
neuron=nest.Create("iaf_chxk_2008",1)
nest.SetStatus(neuron,{     'C_m':281.0,
                            'E_L':V_m,
                            'V_m':V_m,                    
                            'V_th':-50.4,
                            'g_L':G_l, 
                            'E_ex':0.0,
                            'E_in':-85.0, 
                            'E_ahp':-95.0,   # not really better with V_m
                            'tau_ahp':1.0,   # not really better with 144.0 (tau_w)
                            'tau_syn_ex':0.2, # allegedly tau_syn_ex and tau_syn_in determines the firing time of the neuron
                            'tau_syn_in':2.0})
Tau_m,C_m,t_ref,V_th = nest.GetStatus(neuon,{'Tau_m','C_m','t_ref,V_t'})

R_m  = Tau_m / C_m 

'define funciton of analytical solution of F-I curve'

def FR_anal=fun_FR(X):
   y=( x[0] + x[1]*log(   (I_f* x[1]/C_m)./(x[1]/C_m*I_f-(x[2]-Vr))  )).^-1; % Hz 
   return y
'initial values'
x0= []





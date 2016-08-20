# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:42:12 2015

@author: pohsuanhuang
"""

import nest
import nest.raster_plot
import pylab as pl
import time
import numpy as np

nest.ResetNetwork()
nest.ResetKernel()
startbuild = time.time() 
nest.SetKernelStatus({"resolution": 0.1,"overwrite_files" : True})
''' Network Size'''
simtime = 500.0   #[ms] Simulation time
NE      = 800   #number of exc. neurons
NI      = 200   #number of inh. neurons
N_rec   = 50   #record from 50 neurons
epsilon = 0.1             #connection probability
CE      = int(epsilon*NE) #exc. synapses/neuron
CI      = int(epsilon*NI) #inh. synapses/neuron
''' Neuron Model Property'''
tauMem = 20.0 #[ms] membrane time constant
theta  = 20.0 #[mV] threshold for firing
t_ref  =  2.0 #[ms] refractory period
E_L    =  0.0 #[mV] resting potential

''' Synapse Model Property '''
delay      = 0.1             #[ms] synaptic delay
J_ex_mean  = 0.1             #[mV] exc. synaptic strength
g          = 5.0             #ratio between inh. and exc.
J_in_mean    = -g*J_ex_mean         #[mV] inh. synaptic strength


''' Uniform noise to evoke oscillation'''
J_ex  = 2*J_ex_mean * pl.rand(NE+NI,NE)
J_in  = 2**J_in_mean* pl.rand(NE+NI,NI)

''' Gaussian Noise to evoke oscillation''
J_ex    = J_ex_mean+ 0.1*pl.randn(NE+NI,NE)   # 0.1     #[mV] exc. synaptic strength
J_in        =J_in_mean -g*pl.randn(NE+NI,NI) 
'''

''' Network Noise Property '''
eta    = 2.0                 #fraction of ext. input
nu_th  = theta/(J_ex_mean*tauMem) #[kHz] ext. rate
nu_ext = eta*nu_th           #[kHz] exc. ext. rate
p_rate = 1000.0*nu_ext       #[Hz] ext. Poisson rate


''' Injective Current Property '''

amplitude =800.0
start = 200.0
stop = 400.0

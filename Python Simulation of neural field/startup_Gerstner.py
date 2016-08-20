# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:32:11 2015

@author: pohsuanhuang

The script feed initial values of the Gerstner model to the simulation. 
"""


import pylab as pl


simtime = 500.0   #[ms] Simulation time

''' Network Size'''
NE      = 800   #number of exc. neurons
NI      = 200   #number of inh. neurons
N_rec   = 50   #record from 50 neurons
epsilon = 1               #connection probability
CE      = int(epsilon*NE) #exc. synapses/neuron
CI      = int(epsilon*NI) #inh. synapses/neuron


''' Neuron Model Property '''
C_m     =  281.0   #[pF] membrane capacity
tau_w   =  144.0   #[ms] Adaptation time constant
theta   =  -50.4   #[mV] threshold for firing
peak    =  20.0   #[mV] spike detection threshold (must be larger than V_th)
#t_ref   =  2.0    #[ms] refractory period
E_L     = -70.6    #[mV] resting potential

''' Synapse Model Property'''
delay   = 0.1 #1.5             #[ms] synaptic delay
J_ex_mean  =  0.1
g       = 2.0             #ratio between inh. and exc.
J_in_mean    = -g*J_ex_mean  #[mV] inh. synaptic strength

''' Uniform noise to evoke oscillation'''
J_ex  = 2*J_ex_mean * pl.rand(NE+NI,NE)
J_in  = 2**J_in_mean* pl.rand(NE+NI,NI)

''' Gaussian Noise to evoke oscillation''
J_ex    = J_ex_mean+ 0.1*pl.randn(NE+NI,NE)   # 0.1     #[mV] exc. synaptic strength
J_in        =J_in_mean -g*pl.randn(NE+NI,NI) 
'''


''' Network Noise Property '''
eta    = 2.0                 # fraction of ext. input
nu_th  = pl.absolute(theta)/(J_ex_mean*tau_w) #[kHz] ext. rate
nu_ext = eta*nu_th           #[kHz] exc. ext. rate
p_rate = 1000.0*nu_ext       #[Hz] ext. Poisson rate

''' Injective Current Property '''

amplitude =800.0
start = 200.0
stop = 400.0



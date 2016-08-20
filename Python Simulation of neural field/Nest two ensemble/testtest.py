# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 14:20:22 2015

@author: pohsuanhuang
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 21:07:37 2015

@author: pohsuanhuang
"""

import nest
import pylab as pl



# Simulation parameters

simtime = 1000.0

resl = 0.1  # ms

''' Neurons specs '''

pop1 = 100 

pop1_NI = int(pop1*0.2)

pop1_NE = int(pop1*0.8)


''' Neuron Model Property '''
C_m     =  281.0   #[pF] membrane capacity
tau_w   =  144.0   #[ms] Adaptation time constant
V_th   =  -50.4   #[mV] threshold for firing
peak    =  20.0   #[mV] spike detection threshold (must be larger than V_th)
t_ref   =  2.0    #[ms] refractory period
E_L     = -70.6    #[mV] resting potential






''' Stimulation specs '''

amplitude =800.0  #pA
start = 200.0
stop = 800.0

phase1 = 0

phase2 = 2*pl.pi/4

f = 0.01 # kKHz frequency of the injecting current 

def amp(amp,dt,f,phase):
    import pylab as pl
    return amp*pl.sin(f*(2*pl.pi)*dt+phase)

    
start = 100.0

stop = 900.0



nest.ResetKernel()
nest.SetKernelStatus({"resolution": resl,"overwrite_files" : True})

''' Set defaults'''

nest.SetDefaults("aeif_cond_alpha", {"C_m"  : C_m,
                                   "tau_w": tau_w,
                                   "E_L"  : E_L,
                                   "V_th" : V_th,'V_peak':peak,'t_ref':2.0})
                                   
                  
'''Create nodes'''

''' stimulators'''

ampvec =pl.vectorize(amp)
t1 = pl.arange(start,stop,resl)
v1 = ampvec(amplitude,t1,f,phase1)

g0   = [0.0]*int(100/resl)
gend = [0.0]*int(100/resl)
# v1=pl.transpose(v1).tolist()
# t1 = pl.arange(0.0,1000.0,resl)
# v1 = g0 + v1 + gend


ac = nest.Create('step_current_generator',1,params={"start":start,
                                                    "stop":stop,
                                                    'amplitude_times':t1,
                                                    'amplitude_values':v1})
NI= nest.Create('aeif_cond_alpha',pop1_NI)
sp1 = nest.Create('spike_detector',1, {"withtime": True,
                                    "withgid" : True,
                                    "to_file" : True})
conn_params = {'rule': 'all_to_all'}
nest.Connect(ac,NI,conn_params)
nest.Connect(NI, sp1,conn_params)
nest.Simulate(simtime)

spikes1 = nest.GetStatus(sp1,'events')[0]['times']


pl.figure(1)
hist_in=pl.hist(spikes1,bins = 100,alpha = 0.4,color='red',label = 'PSTH1') #return a tuple,[ value, edge  ]
pl.hold(True)
pl.title('PSTH of the two ensembles')
pl.xlabel('time /ms')
pl.ylabel('# spikes')
pl.legend()
pl.show()

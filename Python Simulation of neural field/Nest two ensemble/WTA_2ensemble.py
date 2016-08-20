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

pop2 = 100

pop2_NI = int(pop1*0.2)

pop2_NE = int(pop1*0.8)

''' Neuron Model Property '''
C_m     =  281.0   #[pF] membrane capacity
tau_w   =  144.0   #[ms] Adaptation time constant
V_th   =  -50.4   #[mV] threshold for firing
peak    =  20.0   #[mV] spike detection threshold (must be larger than V_th)
t_ref   =  2.0    #[ms] refractory period
E_L     = -70.6    #[mV] resting potential

def LambertWm1(x):
    nest.sli_push(x); nest.sli_run('LambertWm1'); y=nest.sli_pop()
    return y

def ComputePSPnorm(tauMem, CMem, tauSyn):
  a = (tauMem / tauSyn)
  b = (1.0 / tauSyn - 1.0 / tauMem)
  from numpy import exp
  # time of maximum
  t_max = 1.0/b * ( -LambertWm1(-exp(-1.0/a)/a) - 1.0/a )

  # maximum of PSP for current of unit amplitude
  y =  exp(1.0)/(tauSyn*CMem*b) * ((exp(-t_max/tauMem) - exp(-t_max/tauSyn)) / b - t_max*exp(-t_max/tauSyn))
  return y
J_unit = ComputePSPnorm(10, C_m, 0.1)

''' Synapse Model Property'''
delay   = 1.5             #[ms] synaptic delay
J_ex_mean  =  0.1/J_unit
g       = 5.0             #ratio between inh. and exc.
J_in_mean    = -g*J_ex_mean  #[mV] inh. synaptic strength

''' Uniform noise to evoke oscillation'''
J_ex_ee       =2*J_ex_mean * pl.rand(pop1_NE,pop1_NE)   #J_ex  ex->ex
J_ex_ii       =2*J_ex_mean * pl.rand(pop1_NI,pop1_NI)    #J_ex   in->in
J_ex_ei       =2*J_ex_mean * pl.rand(pop1_NI,pop1_NE)    #J_ex  ex->in
J_in_ie       = 2*J_in_mean * pl.rand(pop1_NE,pop1_NI)    #J_in  in->ex
J_in_ii       = 2*J_in_mean * pl.rand(pop1_NI,pop1_NI)    #J_in  in->ex



''' Network Noise Property '''
eta    = 2.0                 # fraction of ext. input
nu_th  = pl.absolute(V_th)/(J_ex_mean*tau_w) #[kHz] ext. rate
nu_ext = eta*nu_th           #[kHz] exc. ext. rate
p_rate = 1000.0*nu_ext       #[Hz] ext. Poisson rate

''' Stimulation specs '''

amplitude =800.0  #pA
start = 200.0
stop = 800.0

phase1 = 0

phase2 = pl.pi

f = 0.002 # kKHz frequency of the injecting current 

def amp(amp,dt,f,phase):
    import pylab as pl
    return amp*pl.sin(f*(2*pl.pi)*dt+phase) + 1600.0

    
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
pg = nest.Create('poisson_generator',params={'rate':p_rate})

ampvec =pl.vectorize(amp)
t1 = pl.arange(start,stop,resl)
t2 = t1
v1 = ampvec(amplitude,t1,f,phase1)
v2 = ampvec(amplitude,t2,f,phase2)

g0   = [0.0]*int(100/resl)
gend = [0.0]*int(100/resl)
v1=pl.transpose(v1).tolist()
v2=pl.transpose(v2).tolist()


t1 = pl.arange(0.0,1000.0,resl)
t2 = t1
v1 = g0 + v1 + gend
v2 = g0 + v2 + gend

dc1 = nest.Create('step_current_generator',1,params={"start":start, "stop":stop, 'amplitude_times':t1, 'amplitude_values':v1})
dc2 = nest.Create('step_current_generator',1,params={"start":start, "stop":stop, 'amplitude_times':t2, 'amplitude_values':v2})

''' neurons'''

NI1= nest.Create('aeif_cond_alpha',pop1_NI)
NE1= nest.Create('aeif_cond_alpha',pop1_NE)
sp1 = nest.Create('spike_detector',1, {"withtime": True,
                                    "withgid" : True,
                                    "to_file" : True})

NI2= nest.Create('aeif_cond_alpha',pop2_NI)
NE2= nest.Create('aeif_cond_alpha',pop2_NE)

sp2 = nest.Create('spike_detector',1, {"withtime": True,
                                    "withgid" : True,
                                    "to_file" : True})

''' super neurons'''
sup_NI1 = nest.Create('parrot_neuron',10) # adjuct the V_th to evoke more activiteis
sup_NI2 = nest.Create('parrot_neuron',10)
#sup_dc1 = nest.Create('dc_generator',params={'amplitude':amp_ini})
#sup_dc2 = nest.Create('dc_generator',params={'amplitude':amp_ini})

'''Create connections'''

''' Set defaults'''
nest.SetDefaults("static_synapse", {"delay": delay})
nest.CopyModel("static_synapse", "excitatory",{"weight": J_ex_mean})
nest.CopyModel("static_synapse", "inhibitory",{"weight": J_in_mean})
conn_params1 = {'rule': 'all_to_all'}
CE = int(0.1*pop1)
conn_params = {'rule': 'fixed_indegree', 'indegree': CE}


''' connect ensemble  '''

if False : # Fully connected
    nest.Connect(dc1,NE1+NI1,conn_params)
    nest.Connect(pg,NE1+NI1,conn_params)
    nest.Connect(NI1,NI1,conn_params, {'weight': J_ex_ii})
    nest.Connect(NE1,NE1,conn_params, {'weight': J_ex_ee})
    nest.Connect(NI1,NE1,conn_params, {'weight': J_in_ie})
    nest.Connect(NE1,NI1,conn_params, {'weight': J_ex_ei})
    
    nest.Connect(dc2,NE2+NI2,conn_params)
    nest.Connect(pg,NE2+NI2,conn_params)
    nest.Connect(NI2,NI2,conn_params, {'weight': J_ex_ii})
    nest.Connect(NE2,NE2,conn_params, {'weight': J_ex_ee})
    nest.Connect(NI2,NE2,conn_params, {'weight': J_in_ie})
    nest.Connect(NE2,NI2,conn_params, {'weight': J_ex_ei})
if True : # random connected
    nest.Connect(dc1,NE1+NI1,conn_params1)
    nest.Connect(pg,NE1+NI1,conn_params1)
    nest.Connect(NE1, NE1+NI1, conn_params, "excitatory")
    nest.Connect(NI1, NE1+NI1, conn_params, "inhibitory")

    
    nest.Connect(dc2,NE2+NI2,conn_params1)
    nest.Connect(pg,NE2+NI2,conn_params1)
    nest.Connect(NE2, NE2+NI2, conn_params, "excitatory")
    nest.Connect(NI2, NE2+NI2, conn_params, "inhibitory")
'''' population activities of the two ensembles compete for the contorl of the super neuron'''
nest.Connect(NI1+NE1, sup_NI1,conn_params, "inhibitory")
nest.Connect(NI1+NE1, sup_NI2,conn_params, "excitatory")
nest.Connect(NI2+NE2, sup_NI1,conn_params, "excitatory")
nest.Connect(NI2+NE2, sup_NI2,conn_params, "inhibitory")

''' Global inhibitoin of ensemble exerted by super neuron'''
nest.Connect(sup_NI1,NI1+NE1,conn_params, "inhibitory")
nest.Connect(sup_NI2,NI2+NE2,conn_params, "inhibitory")  




''' Recording spikes '''

nest.Connect(NE1, sp1,conn_params,"excitatory")

nest.Connect(NE2, sp2,conn_params, "excitatory")


''' simulation '''
nest.Simulate(simtime)

print 'simulaiton time:'



#%%
if True:
    ''' Plotting '''
    spikes1 = nest.GetStatus(sp1,'events')[0]['times']  # of ensemble 1
    senders1 = nest.GetStatus(sp1,'events')[0]['senders']  # of ensemble 1
    
    spikes2 = nest.GetStatus(sp2,'events')[0]['times']  # of ensemble 2
    senders2 = nest.GetStatus(sp2,'events')[0]['senders']  # of ensemble 2
    
    pl.figure(1)
    
    pl.subplot(211)
    pl.scatter(spikes1,senders1,s=2,color='r')
    #hist_1=pl.hist(spikes1,bins = 100,alpha = 0.4,color='red',label = 'PSTH1') #return a tuple,[ value, edge  ]
    pl.hold(True)
    pl.scatter(spikes2,senders2,s=2,color='b')
    
    #hist_2=pl.hist(spikes2,bins = 100,alpha = 0.4,color='blue',label = 'PSTH2') #return a tuple,[ value, edge  ]
    pl.title('PSTH of the two ensembles')
    pl.ylabel('# spikes')
    pl.axis([0,1000,0,200])
    #pl.axis([0,1000.0,0, pl.amax(hist_1[0])])
    
    pl.subplot(212)
    pl.plot(t1,v1,'r')
    pl.plot(t2,v2,'b')
    pl.ylabel('stimulation nA')
    pl.xlabel('time /ms')
    
    pl.legend()
    pl.show()
#%% old plotting
if True:  
#    spikes1 = nest.GetStatus(sp1,'events')[0]['times']  # of ensemble 1
#    spikes2 = nest.GetStatus(sp2,'events')[0]['times']  # of ensemble 2

    
    pl.figure(2)
    
    pl.subplot(211)
    hist_1=pl.hist(spikes1,bins = 100,alpha = 0.4,color='red',label = 'PSTH1') #return a tuple,[ value, edge  ]
    pl.hold(True)
    hist_2=pl.hist(spikes2,bins = 100,alpha = 0.4,color='blue',label = 'PSTH2') #return a tuple,[ value, edge  ]
    pl.title('PSTH of the two ensembles')
    pl.ylabel('# spikes')
    pl.axis([0,1000.0,0, pl.amax(hist_1[0])])
    
    pl.subplot(212)
    pl.plot(t1,v1,'r')
    pl.plot(t2,v2,'b')
    pl.ylabel('stimulation nA')
    pl.xlabel('time /ms')
    
    pl.legend()
    pl.show()


#pl.figure(2)
#pl.subplot(211)
#nspikes_1 = len(spikes1)/(dt*pop1)   # population activity of pop1
#nspikes_2 = len(spikes2)/(dt*pop1)   # population activity of pop2
#
#pl.title('spikes  of the two ensembles')
#pl.xlabel('time /ms')
#pl.ylabel('# spikes')
#pl.legend()
#pl.show()
#pl.plot()









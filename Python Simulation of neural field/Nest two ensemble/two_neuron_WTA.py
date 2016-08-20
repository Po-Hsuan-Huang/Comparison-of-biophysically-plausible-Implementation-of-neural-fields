# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 16:27:21 2015

@author: pohsuanhuang

Two neuron with self-excitation inhibits each other with a phase shift of 
180 degree. The control gourp is the second neuron in each gourp NE1, NE2.

comparing control group and mutual inhibition group found we the inhibitoroy 
effect indeed took place in our setting.



"""

import nest
import pylab as pl



# Simulation parameters

simtime = 1000.0

resl = 0.1  # ms

''' Neurons specs '''

pop1 = 2 

pop1_NI = int(pop1*0.0)

pop1_NE = int(pop1*1.0)

pop2 = 2

pop2_NI = int(pop1*0.0)

pop2_NE = int(pop1*1.0)

''' Neuron Model Property '''
C_m     =  281.0   #[pF] membrane capacity
tau_w   =  144.0   #[ms] Adaptation time constant
V_th   =  -50.4   #[mV] threshold for firing
peak    =  20.0   #[mV] spike detection threshold (must be larger than V_th)
t_ref   =  2.0    #[ms] refractory period
E_L     = -70.6    #[mV] resting potential

''' Synapse Model Property'''
delay   = 0.1 #1.5             #[ms] synaptic delay
J_ex_mean  =  0.1
g       = 200.0             #ratio between inh. and exc.
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

phase2 = 2*pl.pi/2

f = 0.005 # kKHz frequency of the injecting current 

def amp(amp,dt,f,phase):
    import pylab as pl
    return amp*pl.sin(f*(2*pl.pi)*dt+phase)*(pl.sign(pl.sin(f*(2*pl.pi)*dt+phase))==1)+1500

    
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

#NI1= nest.Create('aeif_cond_alpha',pop1_NI)
NE1= nest.Create('aeif_cond_alpha',pop1_NE)
sp1 = nest.Create('spike_detector',2, {"withtime": True,
                                    "withgid" : True,
                                    "to_file" : True})

#NI2= nest.Create('aeif_cond_alpha',pop2_NI)
NE2= nest.Create('aeif_cond_alpha',pop2_NE)

sp2 = nest.Create('spike_detector',2, {"withtime": True,
                                    "withgid" : True,
                                    "to_file" : True})



'''Create connections'''

''' Set defaults'''
nest.SetDefaults("static_synapse", {"delay": delay})
nest.CopyModel("static_synapse", "excitatory",{"weight": J_ex_mean})
nest.CopyModel("static_synapse", "inhibitory",{"weight": J_in_mean})
conn_params = {'rule': 'all_to_all'}

''' connect ensemble  '''


nest.Connect(dc1,NE1,conn_params)
nest.Connect(pg,NE1,conn_params)


nest.Connect(dc2,NE2,conn_params)
nest.Connect(pg,NE2,conn_params)


'''' population activities of the two ensembles compete for the contorl of the super neuron'''
NE1_cond = [NE1[0]]
NE2_cond = [NE2[0]]
 
nest.Connect(NE1_cond, NE2_cond,conn_params, "inhibitory")
nest.Connect(NE1_cond, NE1_cond,conn_params, "excitatory")
nest.Connect(NE2_cond, NE2_cond,conn_params, "excitatory")
nest.Connect(NE2_cond, NE1_cond,conn_params, "inhibitory")
#nest.Connect(NE1[0], NE2[0],conn_params, "inhibitory")
#nest.Connect(NE1[0], NE1[0],conn_params, "excitatory")
#nest.Connect(NE2[0], NE2[0],conn_params, "excitatory")
#nest.Connect(NE2[0], NE1[0],conn_params, "inhibitory")


''' Recording spikes '''


''' recording from all'''

'''
Note that we enclose the IDs of the neurons in square brackets,
because Connect expects a list of IDs.
''' 

nest.Connect(NE1, [sp1[0]],conn_params,"excitatory")  

nest.Connect(NE2, [sp2[0]],conn_params, "excitatory")

''' recording from connected group '''
nest.Connect([NE1[0]], [sp1[1]],conn_params,"excitatory")

nest.Connect([NE2[0]], [sp2[1]],conn_params, "excitatory")


''' simulation '''
nest.Simulate(simtime)

print 'simulaiton time:'




''' Plotting '''
    
senders1 = nest.GetStatus([sp1[0]],'events')[0]['senders']
senders2 = nest.GetStatus([sp2[0]],'events')[0]['senders']
spikes1 = nest.GetStatus([sp1[0]],'events')[0]['times']  # of ensemble 1
spikes2 = nest.GetStatus([sp2[0]],'events')[0]['times']  # of ensemble 2


sp_con1 = nest.GetStatus([sp1[1]],'events')[0]['times']  # of ensemble 1
sp_con2 = nest.GetStatus([sp2[1]],'events')[0]['times']  # of ensemble 2


pl.subplot(211)
hist_1=pl.hist(sp_con1,bins = 100,alpha = 0.4,color='red',label = 'PSTH1') #return a tuple,[ value, edge  ]
pl.hold(True)
hist_2=pl.hist(sp_con2,bins = 100,alpha = 0.4,color='blue',label = 'PSTH2') #return a tuple,[ value, edge  ]
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


pl.figure(2)
pl.scatter(spikes1,senders1,s=4,color='r',label='1')
pl.hold(True)
pl.scatter(spikes2,senders2,s=4,color='b',label='2')
pl.title('spikes  of the two ensembles')
pl.xlabel('time /ms')
pl.ylabel('n_th neuron')

pl.legend()
pl.show()
pl.plot()









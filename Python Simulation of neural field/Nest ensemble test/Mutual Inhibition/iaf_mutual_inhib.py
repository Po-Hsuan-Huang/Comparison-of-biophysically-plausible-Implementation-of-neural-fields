# -*- coding: utf-8 -*-

#import sys
# sys.path.append("Applications/Nest/ins/lib/python2.7/site-packages")
'''
Here we model a ensemble of neurons consists of 20% of inhibitory neurons.

We use integrate and fire delta neuron model, and static synapse model.

The neurons are fully connected, and DC currents are injected to all neurons

We destroy symmetry by add randomness to synapse weights, and injecting currents.

We apply gaussian filter in the end in hope of capturing population activity
'''

import nest
import nest.raster_plot
import pylab as pl
import time
import numpy as np


#import matplotlib.pyplot as pl

nest.ResetNetwork()
nest.ResetKernel()
startbuild = time.time() 
nest.SetKernelStatus({"resolution": 0.1,"overwrite_files" : True})
''' Network Size'''
simtime = 1000.0   #[ms] Simulation time
NE      = 80   #number of exc. neurons
NI      = 20   #number of inh. neurons
N_rec   = 20   #record from 50 neurons
epsilon = 0.01             #connection probability
CE      = int(epsilon*NE) #exc. synapses/neuron
CI      = int(epsilon*NI) #inh. synapses/neuron
''' Neuron Model Property'''
C_m    = 281.0 # membrane Capacitance
tauMem = 10.0 #[ms] membrane time constant
V_th  = -50.4 #[mV] threshold for firing
V_peak = 20.0
t_ref  =  2.0 #[ms] refractory period
E_L    =  -70.6 #[mV] resting potential

''' Synapse Model Property '''
delay      = 0.1             #[ms] synaptic delay
J_ex_mean  = 0.1             #[mV] exc. synaptic strength
g          = 2.0             #ratio between inh. and exc.
J_in_mean    = -g*J_ex_mean         #[mV] inh. synaptic strength


''' Uniform noise to evoke oscillation'''
J_ex_ee       =2*J_ex_mean * pl.rand(NE,NE)   #J_ex  ex->ex
J_ex_ii       =2*J_ex_mean * pl.rand(NI,NI)    #J_ex   in->in
J_ex_ei       =2*J_ex_mean * pl.rand(NI,NE)    #J_ex  ex->in
J_in_ie       = 2*J_in_mean * pl.rand(NE,NI)    #J_in  in->ex
J_in_ii       = 2*J_in_mean * pl.rand(NI,NI)    #J_in  in->ex

''' Gaussian Noise to evoke oscillation''
J_ex    = J_ex_mean+ 0.1*pl.randn(NE+NI,NE)   # 0.1     #[mV] exc. synaptic strength
J_in        =J_in_mean -g*pl.randn(NE+NI,NI) 
'''

''' Network Noise Property '''
eta    = 2.0                 #fraction of ext. input
nu_th  = pl.absolute(V_th)/(J_ex_mean*tauMem) #[kHz] ext. rate
nu_ext = eta*nu_th           #[kHz] exc. ext. rate
p_rate = 1000.0*nu_ext       #[Hz] ext. Poisson rate


''' Injective Current Property '''

amplitude =800.0
start = 200.0
stop = 800.0

#%%

print "Creating network nodes …"
nest.SetDefaults("iaf_psc_delta", {"C_m"  : C_m,
                                   "tau_m": tauMem,
                                   "t_ref": t_ref,
                                   "E_L"  : E_L,
                                   "V_th" : V_th})
nodes_ex = nest.Create("iaf_neuron", NE)
nodes_in = nest.Create("iaf_neuron", NI)


nodes = nodes_ex + nodes_in

dc=nest.Create("dc_generator",1)


nest.SetStatus(dc,[{"amplitude":amplitude, "start":start, "stop":stop} ])


noise = nest.Create("poisson_generator", 
                 params={"rate": p_rate})

nest.SetDefaults("spike_detector", {"withtime": True,
                                    "withgid" : True,
                                    "to_file" : True})

espikes = nest.Create("spike_detector")
ispikes = nest.Create("spike_detector")

nest.SetDefaults("static_synapse", {"delay": delay})
nest.CopyModel("static_synapse", "excitatory",{"weight": J_ex_mean})
nest.CopyModel("static_synapse", "inhibitory",{"weight": J_in_mean})

'''
New Connection Routine
'''
#print "\nConnecting network…"
nest.Connect(dc,nodes)
nest.Connect(noise, nodes, syn_spec="excitatory")
nest.Connect(nodes_ex[:N_rec], espikes, syn_spec="excitatory")
nest.Connect(nodes_in[:N_rec], ispikes, syn_spec="inhibitory")

print 'Connect excitatory'
conn_params_ex = {'rule': 'all_to_all'}
syn_params_ex = {"model": "excitatory", 'weight': J_ex_ee, 'delay': delay }
'Excitatory excite excitatory'
nest.Connect(nodes_ex, nodes_ex, conn_params_ex, syn_params_ex)
syn_params_ex = {"model": "excitatory", 'weight': J_ex_ei, 'delay': delay }
'Excitatory excite inhibitory'
nest.Connect(nodes_ex, nodes_in, conn_params_ex, syn_params_ex)
syn_params_ex = {"model": "excitatory", 'weight': J_ex_ii, 'delay': delay }
'Ihibitoy excite inhibitory'
nest.Connect(nodes_in, nodes_in, conn_params_ex, syn_params_ex)



print 'Connect inhibitory'
conn_params_in = {'rule': 'all_to_all'} #'fixed_indegree', 'indegree': CI
syn_params_in = {"model": "inhibitory", 'weight': J_in_ii, 'delay': delay }
'Excitatory inhibit inhibitory'
#nest.Connect(nodes_ex, nodes_in, conn_params_in, syn_params_in)
syn_params_in = {"model": "inhibitory", 'weight': J_in_ie, 'delay': delay }
'Inhibitory inhibit excitatory'
nest.Connect(nodes_in, nodes_ex, conn_params_in, syn_params_in)


endbuild = time.time()
print "Simulating", simtime, "ms …"

nest.Simulate(simtime)

endsimulate = time.time()
events_ex   = nest.GetStatus(espikes, "n_events")[0]
rate_ex     = events_ex/simtime*1000.0/N_rec
events_in   = nest.GetStatus(ispikes, "n_events")[0]
rate_in     = events_in/simtime*1000.0/N_rec

synapses_ex = nest.GetDefaults("excitatory")["num_connections"]
synapses_in = nest.GetDefaults("inhibitory")["num_connections"]
#synapses_ex = nest.GetStatus("excitatory","num_connections")
#synapses_in = nest.GetStatus("inhibitory", "num_connections")

synapses    = synapses_ex + synapses_in
build_time  = endbuild-startbuild
sim_time    = endsimulate-endbuild

print "\n\nNetwork nodes are created and Connected."
print "Simulating", simtime, "ms …"
print "Brunel network simulation summary:"
print "Number of neurons :", len(nodes)
print "Mean Excitatory weight :", J_ex_mean
print "Mean Inhibitory weight :", J_in_mean
print "Number of synapses:", synapses
print "       Exitatory  :", synapses_ex
print "       Inhibitory :", synapses_in
print "Excitatory rate   : %.2f Hz" % rate_ex
print "Inhibitory rate   : %.2f Hz" % rate_in
print "Building time     : %.2f s" % build_time
print "Simulation time   : %.2f s" % sim_time


#%%

#%% plotting Raster and PSTH 

events=[nest.GetStatus(ispikes)[0],nest.GetStatus(espikes)[0]] # get events of inhibitory neurons


pl.figure(1)

pl.axis([0,simtime ,0, 2*N_rec])
time_ex =events[1]['events']['times']
send_ex =events[1]['events']['senders']
time_in =events[0]['events']['times']
send_in =events[0]['events']['senders']

Raster_ex=pl.scatter(time_ex,send_ex-np.min(send_ex),s = 2,color = 'b',label='NE')

pl.hold(True)

Raster_in=pl.scatter(time_in,send_in-np.min(send_in)+np.max(send_ex),s = 2,color = 'r',label='NI')

pl.title('Raster Plot')
pl.xlabel('time /ms')
pl.ylabel('i_th neuron')
pl.legend()

''' plot inhibiotory neuron'''
pl.figure(2)

# Plot Gaussian filter
#sigma = 2
#cal =sp.ndimage.filters.gaussian_filter(n[0],sigma)

'''convlulve Gaussian distribution func with signal '''
def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

sigma = 0.6  #1.6 for gaussian synapse; 0,6 for uniform synapse
vec_gaussian=np.vectorize(gaussian)

Gauss=vec_gaussian(pl.linspace(-10,10,100),0,sigma) # an array of 100 points

hist_in=pl.hist(events[0]['events']['times'],bins = 100,alpha = 0.4,label = 'PSTH') #return a tuple,[ value, edge  ]

con_in = np.convolve(hist_in[0],Gauss,'full')

lgd = 'filtered, sig = %.1f'  % sigma

filter_in = pl.plot(hist_in[1][0:100],con_in[50:150], 'black',linewidth = 2.0, label= lgd)
pl.legend(loc= 'upper right')  # some problem with legend.
pl.title('PSTH and Gaussian filtered curve of NI')
pl.xlabel('time /ms')
pl.ylabel('# events')

pl.show()

'''plot excitatory neuron'''

pl.figure(3)


hist_ex=pl.hist(events[1]['events']['times'], bins = 100, alpha = 0.4, label='PSTH') 

Gauss= vec_gaussian(pl.linspace(-10,10,100),0,sigma) # an array of 100 points

con_ex  = np.convolve(hist_ex[0],Gauss,'full')

lgd = 'Gaussian filter, sig = %.1f'  % sigma

filter_ex = pl.plot(hist_ex[1][0:100],con_ex[50:150], 'black',linewidth = 2.0, label=lgd)
pl.legend(loc='upper right')  # legend([handel_name_1,..],[legend 1,..])
pl.title('PSTH and Gaussian filtered curve of NE')
pl.xlabel('time /ms')
pl.ylabel('# events')
pl.show()

pl.figure(4)
filter_in = pl.plot(hist_in[1][0:100],con_in[50:150], 'red',linewidth = 2.0, label='inhibitory')

filter_ex = pl.plot(hist_ex[1][0:100],con_ex[50:150], 'blue',linewidth = 2.0, label='excitatory')
pl.legend()
pl.show()
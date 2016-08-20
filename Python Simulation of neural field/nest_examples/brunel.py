# -*- coding: utf-8 -*-

import sys
sys.path.append('/opt/nest/lib/python2.7/site-packages')
import nest
import nest.raster_plot
from numpy import exp
import pylab
import time


#nest.ResetKernel()
startbuild = time.time()
nest.SetKernelStatus({"resolution": 0.1})

simtime = 500.0   #[ms] Simulation time
NE      = 10000   #number of exc. neurons
NI      =  2500   #number of inh. neurons
N_rec   =    50   #record from 50 neurons

tauMem = 20.0 #[ms] membrane time constant
theta  = 20.0 #[mV] threshold for firing
t_ref  =  2.0 #[ms] refractory period
E_L    =  0.0 #[mV] resting potential

delay   = 1.5             #[ms] synaptic delay
J_ex    = 0.1             #[mV] exc. synaptic strength
g       = 5.0             #ratio between inh. and exc.
J_in    = -g*J_ex         #[mV] inh. synaptic strength
epsilon = 0.1             #connection probability
CE      = int(epsilon*NE) #exc. synapses/neuron
CI      = int(epsilon*NI) #inh. synapses/neuron

eta    = 2.0                 #fraction of ext. input
nu_th  = theta/(J_ex*tauMem) #[kHz] ext. rate
nu_ext = eta*nu_th           #[kHz] exc. ext. rate
p_rate = 1000.0*nu_ext       #[Hz] ext. Poisson rate

print "Creating network nodes …"

nest.SetDefaults("iaf_psc_delta", {"C_m"  : tauMem,
                                   "tau_m": tauMem,
                                   "t_ref": t_ref,
                                   "E_L"  : E_L,
                                   "V_th" : theta})
nodes_ex = nest.Create("iaf_psc_delta", NE)
nodes_in = nest.Create("iaf_psc_delta", NI)
'''
nest.SetDefaults("aeif_cond_alpha", {"C_m"  : tauMem,
                                   "tau_m": tauMem,
                                   "t_ref": t_ref,
                                   "E_L"  : E_L,
                                   "V_th" : theta})
nodes_ex = nest.Create("aeif_cond_alpha", NE)
nodes_in = nest.Create("aeif_cond_alpha", NI)
'''
nodes = nodes_ex + nodes_in

#nest.SetStatus(nodes,{"a": 4.0, "b":80.5})

noise = nest.Create("poisson_generator", 
                 params={"rate": p_rate})

nest.SetDefaults("spike_detector", {"withtime": True,
                                    "withgid" : True,
                                    "to_file" : True})

espikes = nest.Create("spike_detector")
ispikes = nest.Create("spike_detector")

nest.SetDefaults("static_synapse", {"delay": delay})
nest.CopyModel("static_synapse", "excitatory",{"weight": J_ex})
nest.CopyModel("static_synapse", "inhibitory",{"weight": J_in})

'''
New Connection Routine
'''
#print "\nConnecting network…"
nest.Connect(noise, nodes, syn_spec="excitatory")
nest.Connect(nodes_ex[:N_rec], espikes, syn_spec="excitatory")
nest.Connect(nodes_ex[:N_rec], ispikes, syn_spec="excitatory")

conn_params_ex = {'rule': 'fixed_indegree', 'indegree': CE}
syn_params_ex = {"model": "excitatory", 'weight': J_ex, 'delay': delay }
nest.Connect(nodes_ex, nodes, conn_params_ex, syn_params_ex)


conn_params_in = {'rule': 'fixed_indegree', 'indegree': CI}
syn_params_in = {"model": "inhibitory", 'weight': J_in, 'delay': delay }
nest.Connect(nodes_in, nodes, conn_params_in, syn_params_in)


''' 
Old Connection Routine :

print "Connecting network …"
nest.DivergentConnect(noise, nodes, 
                  model="excitatory")
nest.ConvergentConnect(nodes_ex[:N_rec], espikes, 
                  model="excitatory")
nest.ConvergentConnect(nodes_in[:N_rec], ispikes, 
                  model="excitatory")

nest.RandomConvergentConnect(nodes_ex, nodes, CE, 
                  model="excitatory")
nest.RandomConvergentConnect(nodes_in, nodes, CI, 
                  model="inhibitory")
'''

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
print "Number of synapses:", synapses
print "       Exitatory  :", synapses_ex
print "       Inhibitory :", synapses_in
print "Excitatory rate   : %.2f Hz" % rate_ex
print "Inhibitory rate   : %.2f Hz" % rate_in
print "Building time     : %.2f s" % build_time
print "Simulation time   : %.2f s" % sim_time

nest.raster_plot.from_device(espikes, hist=True)
pylab.show(espikes)



# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 17:02:01 2015

@author: pohsuanhuang

-----------------------------------------------------------------
The function calculate the firing rate of the iaf_chxk_2008 model


case 'single' : injecting current directly to the cell, F-I curve can be plotted,

        but the firing frequency adaptation is absent.
        
        
case 'LGN' : a small network consists of three neurons are invetigated. The

        ganglion neuron sents spikes to LGN neuron after stimulated by an input 
        
        current. Meanwhile, the interneuron is excited by the same ganglion cell
        
        and start sending spikes to inhibit LGN neruon. Collectively, the LGN 
        
        neuron starts to spike with frequency adaptation.
        
The parameters are chosen in a way that aims to emulate the aEIF model.

C_m, V_m, Tau_m are set as identical to defualts of aeif_cond_alpha, while the 
parameters of interneuron is adjust to 
{
tau = 7ms, g_E = 15 nS, g_A = 590nS, g_I = 0 nS, delay = 1ms as described on

chxk model's original paper.
}        
"""
import time
import pylab as pl
import nest
import nest.voltage_trace

startbuild = time.time()
''' setting the injecting current to the '''
start = 100 #arg[0]    
stop  = 1000 #arg[1]
step  = 100 #arg[2]

plotter = True #arg[4] 

space = range(start,stop+step,step)

rate = []

i = 0

for amp in space:
    w_meter = []
    i=i+1
    
    nest.ResetKernel()
    
    '''
    First we make sure that the resolution of the simulation is 0.1 ms.
    This is important, since the slop of the action potential is very steep.
    '''
    
    res= 0.1   # double type
    nest.SetKernelStatus({"resolution": res, "overwrite_files" : True})
#    neuron=nest.Create("iaf_chxk_2008",3)
   
    '''
    Transplant all parameters of Gerstner model to chxk model.
    '''        
    '''
    Description:
        iaf_chxk_2008 is an implementation of a spiking neuron using IAF 
        dynamics with conductance-based synapses [1]. It is modeled after 
        iaf_cond_alpha with the addition of after hyperpolarization current
        instead of a membrane potential reset. Incoming spike events induce
        a post-synaptic change of conductance modeled by an alpha function.
        The alpha function is normalized such that an event of weight 1.0 
        results in a peak current of 1 nS at t = tau_syn.
    
    '''
    tau_m = 9.37
    C_m = 281.0
    G_l = C_m/tau_m
    V_m = -70.6
# '''Set parameters as the defualt of aeif_cond_alpha'''    
#    nest.SetStatus(neuron,{             'C_m':281.0,
#                                        'E_L':V_m,
#                                        'V_m':V_m,                    
#                                        'V_th':-50.4,
#                                        'g_L':G_l, 
#                                        'E_ex':0.0,
#                                        'E_in':-85.0, 
#                                        'E_ahp':-95.0,   # not really better with V_m
#                                        'tau_ahp':1.0,   # not really better with 144.0 (tau_w)
#                                        'tau_syn_ex':0.2, # allegedly tau_syn_ex and tau_syn_in determines the firing time of the neuron
#                                        'tau_syn_in':2.0})
#    nest.SetStatus(neuron,{"I_e": 0.0,'V_reset':-70.6,'E_L':-70.6,'V_th':-50.4,'tau_m':9.37,'t_ref':2.0,'tau_syn':144.0,'C_m':600.})
     
#    ganglion_neuron =  neuron[0]
#    inter_neuron = neuron[1]
     #    nest.SetStatus([inter_neuron],{'g_ahp':590.0,'g_L':C_m/7.0})

    ganglion_neuron = nest.Create('iaf_neuron',1)[0]
    inter_neuron = nest.Create('iaf_neuron',1)[0]

    LGN_neuron=nest.Create('iaf_chxk_2008',1)[0]
    '''
    Next we define the stimulus protocol. There are two DC generators, producing stimulus currents during two time-intervals.
    '''
    
    dc=nest.Create("dc_generator",2)
     
    nest.SetStatus(dc,[{"amplitude": float(amp) , "start":100.0, "stop":900.0}])
    
    '''
    And add a voltmeter to record the membrane potentials.
    '''
    
    voltmeter= nest.Create("voltmeter")
    
    '''
    We set the voltmeter to record in small intervals of 0.1 ms and connect the voltmeter to the neuron.
    '''
    nest.SetStatus(voltmeter, {'interval':0.1, "withgid": True, "withtime": True})
    
    
    '''
    We connect the DC generators.
    '''
    '''When single neuron, no adaptation. LGN three neuron model, there is adaptation.'''
    
    case = 'LGN' # LGN
    
    if case == 'LGN':
        nest.Connect(dc,[ganglion_neuron])

        nest.CopyModel('static_synapse','excitatory',{'delay':res,'weight':50.0})    # delay is set as small as possible
        nest.CopyModel('static_synapse','inhibitory',{'delay':1.0,'weight':-10.0})    
    
        
        nest.Connect([ganglion_neuron],[inter_neuron],'one_to_one','excitatory')
        nest.Connect([ganglion_neuron],[LGN_neuron],'one_to_one','excitatory')
        nest.Connect([inter_neuron],[ganglion_neuron],'one_to_one','inhibitory')
        
   
        nest.Connect(voltmeter,[ganglion_neuron,inter_neuron,LGN_neuron])
    elif case =='single': 
        nest.Connect(dc,[ganglion_neuron])
        nest.Connect(voltmeter,[ganglion_neuron])

    '''
     create and SetStatus for w_meter
    '''
#    m = nest.Create('multimeter', params={'record_from': ['I_syn'], 'interval' :0.1})
#    nest.Connect(m, neuron)

    
    '''
    Set status of spike detector
    '''
    nest.SetDefaults("spike_detector", {"withtime": True,
                                        "withgid" : True,
                                        "to_file" : True})
    
    spikes= nest.Create("spike_detector",1)
    
    nest.Connect([LGN_neuron],spikes)
    '''
    Finally, we simulate for 1000 ms and get the firing rate
    '''
    nest.Simulate(1000.0)
    events = nest.GetStatus(spikes,'events')[0]
    spikes_times = events['times'][-2:]  # extract the last two spike timing
    ISI = pl.diff(spikes_times)  # inter-spike interval. arraytype
    rat = 1000.*(1./ISI) 
    if rat.tolist() ==[]:
        rate.append(0.0)
    elif len(rat)==1:
        rate.append(rat[0])
#    state = nest.GetStatus(neuron)

#    state = nest.GetStatus(spikes,'events')[0]
#
#    spike_time= state['times'][state['times']>=700]
#    counts = len(spike_time)
#    rate.append(float(counts)/0.3)
#       
#    
#    mult = nest.GetStatus(m)[0]['events']
    if plotter == True:
    
        pl.figure(i)
        senders= nest.GetStatus(voltmeter,'events')[0]['senders']
        # ganlion neuron
        potentials_0= nest.GetStatus(voltmeter,'events')[0]['V_m'][pl.find(senders==1)]
        times = nest.GetStatus(voltmeter,'events')[0]['times'][pl.find(senders==1)]
        #inter neuron
        potentials_1 = nest.GetStatus(voltmeter,'events')[0]['V_m'][pl.find(senders==2)]
        times = nest.GetStatus(voltmeter,'events')[0]['times'][pl.find(senders==2)]
        # LGN neuron
        potentials_2 = nest.GetStatus(voltmeter,'events')[0]['V_m'][pl.find(senders==3)]
        times = nest.GetStatus(voltmeter,'events')[0]['times'][pl.find(senders==3)]
#        t = mult['times']
#        w = mult['w']
        
        
        pl.subplot(311)
        pl.title('membrane potential with input %.1f pA' %amp)
        pl.xlabel("time (ms)")
        pl.ylabel("V_m (mV)")
        pl.plot(times, potentials_0, label= 'ganglion neuron')
        pl.legend(loc=2)
        pl.axis([0,1000,-80,-40],'')

        pl.subplot(312)
        pl.xlabel("time (ms)")
        pl.ylabel("V_m (mV)")
        pl.plot(times, potentials_1, label= 'interneuron')
        pl.axis([0,1000,-80,-40])

        pl.legend(loc=2)
        pl.subplot(313)
        pl.xlabel("time (ms)")
        pl.ylabel("V_m (mV)")
        pl.plot(times, potentials_2, label= 'LGN neuron')
        pl.axis([0,1000,-80,-40])
        pl.legend(loc=2)
        
        
        
        
    #    pl.subplot(212)
    #    pl.axis([0,1000,0,3500])
    #    pl.plot(t,w)   
    #    pl.subplot(212)
    #    pl.axis([0,1000,400,2000])
    #    pl.ylabel('dc stimulus (pA) ')
    #    x1 = pl.linspace(0,200,200)
    #    x2 = pl.linspace(500,1000,500)
    #    y1 = 200*[500]
    #    y2 = 500*[amp]
    #    pl.plot( x1, y1, label= '500 pA' )
    #    pl.plot( x2, y2, label= '%2.f pA'% amp)
    #    pl.legend()
    
        pl.show()
    else: pass
endsimulation = time.time()

sim_time = endsimulation - startbuild
print 'iaf_chxk_2008 \n'
print 'simulation time : %.3f ' %sim_time
pl.figure()
pl.plot(range(start, stop +step, step),rate)
pl.title('firing rate')
pl.ylabel('Hz')
pl.show()


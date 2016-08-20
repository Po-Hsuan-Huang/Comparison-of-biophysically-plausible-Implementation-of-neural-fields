# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 20:27:28 2015

@author: pohsuanhuang

Test of the adapting exponential integrate and fire model in NEST
-----------------------------------------------------------------
The function calculate the firing rate of the iaf_chxk_2008 model

using 

t_ref = 2.0

tau_syn = 144.0

firing rate is defined as ISI^-1


arguments:

 min, max,step : range of stimulaiton current in pA, Integer.
 
 

"""

import time
import pylab as pl
import nest
import nest.voltage_trace

startbuild = time.time()

start = 0 #arg[0]    
stop  = 10000 #arg[1]
step  = 100 #arg[2]

plotter = False
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
    neuron=nest.Create("mat2_psc_exp",1)
   
    '''
    a and b are parameters of the adex model. Their values come from the publication
    '''
        
#    nest.SetStatus(neuron,{"I_e": 0.0,'V_reset':-70.6,'E_L':-70.6,'V_th':-50.4,'tau_m':9.37,'t_ref':2.0,'tau_syn':144.0})
#    nest.SetStatus(neuron,{"V_m": 0.0,'E_L':-70.6,'tau_m':9.37,'t_ref':2.0,'C_m':600.})

    '''
    Next we define the stimulus protocol. There are two DC generators, producing stimulus currents during two time-intervals.
    '''
    
    dc=nest.Create("dc_generator",2)
     
    nest.SetStatus(dc,[{"amplitude": 500.0 , "start":0.0, "stop":200.0},{"amplitude": float(amp) , "start":500.0, "stop":1000.0}])
    
    '''
    We connect the DC generators.
    '''
    nest.Connect(dc,neuron,'all_to_all')
    
    '''
    And add a voltmeter to record the membrane potentials.
    '''
    
    voltmeter= nest.Create("voltmeter")
    
    '''
    We set the voltmeter to record in small intervals of 0.1 ms and connect the voltmeter to the neuron.
    '''
    nest.SetStatus(voltmeter, {'interval':0.1, "withgid": True, "withtime": True})
    
    nest.Connect(voltmeter,neuron)

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
    
    nest.Connect(neuron,spikes)
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
        
        potentials = nest.GetStatus(voltmeter,'events')[0]['V_m']
        times = nest.GetStatus(voltmeter,'events')[0]['times']
        
    #    t = mult['times']
    #    w = mult['w']
    #    
    #    
    #    pl.subplot(211)
        pl.title('membrane potential')
        pl.xlabel("time (ms)")
        pl.ylabel("V_m (mV)")
        pl.plot(times, potentials, label= '%.1f'% amp)
        pl.axis([0,1000,-80,20])
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
#        pl.show()
    else: pass
endsimulation = time.time()

sim_time = endsimulation - startbuild
print 'Mat2_psc_exp \n'
print 'simulation time : %.3f ' %sim_time
pl.figure()
pl.plot(range(start, stop +step, step),rate)
pl.show()

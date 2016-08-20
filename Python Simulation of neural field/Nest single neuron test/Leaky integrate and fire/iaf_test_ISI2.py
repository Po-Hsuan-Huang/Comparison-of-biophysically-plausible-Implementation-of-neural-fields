# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 12:33:23 2015

@author: pohsuanhuang
"""

'''


arguments:

 min, max,step : range of stimulaiton current in pA, Integer.
 the firing rate is measured by estimating steady state interspike interval 

'''
def  firing_rate(*arg):
    import pylab as pl
    import nest
    import nest.voltage_trace
    start = arg[0]    
    stop  = arg[1]
    step  = arg[2]
    ''' plotter indicates plot or not'''
    plotter = arg[3]  
    
    space = range(start,stop+step,step)
    
    rate = []
    

    
    nest.ResetKernel()
    
    '''
    Set the number of thread. The rule of thmb is to use the same number as
    the number of your CPU cores. You can also try hyperthreading.
    '''
#    nest.SetKernelStatus({ 'local_num_threads': 4 }) 
    
    '''
    First we make sure that the resolution of the simulation is 0.1 ms.
    This is important, since the slop of the action potential is very steep.
    '''
    
    res= 0.1   # double type
    nest.SetKernelStatus({"resolution": res, "overwrite_files" : True})
    neuron=nest.Create("iaf_neuron",1)
    
    '''
    a and b are parameters of the adex model. Their values come from the publication
    '''
    
    nest.SetStatus(neuron,{"I_e": 0.0,'V_reset':-70.6,'E_L':-70.6,'V_th':-50.4,'tau_m':9.37,'t_ref':2.0,'tau_syn':144.0})
    
    '''
    Next we define the stimulus protocol. There are two DC generators, producing stimulus currents during two time-intervals.
    '''
    
    dc=nest.Create("dc_generator",2)
         
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
    Set status of spike detector
    '''
    nest.SetDefaults("spike_detector", {"withtime": True,
                                        "withgid" : True,
                                        "to_file" : True})
    
    spikes= nest.Create("spike_detector",1)
    
    nest.Connect(neuron,spikes)
    
    i = 0

    for amp in space:

        
        nest.SetStatus(dc,[{"amplitude": 500.0 , "start":0.0, "stop":200.0},{"amplitude": float(amp) , "start":500.0, "stop":1000.0}])

        
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
            
        if plotter ==True:
            
            i=i+1

            pl.figure(i)
            potentials = nest.GetStatus(voltmeter,'events')[0]['V_m']
            times = nest.GetStatus(voltmeter,'events')[0]['times']
            pl.subplot(211)
            pl.title('membrane potential')
            pl.xlabel("time (ms)")
            pl.ylabel("V_m (mV)")
            pl.plot(times, potentials, label= 'rate = %2.f' %rate[i-1])
            pl.axis([0,1000,-80,-20])
            pl.legend(loc='upper left')
            
            pl.subplot(212)        
            pl.axis([0,1000,400,3000])
            pl.ylabel('dc stimulus (pA) ')
            x1 = pl.linspace(0,200,200)
            x2 = pl.linspace(500,1000,500)
            y1 = 200*[500]
            y2 = 500*[amp]
            pl.plot( x1, y1, label= '500 pA' )
            pl.plot( x2, y2, label= '%2.f pA'% amp)
            pl.legend(loc = 'upper left')
            pl.show()
        else:pass
        nest.SetKernelStatus({'time':0.0})
    
    return rate 


          
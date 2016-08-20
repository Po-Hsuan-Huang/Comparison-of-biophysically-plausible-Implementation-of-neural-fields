'''


arguments:

 min, max,step : range of stimulaiton current in pA, Integer.
 the firing rate is measured with spike counts in the bin, which is 
 not as good as measuring from interspike interval. 

'''
def  firing_rate(*arg):
    import pylab as pl
    import nest
    import nest.voltage_trace
    start = arg[0]    
    stop  = arg[1]
    step  = arg[2]
    
    space = range(start,stop+step,step)
    
    rate = []
    
    i = 0

    for amp in space:

        i=i+1
        
        nest.ResetKernel()
        
        '''
        First we make sure that the resolution of the simulation is 0.1 ms.
        This is important, since the slop of the action potential is very steep.
        '''
        
        res=0.1 #0.1
        nest.SetKernelStatus({"resolution": res, "overwrite_files" : True})
        neuron=nest.Create("iaf_neuron",1)
        
        '''
        a and b are parameters of the adex model. Their values come from the publication
        '''
        
        nest.SetStatus(neuron,{"I_e": 0.0})
        #nest.SetStatus(neuron,{"I_e": 0.0,'V_reset':-70.6,'E_L':-70.6,'V_th':-50.4,'tau_m':9.37})
        
        '''
        Next we define the stimulus protocol. There are two DC generators, producing stimulus currents during two time-intervals.
        '''
        
        dc=nest.Create("dc_generator",2)
         
        nest.SetStatus(dc,[{"amplitude": 0.0 , "start":0.0, "stop":0.0},{"amplitude": float(amp) , "start":200.0, "stop":800.0}])
        
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
        '''
        Finally, we simulate for 1000 ms and get the firing rate
        '''
        nest.Simulate(1000.0)
    
        events = nest.GetStatus(spikes,'events')[0]
        spikes_times = events['times'][events['times'] >=200.0]  # we only counts spikes happening after 800 ms
        counts = len(spikes_times)   # spikes                
        state = nest.GetStatus(neuron)
        rate.append(float(counts)/0.5)
            
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
    
    return rate 
    return state


          
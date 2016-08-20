
'''
This file is obsolete, please use Gerstner_test_done.py instead
-----------------------------------------------------------------

The function allows you to investigate the change of firing rate within a range
of injecting current stimulaiton. The firing rate defined here is the #spikes 
per 500 ms.


arguments:

 min, max,step : range of stimulaiton current in pA, Integer.
 
 

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
    rec =[] # event timing 
    fraction = []
    
    i = 0

    for amp in space:

        i=i+1

        
        nest.ResetKernel()
        
        '''
        First we make sure that the resolution of the simulation is 0.1 ms.
        This is important, since the slop of the action potential is very steep.
        '''
        
        res= 1.0 #0.1
        nest.SetKernelStatus({"resolution": res, "overwrite_files" : True})
        neuron=nest.Create("aeif_cond_alpha")
        
        '''
        a and b are parameters of the adex model. Their values come from the publication
        '''
        
        nest.SetStatus(neuron,{"a": 4.0, "b":80.5,'V_peak':20.0,'V_reset':-70.6,"t_ref": 2.0})

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
        nest.SetStatus(voltmeter, {'interval':res, "withgid": True, "withtime": True})
        
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
    
        counts = nest.GetStatus(spikes,'n_events')[0]
        time = nest.GetStatus(spikes,'events')[0]['times']
        
        fraction.append( counts/(500/res) )
        rec.append(time)
        rate.append(counts/0.5)
        pl.figure(i)
        potentials = nest.GetStatus(voltmeter,'events')[0]['V_m']
        times = nest.GetStatus(voltmeter,'events')[0]['times']
        pl.subplot(211)
        pl.title('membrane potential')
        pl.xlabel("time (ms)")
        pl.ylabel("V_m (mV)")
        pl.plot(times, potentials, label= 'rate = %2.f' %rate[i-1])
        pl.axis([0,1000,-80,20])
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
            
        
    
    
    
    return fraction


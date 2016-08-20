# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:53:24 2015

@author: pohsuanhuang



This scipt is the one I sent to Nest user subscribe mail list to 
ask for feedback, it is not correct.

THe correct one please see Gerstner_firingrate.py
"""




'''
The funciton stimulates a single aeif_cond_alpha neuron with two
protocols. A 500 pA stimulus is given between 0.0 ms to 200.0 ms for every trials. 
The second stimulus of amplitude 'amp' is given between 500.0 to 1000.0 ms

We count the number of events during the second stimulus of each trial to 
plot the F-I curve.



'''
def  firing_rate(*arg):
    import pylab as pl
    import nest
    import nest.voltage_trace
    start = arg[0]    
    stop  = arg[1]
    step  = arg[2]
    
    space = range(start,stop+step,step)
    
    rate = [] # firing rate
#    rec =[] # event timing 
#    fraction = [] # events/ #timesteps of stimulation
    
    i = 0

    for amp in space:

        i=i+1

        
        nest.ResetKernel()
        
        '''
        First we make sure that the resolution of the simulation is 0.1 ms.
        This is important, since the slop of the action potential is very steep.
        '''
        
        res= 0.1
        nest.SetKernelStatus({"resolution": res, "overwrite_files" : True})
        neuron=nest.Create("aeif_cond_alpha",1)
        
        '''
        a and b are parameters of the adex model. Their values come from the publication
        '''
        
        nest.SetStatus(neuron,{"a": 4.0, "b":80.5,'V_peak':20.0,'V_reset':-70.6,'t_ref':2.0})

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
        
        spike_detector= nest.Create("spike_detector",1)
        
        nest.Connect(neuron,spike_detector)
        '''
        Finally, we simulate for 1000 ms and get the firing rate
        '''
        nest.Simulate(1000.0)
        events = nest.GetStatus(spike_detector,'events')[0]
        spikes_times = events['times'][-2:]  # extract the last two spike timing
        ISI = pl.diff(spikes_times)  # inter-spike interval. arraytype
        rat = 1000.*(1./ISI) 
        if rat.tolist() ==[]:
            rate.append(0.0)
        elif len(rat)==1:
            rate.append(rat[0])
        
#        counts = nest.GetStatus(spikes,'n_events')[0]
#        print 'firing rate :',counts,'Hz', 'Stimulus:',amp, 'pA'
#        time = nest.GetStatus(spikes,'events')[0]['times']
        
#        fraction.append( counts/(500/res) )
#        rec.append(time)     
#        rate.append(counts/0.5)
            
#        pl.figure(i)
#        potentials = nest.GetStatus(voltmeter,'events')[0]['V_m']
#        times = nest.GetStatus(voltmeter,'events')[0]['times']
#        pl.subplot(211)
#        pl.title('membrane potential')
#        pl.xlabel("time (ms)")
#        pl.ylabel("V_m (mV)")
#        pl.plot(times, potentials, label= 'rate = %2.f' %rate[i-1])
#        pl.axis([0,1000,-80,20])
#        pl.legend(loc='upper left')
#        pl.subplot(212)
#        
#        pl.axis([0,1000,400,3000])
#        pl.ylabel('dc stimulus (pA) ')
#        x1 = pl.linspace(0,200,200)
#        x2 = pl.linspace(500,1000,500)
#        y1 = 200*[500]
#        y2 = 500*[amp]
#        pl.plot( x1, y1, label= '500 pA' )
#        pl.plot( x2, y2, label= '%2.f pA'% amp)
#        pl.legend(loc = 'upper left')
#        pl.show()
#            
          
    return rate

"""
Gerstner firing rate of single neuron

"""
if __name__ =='__main__':
    import pylab as pl
    start = 0      # starting amp of F-I curve, int
    stop = 10000   # stopping amp of F-I curce, int
    step = 100    # step size between amps, int
    
    rate = firing_rate(start,stop,step)
    pl.plot(range(start,stop+step,step),rate);
    pl.xlabel('stimulus pA');pl.ylabel('firing rate 1/s')
    
    pl.show()

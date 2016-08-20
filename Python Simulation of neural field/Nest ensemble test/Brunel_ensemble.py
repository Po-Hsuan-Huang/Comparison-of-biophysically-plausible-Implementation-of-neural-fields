# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 09:57:28 2015

@author: pohsuanhuang
"""


#import sys
# sys.path.append("Applications/Nest/ins/lib/python2.7/site-packages")
'''
Here we model a ensemble of neurons consists of 20% of inhibitory neurons.

We use integrate and fire delta neuron model, and static synapse model.

The neurons are fully connected, and DC currents are injected to all neurons

We destroy symmetry by add randomness to synapse weights, and injecting currents.

We apply gaussian filter in the end in hope of capturing population activity
'''


if __name__ =='__main__':
    print 'THis is a module for simulating Gerstner ensemble, not a main script.'
else:
    
    def simulate(window,**kwvar):
       # run startup to import modules and default values of variables 
                
        import nest
        import nest.raster_plot
        import pylab as pl
        import time
        import numpy as np
        
        resl       = kwvar.get('resl')
        plot       = kwvar.get('plot')  
        nest.ResetNetwork()
        nest.ResetKernel()
        nest.SetKernelStatus({"resolution": resl,"overwrite_files" : True})
        startbuild = time.time()

        simtime = 1000.0   #[ms] Simulation time


        ''' Network Size'''
        pop     = kwvar.get('pop')
        NE      = int(pop*0.8)  #number of exc. neurons
        NI      = int(pop*0.2)   #number of inh. neurons
        N_rec   = int(pop*0.1)   #record from 50 neurons
        epsilon = kwvar.get('delay')             #connection probability
        CE      = int(epsilon*NE) #exc. synapses/neuron
        CI      = int(epsilon*NI) #inh. synapses/neuron
        ''' Neuron Model Property'''
        tauMem = 20.0 #[ms] membrane time constant
        theta  = 20.0 #[mV] threshold for firing
        t_ref  =  2.0 #[ms] refractory period
        E_L    =  0.0 #[mV] resting potential
        
        ''' Synapse Model Property '''
        delay      = 0.1             #[ms] synaptic delay
        J_ex_mean  = 0.1             #[mV] exc. synaptic strength
        g          = 5.0             #ratio between inh. and exc.
        J_in_mean    = -g*J_ex_mean         #[mV] inh. synaptic strength
        
        
        ''' Uniform noise to evoke oscillation'''
        J_ex  = 2*J_ex_mean * pl.rand(NE+NI,NE)
        J_in  = 2**J_in_mean* pl.rand(NE+NI,NI)
        
        ''' Gaussian Noise to evoke oscillation''
        J_ex    = J_ex_mean+ 0.1*pl.randn(NE+NI,NE)   # 0.1     #[mV] exc. synaptic strength
        J_in        =J_in_mean -g*pl.randn(NE+NI,NI) 
        '''
        
        ''' Network Noise Property '''
        eta    = 2.0                 #fraction of ext. input
        nu_th  = theta/(J_ex_mean*tauMem) #[kHz] ext. rate
        nu_ext = eta*nu_th           #[kHz] exc. ext. rate
        p_rate = 1000.0*nu_ext       #[Hz] ext. Poisson rate
        
        
        ''' Injective Current Property '''
        
        amplitude =kwvar.get('amp') 
        start = 200.0
        stop = 800.0

#%%

        print "Creating network nodes …"
        nest.SetDefaults("iaf_psc_delta", {"C_m"  : tauMem,
                                           "tau_m": tauMem,
                                           "t_ref": t_ref,
                                           "E_L"  : E_L,
                                           "V_th" : theta})
        nodes_ex = nest.Create("iaf_psc_delta", NE)
        nodes_in = nest.Create("iaf_psc_delta", NI)
        
        
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
        
        conn_params_ex = {'rule': 'all_to_all'}
        syn_params_ex = {"model": "excitatory", 'weight': J_ex, 'delay': delay }
        nest.Connect(nodes_ex, nodes, conn_params_ex, syn_params_ex)
        
        
        conn_params_in = {'rule': 'all_to_all'}
        syn_params_in = {"model": "inhibitory", 'weight': J_in, 'delay': delay }
        nest.Connect(nodes_in, nodes, conn_params_in, syn_params_in)
        
        
        endbuild = time.time()
        print "Simulating", simtime, "ms …"
        
        nest.Simulate(simtime)
        '''
        Calculating the mean firing rate of the population.
        
        The stimulation period is [200ms 800 ms], but only use the response 
        in [500ms 800ms] to acquire stable neural response. 
        
         
        '''       
        endsimulate = time.time()
        events_ex   = nest.GetStatus(espikes, "events")[0]
        spikes_times_ex = events_ex['times'][events_ex['times'] >= 500] #sampling spikes after 500 ms 
        rate_ex     = len(spikes_times_ex)/0.3/N_rec
        events_in   = nest.GetStatus(ispikes, "events")[0]
        spikes_times_in = events_in['times'][events_in['times'] >= 500] #sampling spikes after 500 ms 
        rate_in     =len(spikes_times_in)/0.3/N_rec
        
        synapses_ex = nest.GetDefaults("excitatory")["num_connections"]
        synapses_in = nest.GetDefaults("inhibitory")["num_connections"]
        #synapses_ex = nest.GetStatus("excitatory","num_connections")
        #synapses_in = nest.GetStatus("inhibitory", "num_connections")
        
        synapses    = synapses_ex + synapses_in
        build_time  = endbuild-startbuild
        sim_time    = endsimulate-endbuild
        
#        print "\n\nNetwork nodes are created and Connected."
#        print "Simulating", simtime, "ms …"
#        print "Brunel network simulation summary:"
#        print "Number of neurons :", len(nodes)
#        print "Number of synapses:", synapses
#        print "       Exitatory  :", synapses_ex
#        print "       Inhibitory :", synapses_in
#        print "Excitatory rate   : %.2f Hz" % rate_ex
#        print "Inhibitory rate   : %.2f Hz" % rate_in
#        print "Building time     : %.2f s" % build_time
#        print "Simulation time   : %.2f s" % sim_time      
    
        '''
        convlulve Gaussian distribution func with PSTH and take the mean to calculate Population activity
        
        We use 20/resl as the size of bin of the PSTH, and then doing Gaussian 
        
        filtering to the PSTH, taking the mean of the filtered signal as the  
        
        
        '''     
        if len(events_in['times'])==0  or len(events_ex['times'])==0:
           print 'no event'           
           population_activity = [0.0,0.0]
           return population_activity
        else:           
           def gaussian(x, mu, sig):
                 return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)
            
           #sigma = 0.6  #1.6 for gaussian synapse; 0,6 for uniform synapse
           sigma      = kwvar.get('sigma')       
           vec_gaussian=np.vectorize(gaussian)
            
           Gauss=vec_gaussian(pl.linspace(-10,10,20/resl),0,sigma) 
           
           from scipy.stats import binned_statistic     
           n_bin = int(round(simtime/resl))
           bin_num =20/resl
           bin_count_in = binned_statistic(events_in['times'],events_in['times'],statistic = 'count',bins = bin_num ,range=(500,stop))       
           bin_count_ex = binned_statistic(events_ex['times'],events_ex['times'],statistic = 'count',bins = bin_num ,range=(500,stop))       
           cvl_in = np.convolve(bin_count_in[0],Gauss,'full')
           cvl_ex = np.convolve(bin_count_ex[0],Gauss,'full')
           
           # calculate population activity
           bin_start_plot = bin_num/2
           bin_stop_plot  = bin_num/2+bin_num 
           actv_in =cvl_in[ bin_start_plot : bin_stop_plot] # only the fully overlapped part of convolution is used
           actv_ex =cvl_ex[ bin_start_plot : bin_stop_plot]
#           pop_in = binned_statistic(actv_in,actv_in,statistic='count',bins =20/window)
#           pop_ex = binned_statistic(actv_ex,actv_ex,statistic='count',bins =20/window)
#           print events[0]['events']['times']
#           print 'NI population activity is :',pl.mean(pop_in[0] )
#           print 'NE population activity is :',pl.mean(pop_ex[0] )
#           print '\n'; print '\n';print'\n'
#           
#           population_activity    = [0.0,0.0]       
#           population_activity[0] = pl.mean(pop_in[0] )
#           population_activity[1] = pl.mean(pop_ex[0])
#           print population_activity
#           return population_activity
           population_activity  = [0.0,0.0]       
           population_activity[0] = pl.mean(actv_in)/(N_rec*(simtime/bin_num))
           population_activity[1] = pl.mean(actv_ex)/(N_rec*(simtime/bin_num))
           print population_activity
           return population_activity
       #%% plot figures *args = N_rec, sim_time, events,cvl_in, clv_ex,resl,sigma,bin_start_plot,bin_stop_plot
        print plot
        if plot == True :
           print 'Plotting...'
       
           pl.figure(1)
            
           pl.axis([0,simtime ,0, 2*N_rec])
           time_ex =events_ex['times']
           send_ex =events_ex['senders']
           time_in =events_in['times']
           send_in =events_in['senders']
            
           pl.scatter(time_ex,send_ex-np.min(send_ex),s = 2,color = 'b',label='NE')
        
           pl.hold(True)
            
           pl.scatter(time_in,send_in-np.min(send_in)+np.max(send_ex),s = 2,color = 'r',label='NI')
        
           pl.title('Raster Plot')
           pl.xlabel('time /ms')
           pl.ylabel('i_th neuron')
           pl.legend()
        
           ''' plot inhibiotory neuron'''
           pl.figure(2)
    
           hist_in=pl.hist(events_in['times'],bins = bin_num ,alpha = 0.4,label = 'PSTH') #return a tuple,[ value, edge  ]      
           lgd = 'filtered, sig = %.1f'  % sigma
           pl.plot(hist_in[1][0:bin_num],cvl_in[ bin_start_plot : bin_stop_plot], 'black',linewidth = 2.0, label= lgd)
           pl.legend(loc= 'upper right')  # some problem with legend.
           pl.title('PSTH and Gaussian filtered curve of NI')
           pl.xlabel('time /ms')
           pl.ylabel('# events')
        
           pl.show()
            
           '''plot excitatory neuron'''
            
           pl.figure(3)   
     
           hist_ex=pl.hist(events_ex['times'], bins = bin_num , alpha = 0.4, label='PSTH')            
           lgd = 'Gaussian filter, sig = %.1f'  % sigma    
           pl.plot(hist_ex[1][0:bin_num],cvl_ex[ bin_start_plot : bin_stop_plot], 'black',linewidth = 2.0, label=lgd)
           pl.legend(loc='upper right')  # legend([handel_name_1,..],[legend 1,..])
           pl.title('PSTH and Gaussian filtered curve of NE')
           pl.xlabel('time /ms')
           pl.ylabel('# events')
           pl.show()
    
    
def saveLoad(opt):
    import pickle
    global calc
    if opt == "save":
        a = file( 'filename' , 'wb')
        pickle.dump(calc, a, 2)
        a.close
        print 'data saved'
    elif opt == "load":
        f = file('filename' , 'rb')
        calc = pickle.load(f)
    else:
        print 'Invalid saveLoad option'
    

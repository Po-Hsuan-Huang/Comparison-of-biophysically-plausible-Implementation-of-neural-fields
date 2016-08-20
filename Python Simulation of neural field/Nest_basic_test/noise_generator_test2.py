# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 20:16:29 2015

@author: pohsuanhuang
"""
import nest
import pylab as pl
nest.ResetKernel()
nest.SetKernelStatus({'local_num_threads': 4})  # show this work for multiple threads
g = nest.Create('poisson_generator',
                params={'rate': 10.0, 
                        })
p = nest.Create('parrot_neuron', 100)
s = nest.Create('spike_detector')
nest.Connect(g, p, 'all_to_all')
nest.Connect(p, s, 'all_to_all')
nest.Simulate(1000.0)
ev = nest.GetStatus(s)[0]['events']
pl.figure()
pl.plot(ev['times'], ev['senders']-min(ev['senders']),'bo')
pl.yticks([])
pl.title('One spike train for all targets')
pl.show()
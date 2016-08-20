# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 17:43:36 2015

@author: pohsuanhuang
"""

import nest

import pylab as pl

nest.ResetKernel()

p=nest.Create('poisson_generator',1, params=[{'rate':10.0}])
n=nest.Create('parrot_neuron',100)


sp = nest.Create('spike_detector')
nest.Connect(p,n,'all_to_all')
nest.Connect(n,sp,'all_to_all')

nest.Simulate(200.)

ev = nest.GetStatus(sp)[0]['events']
# plotting


pl.plot(ev['times'],ev['senders']-min(ev['senders']),'o')
pl.show()
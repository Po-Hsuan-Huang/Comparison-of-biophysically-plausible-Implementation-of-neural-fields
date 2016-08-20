# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 17:30:47 2015

@author: pohsuanhuang

Test Nest Gerstenr model 

1. Compare single model behavior with Matlab result. plot firing pattern 
and F-I curve.


"""

#%%
import sys
sys.path.append('/Users/pohsuanhuang/Desktop/Curriculum Section/2015WS/Lab rotation_Martin_Giese/Nest_code/Nest practice/Gerstner') 
import pylab as pl
import time
"""
Gerstner firing rate

Single neuron

"""

startbuild= time.time()

#import Gerstner_test_done as GT  # Using spike counts per window to estimate firing rate. not accurate
import Gerstner_test_ISI as GT
print 'Gerstner \n'
start = 0
stop = 10000
step = 100

rate_stable = GT.firing_rate_stable(start,stop,step,False)
rate_onset = GT.firing_rate_onset(start,stop,step,False)

endsimulate= time.time()

sim_time = endsimulate-startbuild
pl.figure(1)
pl.plot(range(start,stop+step,step),rate_stable,label='stable');
pl.hold(True)
pl.plot(range(start,stop+step,step),rate_onset, '-.',label = 'onset');
pl.legend(loc='upper left')
pl.xlabel('stimulus pA');pl.ylabel('firing rate 1/s')
pl.title('Gerstner')
pl.show()
print 'simluation time lapse : %.3f '% sim_time


import pickle
pickle.dump((rate_onset,rate_stable),open('save.p','w'))
#%%
"""
iaf firing rate
Single neuron
"""
import sys
sys.path.append('/Users/pohsuanhuang/Desktop/Curriculum Section/2015WS/Lab rotation_Martin_Giese/Nest_code/Nest practice/integrate and fire')
import pylab as pl
import time
startbuild= time.time()

#import iaf_test as iT   # Using spike counts per window to estimate firing rate. not accurat
import iaf_test_ISI as iT


start =0
stop = 10000
step = 100

rate = iT.firing_rate(start,stop,step,False)

endsimulate= time.time()

sim_time = endsimulate-startbuild
pl.figure(2)
pl.plot(range(start,stop+step,step),rate)
pl.xlabel('stimulus pA');pl.ylabel('firing rate 1/s')
pl.title('LIF')
pl.show()
print 'simluation time lapse : %.3f '% sim_time








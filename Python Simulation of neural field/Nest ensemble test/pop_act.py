# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 09:31:37 2015

@author: pohsuanhuang
"""

'''
This code runs specified neuron model with the specified parameters
// required arguments
model    : 'Gerstner','iaf'
window   :  population activity grouping time window , cannot be smaller than resolution 
            or bigger than sim_time

//optional parameters (keyworded arguments)
pop      : size of the neurons population
resl     :resolution of the simulation
delay    : synapse delay. in ms
amp      :  value of injecting current in pA
sigma    : Gaussian filter sigma
plot     : True / False

//default values

pop      :1000
resl     :0.1
delay    :0.1
amp      :800
sigma    :0.6
plot     :False

'''

#def main():



def pop_act(model,window,**kwargs):
    
    print 'model name :', model
    print 'window :', window
    
    dic={'pop':100, 'resl':0.1, 'delay':0.1,'amp':800.0, 'sigma':0.6,'plot':False}
    kwvar=dic
    kwvar.update(kwargs)
    if model =='iaf':
       print ' run Brunel model...'
       import Brunel_ensemble       
       pop_activity = Brunel_ensemble.simulate(window,**kwvar) 
       return pop_activity
    elif model =='aeif':
       print 'run Gerstner model...'
       import Gerstner_ensemble
       pop_activity = Gerstner_ensemble.simulate(window,**kwvar)
       return pop_activity
    else :
       print 'model name incorrect'


if __name__ == '__main__':
    import pylab as pl
    events_in = []
    events_ex = []
    amp = []
    pop = 100
    step = 100.0
    for i in range(50) : 
        log = pop_act('aeif',10, pop=100, amp = step*(1+i),resl= 0.1, plot = False)
        popin = log[0]  #pop_act for inhibitory neurons
        popex = log[1]  #pop_act for excitatory neurons
        popact= 0.2*popin+0.8*popex
        events_in.append(popact)        
        
        amp.append(step*(1+i))
    pl.figure(1)   
    pl.scatter(amp,events_in,color = 'b')
    pl.xlabel('injecting current (pA)')
    pl.ylabel('pop activity (#/window)')
    pl.title('threshold function of the ensemble')
    pl.axis('tight')
    pl.show()
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 17:13:35 2015

@author: pohsuanhuang

Description of Dynamic neural field:

A simple neural field consists of 31 indentical neurons encoding spacial informaiton of 

31 locations are assosiated to each other via interaction kernel funW. The neural field

recieve stimulus funS, which is a function of location. The intensity of funS 

is proportional to the intensity of injecting current to each neuron.


The neural activity u(x,t) is interpreted as membrane potential, and 
the threshold funciton of the neural field is actualized through spike
triggered neuron interaction.





Task 1:

Use different intensity of funS to find the stable states of the field.

i.e., adjuct variable fac and h and observe the dynamics.

Task 2:

Change the stimulating mechanism from intensity encoding to firing rate encoding.

What are the differences ? Why ?  



"""
import time

startbuild = time.time()
import numpy as np
import nest
import nest.raster_plot
import pylab as pl

''' Simulation parameters '''

simtime = 1500.0

resl = 0.1  # ms

''' Neurons specs '''

pop = 31 # number of neuron ensemble must be a odd number
h = 2*pl.pi/(pop-1) # interval between neurons, receptive field from -pi to pi.


''' Neuron Model Property '''
C_m     =  281.0   #[pF] membrane capacity
tau_w   =  144.0   #[ms] Adaptation time constant
V_th   =  -50.4   #[mV] threshold for firing
peak    =  20.0   #[mV] spike detection threshold (must be larger than V_th)
t_ref   =  2.0    #[ms] refractory period
E_L     = -70.6    #[mV] resting potential

''' Synapse Model Property'''
delay   = 0.1 #1.5             #[ms] synaptic delay
'''weight kernel 
It is a Mexican hat function consists of two cosine,

but one can also use Gabor function.



'''


def  funW(x) :
    A = 5.5   
    a =  0.5
    B =  5.0
    b =  2.0
#    y1 = A*pl.exp(-x**2 /(4*a**2))
#    y2 = - B*pl.exp(-x**2/(4*b**2))
    
    fac = 10 # Kernel strength
    h = 3 # resting level
    ''' Mexican Hat'''
    y= A*pl.exp(-x**2 /(4*a**2))- B*pl.exp(-x**2/(4*b**2)) -h
    ''' Gabor'''
#    y= A/(pl.sqrt(pl.pi)*b) * pl.exp(-x**2/(4*b**2)) * pl.cos(b*x) -h

    y = fac * y
    return y
vec_funW =pl.vectorize(funW)
s = pl.linspace(-pl.pi,pl.pi,pop)
weight = vec_funW(s)    # weight matrix   1 by pop   

 
''' Stimulation specs '''

amplitude =1800.0  #pA
start = 200.0
stop = 800.0


''' define the stimulation  '''

def funS(x):
    c =0.6
    mean = 0.0
    I = 10
    y = I/(pl.sqrt(2*pl.pi)*c) * pl.exp(-(x-mean)**2/(2*c**2))
    return y
    
vec_funS = pl.vectorize(funS)
s = pl.linspace(-pl.pi,pl.pi,pop)
stim = amplitude*vec_funS(s) # stimulation matrix 1 by pop


nest.ResetKernel()


nest.SetKernelStatus({'local_num_threads':4})
nest.SetKernelStatus({"resolution": resl,"overwrite_files" : True})


''' Set defaults'''

nest.SetDefaults("aeif_cond_alpha", {"C_m"  : C_m,
                                   "tau_w": tau_w,
                                   "E_L"  : E_L,
                                   "V_th" : V_th,'V_peak':peak,'t_ref':2.0})
'''Create nodes'''

''' stimulatus'''

'''
 Poisson generator sents independent randomized spikes to each nodes.
'''
params_pg =[]
params_pg =[{'rate':stim[i]} for i in range(0,pop)] # parameter of multiple nodes is a list of dictionary
nest.SetDefaults('poisson_generator',{"start":start, "stop":stop})
pg=nest.Create('poisson_generator', pop , params_pg) 

params_dc =[{'amplitude':stim[i]} for i in range(0,pop)] # parameter of multiple nodes is a list of dictionary
nest.SetDefaults('dc_generator',{"start":start, "stop":stop})
dc = nest.Create('dc_generator',pop,params_dc)
''' neurons'''

neuron= nest.Create('aeif_cond_alpha',pop)



'''spike detector '''
sp = nest.Create('spike_detector',1, {"withtime": True,
                                    "withgid" : True,})
                                    
                                    
''' multimeter'''

mult = nest.Create('multimeter',params = {'withtime':True,'record_from':['V_m']})

'''Create connections'''



nest.SetDefaults("static_synapse", {"delay": delay})

'''connect stimulation to neuorns'''
for i  in range(0,pop):
#    nest.Connect([pg[i]],[neuron[i]])
    nest.Connect([dc[i]],[neuron[i]])

'''connect neurons with weight kernel'''
for i in range (0,pop):
    for j in range(0,pop):
        para = {'weight': weight[i-j]}
        conn_para = {'rule':'one_to_one'}
        nest.Connect([neuron[i]],[neuron[j]],conn_para, para)  # periodic boundary condistion
        
#        if abs(i-j)<=(pop-1)/2:  # distance larger than halfwidth of kernel
#            nest.Connect([neuron[i]],[neuron[j]],{'weight':weight[i-j]})  # periodic boundary condistion
##The following code can be spared because list[-1] = list[end] 
#        elif (i-j)>(pop-1)/2:   # the efferent farther than halfwidth away
#            nest.Connect([neuron[i]],[neuron[j]],{'weight':weight[ (i-j-pop)]})  # periodic boundary condistion
#        elif (i-j)<-(pop-1)/2:
#            nest.Connect([neuron[i]],[neuron[j]],{'weight':weight[ (i-j+pop)]})  # periodic boundary condistion
'''connect spike dectector to neurons'''

nest.Connect(neuron,sp,{'rule':'all_to_all'})
'''connect neurons with multimeter'''
nest.Connect(mult,neuron,{'rule':'all_to_all'})

''' simulation '''
nest.Simulate(simtime)

endsimulate = time.time()

sim_time = endsimulate - startbuild
print 'simulaiton time: %f' %sim_time
pl.clf
pl.figure(1)
#nest.raster_plot.from_device(sp, hist=True)
y = nest.GetStatus(sp,'events')[0]
z = nest.GetStatus(neuron,'local_id')[0]
pl.scatter(y['times'],y['senders']-pl.amin(z))
pl.xlim([0.0,1000.])
pl.ylim([0,pop])
pl.xlabel('ms')
pl.ylabel('neuron id ')
pl.show()
#%%
''' 3D surface plot of membrane potential'''

m_data = nest.GetStatus(mult,'events')[0]  # data of multimeter
z = nest.GetStatus(neuron,'local_id')[0]

#X, Y = np.meshgrid(y['times'],y['senders']-pl.amin(z))

#ax.plot_surface(X,Y,y['V_m'])
#uniq= pl.diff(pl.find((m_data['senders'])==m_data['senders'][0]))# find the distance of the same sender 

#if  sum(pl.diff(uniq))==0 :#checking the sequence of the sender set is of the same length
#    shapex=uniq[0]      #
#    shapey=2*len(uniq)    #
#    X = np.reshape(m_data['senders']-pl.amin(z),(shapex,shapey))
#    Y = np.reshape(m_data['times'],(shapex,shapey))
#    Z = np.reshape(m_data['V_m'],(shapex,shapey))
#    Axes3D.plot_surface(X,Y,Z,cmap=cm.viridis)


'''Create meshgrid from raw data m_data'''
m_sender = list(set(m_data['senders']))
m_time   = list(set(m_data['times']))
X,Y,Z,Z_conv = [],[],[],[]

def Gauss(x,sig) : y = 1/(pl.sqrt(2*pl.pi)*sig) * pl.exp(-x**2/(2*sig**2)); return y

for i in m_sender:
    
    X.append( m_data['times'][(m_data['senders'] == i)])
    Y.append( (np.ones((1,len(m_time)))*i -pl.amin(z)).tolist()[0] )
    Z.append( m_data['V_m'][(m_data['senders'] == i)])
    G = Gauss(pl.arange(-20,20,resl),1.2)
    smooth = np.convolve(m_data['V_m'][(m_data['senders'] == i)]-E_L,G,'same')
    Z_conv.append(smooth+E_L)

X,Y,Z,Z_conv= np.array(X),np.array(Y),np.array(Z) ,np.array(Z_conv)

from mpl_toolkits.mplot3d import Axes3D


fig=pl.figure(2,figsize=pl.figaspect(0.5))
#---- First subplot
ax= fig.add_subplot(121,projection='3d')
surf = ax.plot_surface(X,Y,Z,cmap=pl.cm.jet,linewidth=0,rstride=5, )
fig.colorbar(surf)
pl.xlim([0.0,1500.])
pl.ylim([0,pop])
ax.set_zlim(-70.,-40.)
ax.view_init(azim=0, elev=90) # set view angle normal to X-Y plane
ax.set_xlabel('ms')
ax.set_ylabel('neuron id ')
ax.set_zlabel('V_m')
pl.title('V_m')

#---- Second subplot
ax2 = fig.add_subplot(122, projection='3d')
Z_conv = (Z_conv-pl.mean(Z_conv))/(pl.amax(Z_conv)-pl.amin(Z_conv)) *(pl.amax(Z)-pl.amin(Z)) + E_L
surf = ax2.plot_surface(X,Y,Z_conv,cmap=pl.cm.jet,linewidth=0,rstride=5, )
fig.colorbar(surf)
pl.xlim([0.0,1500.])
pl.ylim([0,pop])
ax2.set_zlim(-70.,-40.)
ax2.view_init(azim=0, elev=90)
ax2.set_xlabel('ms')
ax2.set_ylabel('neuron id ')
ax2.set_zlabel('V_m')
pl.title('Convolved')
pl.show()




#%% Colormesh
#fig3= pl.figure(3)
#
#ax2 = fig.add_subplot(111, projection='3d')
#
#
#cmesh = pl.pcolormesh(X,Y,Z_conv, cmap=pl.cm.jet,vmin = -70., vmax=-40.)
#fig3.colorbar(cmesh)
##ax2.set_xticklabels(m_time)
##ax2.set_yticklabels(m_sender)
##pl.clim(-70.,-40.)
#ax2.set_xlabel('ms')
#ax2.set_ylabel('neuron id ')
#ax2.set_zlabel('V_m')
#
#
#
#
#pl.show()
#
pl.figure(3)
pl.subplot(211)
pl.plot(Z[15,:]);pl.hold(True);pl.plot(Z_conv[15,:])
pl.title('V_m')

pl.subplot(212)
pl.title('kernel shape')
pl.plot(G)
pl.show()
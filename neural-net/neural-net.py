#/usr/bin/python3

from __future__ import print_function, division
import sys
import functools
import numpy as np

@functools.lru_cache()
def eta(s, eta_reset, t_membran, spike_delta=10):
    ''' Refractory function '''
    if s == 0:
        return spike_delta
    else:
        return eta_reset*np.exp(-s/t_membran)

@functools.lru_cache()
def eps(s, t_current, t_membran):
    ''' Returns a current, given by a spike that came in s seconds before. '''
    return (1/(1-t_current/t_membran))*(np.exp(-s/t_membran) - np.exp(-s/t_current))

def run(time, weights, dt=0.001, Vi=-0.06):
    '''
        Runs the simulation.
    :param Vi: Initial current
    :return: last current and spike matrix
    '''

    n = weights.shape[0]# Number of neurons
    Vt = -0.05          # Threshold
    Vr = -0.06          # Reset current
    tau_m = 0.020       # Membran time constant
    tau_c = 0.001       # Current time constant

    # Set  nxn spike matrix
    spikes = np.zeros((n,n), dtype=bool)

    V = np.ones(n)*Vi   # Current

    for t in np.arange(0, time+dt, dt):
        np.where(weights)
        # Find all connections
        #for j, i in  np.where(weights):
         #   pass


#######################
# Set up Neural network
#######################

initial_weight = 3
n1 = 5 # Number of neurons in first group
n2 = 5 # Number of neurons in 2nd group
n = n1 + n2 # total number of neurons

weights = np.zeros((n,n))

# Connect group 1 to group 2
weights[:n1,n1:] = np.ones((n1,n2))
weights = weights * initial_weight

print(weights)

run(1, weights)

eta(4, 2, 3)

print('Simulation finished')
print(eta.cache_info())
print(eps.cache_info())
    

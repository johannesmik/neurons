from __future__ import division
import math
import numpy as np
from pylab import *


# Global configs
T = 500                         # Simulation time in ms
dt = 1                          # Simulation timesteps in ms
t_if = -100                     # Last spike at neuron i (the neuron we look at)
t_jf = [50, 60, 70, 80, 90, 100, 110]    # Spikes at neuron j
t_current = 0.00007             # Time of current (t_s)
t_membran = 100                 # Membran-time-constant (t_m)
w = 1                           # Weight connecting neuron i and j
time = np.arange(0, T+dt, dt)   # Timeline
V = np.zeros(len(time))         # Voltage at the Neuron (over sim. time)
nu_reset = 0.0005               # Reset of potential (nu_0)
delta = 5                       # Critical voltage
spike = 100                     # Spike delta

def nu(s):
    if s == 0:
        return spike # Spike delta
    else:
        return - nu_reset*np.exp(-s/t_membran)

def eps(s):
    return (1/(1-t_current/t_membran))*(np.exp(-s/t_membran) - np.exp(-s/t_current))


# Simulation of a single neuron, with a single input neuron

for i, t in enumerate(time):
    # Input current
    input_current = 0
    for t_f in t_jf:
        if t_f < t:
            input_current += eps(t-t_f)
    # We only look at the last spikes here
    u = nu(t-t_if) + w * input_current
    if u > delta:
        t_if = t
        u += spike
    print 'u is ', u, 'at time', t
    V[t] = u

## plot membrane potential trace
plot(time, V)
title('Simplified Spike Response Model SRM0')
ylabel('Membrane Potential (V)')
xlabel('Time (msec)')
ylim([-80,100])
show()
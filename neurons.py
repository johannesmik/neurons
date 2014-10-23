''' Not very efficient code, but quite readable '''

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt

dt = 1                    # time-step in milliseconds
T = 1000                  # total simulation time in milliseconds

nu_reset = 80
t_membran = 30
t_current = 20
c_initial = -60           # Initial current
delta = -55               # Spiking threshold

def eta(s, nu_reset, t_membran):
    ''' Refractory function '''
    if s == 0:
        return 10 # Spike delta

    else:
        return - nu_reset*np.exp(-s/t_membran)

def eps(s, t_current, t_membran):
    ''' Returns a current, given by a spike that came in s seconds before. '''
    return (1/(1-t_current/t_membran))*(np.exp(-s/t_membran) - np.exp(-s/t_current))

class Neuron(object):
    def __init__(self, initial_current=-60):
        self.current = [initial_current]
        self.spikes = np.array([])
        self.connected_neurons = []

    def calculate_current(self, t):
        ''' Return the incoming current for time t and save it '''

        if t == 0:
            return self.current[0]

        input_current = 0
        for weight, neuron in self.connected_neurons:
            for t_f in neuron.spikes[t-100*dt < neuron.spikes]:
                input_current += weight * eps(t - t_f, t_current, t_membran)
        # Accord for refractory time -> eta-function
        if self.spikes.any():
            t_i = max(self.spikes) # Last spike
            input_current += eta(t-t_i, nu_reset, t_membran)
        self.current.append(input_current)
        return input_current

    def connect(self, weight, neuron):
        # Multiple connections to the same neuron possible
        self.connected_neurons.append((weight, neuron))

class PoissonNeuron(Neuron):
    ''' Neuron, homogenous Poisson '''
    def __init__(self, rate):
        super(PoissonNeuron, self).__init__()
        self.rate = rate

    def spike(self, t):
        ''' Is there a spike at time t? Based on homog Poisson process. Save spike '''

        # Calculate and save current, but ignore it (debugging reasons)
        self.calculate_current(t)

        if np.random.poisson(self.rate) >= 1:
            self.spikes = np.append(self.spikes, t)
            return True
        else:
            return False

class SpikingNeuron(Neuron):
    ''' Neuron, spikes based on a input spiketrain '''
    def __init__(self, delta=-60):
        super(SpikingNeuron, self).__init__()
        self.delta = delta

    def spike(self, t):
        ''' Is there a spike at time t? Based on incoming spikes. Save spike. '''

        # Calculate and save current
        self.calculate_current(t)

        if self.current[-1] > self.delta:
            self.spikes = np.append(self.spikes, t)
            return True
        else:
            return False


# Create some neurons and connect them
n0 = PoissonNeuron(0.1)
n1 = SpikingNeuron()
n2 = SpikingNeuron()
n1.connect(1,n0)
n2.connect(1,n1)

# Simulate
time = np.arange(0, T+dt, dt) # Time array
for t in time:
    for n in [n0, n1, n2]:
        n.spike(t)

print("simulation finished")

# Plot results
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, sharex=True)
ax0.set_title('Neuron 0, Poisson')
ax1.set_title('Neuron 1')
ax2.set_title('Neuron 2')

ax0.plot(time, n0.current)
ax0.plot(n0.spikes, np.zeros(len(n0.spikes)), 'ro')

ax1.plot(time, n1.current)
ax1.plot(n1.spikes, np.zeros(len(n1.spikes)), 'ro')

ax2.plot(time, n2.current)
ax2.plot(n2.spikes, np.zeros(len(n2.spikes)), 'ro')
plt.show()
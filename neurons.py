''' Not very efficient code, but quite readable '''


import numpy as np
import matplotlib.pyplot as plt

dt = 1      # time-step in ms
T = 500    # total simulation time in ms
p = 100     # 'discrete' spiketrains taken into account
time = np.arange(0, T+dt, dt) # Time array

def eta(s, nu_reset=5, t_membran=20):
    ''' Refractory function '''
    if s == 0:
        return 10 # Spike delta
    else:
        return - nu_reset*np.exp(-s/t_membran)

def eps(s, t_current=1, t_membran=20):
    ''' Returns a current, given by a spike that came in s milliseconds before. '''
    return (1/(1-t_current/t_membran))*(np.exp(-s/t_membran) - np.exp(-s/t_current))

class Neuron(object):
    def __init__(self):
        self.current = []
        self.spikes = np.array([])
        self.connected_neurons = []

    def calculate_current(self, t):
        ''' Return the incoming current for time t '''
        input_current = 0
        for weight, neuron in self.connected_neurons:
            for t_f in neuron.spikes[t-100*dt < neuron.spikes]:
                input_current += weight * eps(t - t_f)

        # Accord for refractory time
        if self.spikes.any():
            t_i = max(self.spikes) # Last spike
            input_current += eta(t-t_i)

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
        ''' Random spike based on homog. Poisson process'''

        # Calculate and save current, but ignore it
        input_current = self.calculate_current(t)
        self.current.append(input_current)

        # TODO: Spikes are also dependant on the timestep variable!
        if np.random.poisson(self.rate) >= 1:
            self.spikes = np.append(self.spikes, t)
            return True
        else:
            return False

class SpikingNeuron(Neuron):
    ''' Neuron, spikes based on a input spiketrain '''
    def __init__(self, nu_reset=5, delta=2):
        super(SpikingNeuron, self).__init__()
        self.nu_reset = nu_reset
        self.delta = delta

    def spike(self, t):
        ''' Is there a spike at time t? '''
        
        # Calculate and save current
        input_current = self.calculate_current(t)
        self.current.append(input_current)

        if input_current > self.delta:
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
for t in time:
    for n in [n0, n1, n2]:
        n.spike(t)

print "simulation finished"

# Plot results
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, sharex=True)
ax0.set_title('Neuron 0, Poisson')
ax0.set_ylim((-2,1))
ax1.set_title('Neuron 1')
ax2.set_title('Neuron 2')

ax0.plot(time, n0.current)
ax0.plot(n0.spikes, np.zeros(len(n0.spikes)), 'ro')

ax1.plot(time, n1.current)
ax1.plot(n1.spikes, np.zeros(len(n1.spikes)), 'ro')

ax2.plot(time, n2.current)
ax2.plot(n2.spikes, np.zeros(len(n2.spikes)), 'ro')
plt.show()

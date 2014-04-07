import math
from __future__ import division

class Neuron(object):
    def __init__(self, t_m, t_s, n_0):
        self.t_m = t_m
        self.t_s = t_s
        self.n_0 = n_0
        self.spiketrain = [0, 120, 457, 900] # absolute times
        self.connected_neurons = []

    def spike(self, t):
        # Returns true if there is a spike at time t
        # and save spikes in the spiketrain variable
        if t in self.spiketrain:
            return True

        # TODO sum up the current of all connected neurons
        for neuron, weight in self.connected_neurons:
            # TODO All previous spikes to time t
            pass


    def connect(self, n, w):
        # Neuron n, Init. weight w
        self.connected_neurons.append((n, w))

    def kernel(self, s):
        if s == 0:
            return True, 0
        else:
            return False, -self.n_0 * math.exp(-s/self.t_m)

    def current(self, s):
        return 1/(1 - self.t_s/self.t_m) * (math.exp(-s/self.t_m) - math.exp(-s/self.t_s))

delta_t = 1 # in milliseconds



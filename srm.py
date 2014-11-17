#/usr/bin/python3
__author__ = 'johannes'

import numpy as np
import functools

def eta(s, nu_reset, t_membran, spike=10):
    return - nu_reset*np.exp(-s/t_membran)

@functools.lru_cache()
def eps(s, t_current, t_membran):
    return (1/(1-t_current/t_membran))*(np.exp(-s/t_membran) - np.exp(-s/t_current))

def eps_vector(k, size, t_current, t_membran):
    vector = np.zeros(size, dtype=float)

    for i in range(k):
        vector[i] = eps(k-i, t_current, t_membran)

    return vector

class SRM:
    def __init__(self, neurons, threshold, t_current, t_membran, nu_reset, verbose=False):
        """

        :param neurons:
        :param threshold:
        :param t_current: Current-time-constant (t_s)
        :param t_membran: Membran-time-constant (t_m)
        :param nu_reset:
        :param verbose:
        :return:
        """
        self.neurons = neurons
        self.threshold = threshold
        self.t_current = t_current
        self.t_membran = t_membran
        self.nu_reset = nu_reset
        self.verbose = verbose
        self.last_spike = np.ones(self.neurons, dtype=float) * -1000000

    def simulate(self, s, w, t):
        """
        Simulate one timestep at time t. Changes the spiketrain in Place at time t!
        Return the total current of all neurons

        :param s: Spiketrain
        :param w: Weights
        :param t: Evaluation time
        :return: total current of all neurons at timestep t (vector)
        """

        neurons, timesteps = s.shape

        epsilon = eps_vector(t, timesteps, self.t_current, self.t_membran)

        # Calculate current
        incoming_spikes = np.dot(w.T, s)
        incoming_current = np.dot(incoming_spikes, epsilon)
        total_current = eta(np.ones(neurons)*t - self.last_spike, self.nu_reset, self.t_membran) + incoming_current

        # Any new spikes?
        neurons_high_current = np.where(total_current > self.threshold)
        s[neurons_high_current, t] = True

        # Update last_spike
        spiking_neurons = np.where(s[:, t])
        self.last_spike[spiking_neurons] = t

        if self.verbose:
            print("SRM Timestep", t)
            print("Incoming current", incoming_current)
            print("Total current", total_current)
            print("Last spike", self.last_spike)
            print("")

        return total_current


if __name__ == "__main__":

    s = np.array([[0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    w = np.array([[0,0,1],[0,0,1],[0,0,0]])

    neurons, timesteps = s.shape

    model = SRM(neurons, threshold=1, t_current=0.3, t_membran=20, nu_reset=5, verbose=True)

    for t in range(timesteps):

        total_current = model.simulate(s, w, t)

        neurons_high_current = np.where(total_current > 1)
        print("These neurons have a high current: ", neurons_high_current)
        print("Spiketrain")
        print(s)

        print("--------------")

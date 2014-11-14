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

def simulate_linearized(s, w, t, last_spike, t_current, t_membran, nu_reset):
    """
    Simulate one timestep at time t.
    Return the total current of all neurons

    :param s:
    :param w:
    :param threshold:
    :param t_current:
    :param t_membran:
    :param nu_reset:
    :return: total current of all neurons at timestep t (vector)
    """

    neurons, timesteps = s.shape

    epsilon = eps_vector(t, timesteps, t_current, t_membran)

    incoming_spikes = np.dot(w.T, s)
    incoming_current = np.dot(incoming_spikes, epsilon)
    total_current = eta(np.ones(neurons)*t - last_spike, nu_reset, t_membran) + incoming_current

    print("SRM Timestep", t)
    print("Incoming current", incoming_current)
    print("Total current", total_current)
    print("Last spike", last_spike)
    print("")

    return total_current


if __name__ == "__main__":

    s = np.array([[0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    s = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    w = np.array([[0,0,1],[0,0,1],[0,0,0]])

    neurons, timesteps = s.shape

    threshold = 1.0
    t_current = 0.3                 # Time of current (t_s)
    t_membran = 20                  # Membran-time-constant (t_m)
    nu_reset = 5

    last_spike = np.ones(neurons, dtype=int) * -1000000

    for t in range(timesteps):

        total_current = simulate_linearized(s, w, t, last_spike, t_current, t_membran, nu_reset)

        # Update spiketrain. Any new spikes?
        neurons_high_current = np.where(total_current > threshold)
        s[neurons_high_current, t] = True

        # Update last_spike
        spiking_neurons = np.where(s[:, t])
        last_spike[spiking_neurons] = t

        print("This neurons spike", neurons_high_current)
        print(s)

        print("--------------")

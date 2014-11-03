'''
Spike Timing Dependent Plasticity
'''

import numpy as np

def learning_window(x, tau):
    """
    Constant Learning Window

    :param x:
    :return:
    """

    if x > 0:
        return np.exp(x/tau)
    elif x < 0:
        return np.exp(-x/tau)
    else:
        return 0

# TODO: Adaptive Learning window?

def stdp(s , w):
    """
    Calculate the weight change

    :param s: Spiketrain of learning window
    :param w: current weights
    :return:
    """


    eta = 0.01
    w_in = 0.5
    w_out = 0.5
    tau = 10.0

    neurons = s.shape[0]

    connected_neurons = np.array(w, dtype=bool)

    outgoing_spikes = np.sum(s, axis=1)
    incoming_spikes = np.dot(connected_neurons.T, outgoing_spikes)

    # Non-Numpy solution for calculating the learning windows
    incoming_neurons = []
    for i in range(neurons):
        incoming_neurons.append([])
        for j in range(neurons):
            if w[j, i] != 0:
                incoming_neurons[i].append(j)

    print("Incoming neurons:", incoming_neurons)

    outgoing_spikes_time = []
    for i in range(neurons):
        outgoing_spikes_time.append([])
        for time, spike in enumerate(s[i,:]):
            if spike:
                outgoing_spikes_time[i].append(time)

    incoming_spikes_time = []
    for i in range(neurons):
        incoming_spikes_time.append([])
        for j in incoming_neurons[i]:
            incoming_spikes_time[i].extend(outgoing_spikes_time[j])

    print("Outgoing Spike Times:", outgoing_spikes_time)
    print("Incoming Spike Times:", incoming_spikes_time)

    windowed = []
    for i in range(neurons):
        windowed.append(0)
        for incoming_time in incoming_spikes_time[i]:
            for outgoing_time in outgoing_spikes_time[i]:
                #windowed[i].append("W(%i - %i)" % (incoming_time, outgoing_time))
                windowed[i] += learning_window(incoming_time - outgoing_time, tau)

    print("Windowed Learning:", windowed)

    windowed = np.array(windowed)

    # Calculate Weight Change
    weight_change = eta * (incoming_spikes * w_in + outgoing_spikes * w_out + windowed)

    print("Weight Change ", weight_change)

    return weight_change

if __name__ == "__main__":

    s = np.array([[0, 0, 1, 1, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1]])

    w = np.array([[0, 0, 1],
                  [0, 0, 1],
                  [0, 0, 0]])

    print("Spike Train", s)
    print("Weights", w)

    stdp(s, w)
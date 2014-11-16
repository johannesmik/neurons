"""
Spike Timing Dependent Plasticity
"""

import numpy as np


def learning_window_neuron(t1, t2_list, tau):
    """
    Return the sum of the learning windows of one neuron.
    :param t1: current time
    :param t2_list: spiking times of neuron
    """
    sum_result = 0
    for t2 in t2_list:
        sum_result += learning_window(t2 - t1, tau)
    return sum_result


def learning_window(x, tau):
    """
    Constant Learning Window

    :param x:
    :return:
    """

    if x > 0:
        return np.exp(x / tau)
    elif x < 0:
        return np.exp(-x / tau)
    else:
        return 0


def stdp(spikes, weights):
    """
    Calculate the weight change. Analyzes the whole spiketrain, but only at the last time (last column).

    :param spikes: Spiketrain of learning window
    :param weights: current weights
    :return: Changes in weights
    """

    eta = 0.01
    w_in = 0.5
    w_out = 0.5
    tau = 10.0

    neurons, current_time = spikes.shape

    connected_neurons = np.array(weights, dtype=bool)

    last_spikes = spikes[:, -1]
    last_spikes = last_spikes[:, np.newaxis]

    print("Last spikes", last_spikes)

    # Calculate the weight change for presynaptic spikes
    weight_change_presynaptic = last_spikes * connected_neurons * w_in
    print("Weight change in:", weight_change_presynaptic)

    # Calculate the weight change for postsynaptic spikes
    weight_change_postsynaptic = last_spikes.T * connected_neurons * w_out
    print("Weight change out:", weight_change_postsynaptic)

    # Calculate the weight changes in regards of the learning window
    spikes_time = []
    for neuron in range(neurons):
        spikes_time.append([])
        for time, spike in enumerate(spikes[neuron, :]):
            if spike:
                spikes_time[neuron].append(time)
    print("Outgoing spikes time", spikes_time)

    neuron_learnwindow = [learning_window_neuron(current_time, x, tau) for x in spikes_time]
    neuron_learnwindow = np.array(neuron_learnwindow, ndmin=2)
    neuron_learnwindow = neuron_learnwindow.T  # Make it a column-vector
    print("Summe: ", neuron_learnwindow, neuron_learnwindow.shape)

    learning_window_presynaptic = (last_spikes.T * connected_neurons) * neuron_learnwindow
    print("presynaptic learning window", learning_window_presynaptic)

    learning_window_postsynaptic = (last_spikes * connected_neurons) * neuron_learnwindow.T
    print("postsynaptic learning window", learning_window_postsynaptic)

    # Total weight change
    weight_change = eta * (weight_change_presynaptic + weight_change_postsynaptic + learning_window_presynaptic
                           + learning_window_postsynaptic)
    print("type of weight change:", type(weight_change))

    return weight_change

if __name__ == "__main__":

    s = np.array([[0, 0, 1, 1, 1],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1]])

    w = np.array([[0, 0, 1],
                  [0, 0, 1],
                  [0, 0, 0]])

    print("Spike Train", s)
    print("Weights", w)

    w = w + stdp(s, w)
    print("updated weights", w)
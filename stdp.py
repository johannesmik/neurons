'''
Spike Timing Dependent Plasticity
'''

import numpy as np

# TODO find a better name for this function and document it
def learning_window_sum(t1, t2_list, tau):
    """
    Return the sum of learning windows, where
    :param t2: Tuple of times
    """
    sum = 0
    for t2 in t2_list:
        sum += learning_window(t2-t1, tau)
    return sum


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


def stdp(s, w):
    """
    Calculate the weight change. Analyzes the whole spiketrain, but only at the last time (last column).

    :param s: Spiketrain of learning window
    :param w: current weights
    :return: Changes in weights
    """

    eta = 0.01
    w_in = 0.5
    w_out = 0.5
    tau = 10.0

    neurons, current_time = s.shape

    connected_neurons = np.array(w, dtype=bool)

    last_spikes = s[:,-1]
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
        for time, spike in enumerate(s[neuron, :]):
            if spike:
                spikes_time[neuron].append(time)
    print("Outgoing spikes time", spikes_time)

    sum = [learning_window_sum(current_time, x, tau) for x in spikes_time] # TODO find better name
    sum = np.array(sum, ndmin=2)
    sum = sum.T # Make it a column-vector
    print("Summe: ", sum, sum.shape)

    learning_window_presynaptic = (last_spikes.T * connected_neurons) * sum
    print("presynaptic learning window", learning_window_presynaptic)

    learning_window_postsynaptic = (last_spikes * connected_neurons) * sum
    print("postsynaptic learning window", learning_window_postsynaptic)

    # Total weight change
    weight_change = eta * (weight_change_presynaptic + weight_change_postsynaptic + learning_window_presynaptic
                           + learning_window_postsynaptic)
    print("type of weight change:", type(weight_change))

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

    w = w + stdp(s, w)
    print("updated weights", w)
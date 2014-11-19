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

class STDP:
    def __init__(self, eta, w_in, w_out, tau, window_size, verbose=False):
        self.eta = eta
        self.w_in = w_in
        self.w_out = w_out
        self.tau = tau
        self.window_size = window_size  # T_l
        self.verbose = verbose

    def weight_change(self, spikes, weights, t):
        """
        Calculate the weight change at time t. Changes the weights in place.

        :param spikes: Spiketrain
        :param weights: current weights
        :return: Changes in weights
        """

        if weights.dtype != 'float':
            raise ValueError('The weight matrix has to be a float array. (Try to create it with dtype=float)')

        # Trim spiketrain, so that it's 'windowed' (look at variable T_l in the script)
        spikes = spikes[:, max(0, t+1-self.window_size):t+1]

        if not spikes.any():
            if self.verbose:
                print("--------------------------")
                print("Calculating STDP weight change at time")
                print("No spikes found")
            return np.zeros(weights.shape)

        neurons, current_time = spikes.shape

        connected_neurons = np.array(weights, dtype=bool)

        last_spikes = spikes[:, -1]
        last_spikes = last_spikes[:, np.newaxis]

        # Calculate the weight change for presynaptic spikes
        weight_change_presynaptic = last_spikes * connected_neurons * self.w_in

        # Calculate the weight change for postsynaptic spikes
        weight_change_postsynaptic = last_spikes.T * connected_neurons * self.w_out

        # Calculate the weight changes in regards of the learning window
        spikes_time = []
        for neuron in range(neurons):
            spikes_time.append([])
            for time, spike in enumerate(spikes[neuron, :]):
                if spike:
                    spikes_time[neuron].append(time)

        neuron_learnwindow = [learning_window_neuron(current_time, x, self.tau) for x in spikes_time]
        neuron_learnwindow = np.array(neuron_learnwindow, ndmin=2)
        neuron_learnwindow = neuron_learnwindow.T  # Make it a column-vector

        learning_window_presynaptic = (last_spikes.T * connected_neurons) * neuron_learnwindow

        learning_window_postsynaptic = (last_spikes * connected_neurons) * neuron_learnwindow.T

        # Total weight change
        weight_change = self.eta * (weight_change_presynaptic + weight_change_postsynaptic + learning_window_presynaptic
                               + learning_window_postsynaptic)

        # Change the weight in place
        weights = weights.__iadd__(weight_change)
        if self.verbose:
            print("--------------------------")
            print("Calculating STDP weight change at time")
            print("Last spikes", last_spikes)
            print("Weight change in:", weight_change_presynaptic)
            print("Weight change out:", weight_change_postsynaptic)
            print("Outgoing spikes time", spikes_time)
            print("Summe: ", neuron_learnwindow, neuron_learnwindow.shape)
            print("presynaptic learning window", learning_window_presynaptic)
            print("postsynaptic learning window", learning_window_postsynaptic)
            print("type of weight change:", type(weight_change))
            print("updatet weights (function):", weights)
            print("")

        return weight_change


if __name__ == "__main__":

    s = np.array([[0, 0, 1, 1, 1],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1]], dtype=bool)

    w = np.array([[0, 1, 1],
                  [0, 0, 1],
                  [0, 0, 0]], dtype=float)

    print("Spike Train", s)
    print("Weights", w)

    learning_model = STDP(eta=0.05, w_in=0.5, w_out=0.5, tau=10.0, window_size=4, verbose=True)
    print("Weight change: ", learning_model.weight_change(s, w, 2))
    print("updated weights", w)
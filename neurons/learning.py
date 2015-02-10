"""
Spike Timing Dependent Plasticity
"""

import numpy as np


class STDP:
    def __init__(self, eta, w_in, w_out, tau, window_size, verbose=False):
        self.eta = eta
        self.w_in = w_in
        self.w_out = w_out
        self.tau = tau
        self.window_size = window_size  # T_l
        self.verbose = verbose

    def learning_window_neuron_pre(self, t1, t2_list):
        """
        Return the sum of the learning windows of one neuron.
        :param t1: current time
        :param t2_list: spiking times of neuron
        """
        sum_result = 0
        for t2 in t2_list:
            sum_result += self.learning_window(t2 - t1)
        return sum_result

    def learning_window_neuron_post(self, t1, t2_list):
        """

        Return the sum of the learning windows of one neuron.

        :param t1: current time
        :param t2_list: spiking times of neuron
        """
        sum_result = 0
        for t2 in t2_list:
            sum_result += self.learning_window(t1 - t2)
        return sum_result

    def learning_window(self, x):
        """
        Constant Learning Window

        :param x:
        :return:
        """
        if x > 0:
            return - np.exp(-x / self.tau)
        elif x < 0:
            return np.exp(x / self.tau)
        else:
            return 0

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
        current_time -= 1   # because index begins with 0

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

        neuron_learnwindow_pre = [self.learning_window_neuron_pre(current_time, x) for x in spikes_time]
        neuron_learnwindow_pre = np.array(neuron_learnwindow_pre, ndmin=2).T  # Make it a column-vector

        neuron_learnwindow_post = [self.learning_window_neuron_post(current_time, x) for x in spikes_time]
        neuron_learnwindow_post = np.array(neuron_learnwindow_post, ndmin=2).T  # Make it a column-vector

        learning_window_presynaptic = (last_spikes.T * connected_neurons) * neuron_learnwindow_pre
        learning_window_postsynaptic = (last_spikes * connected_neurons) * neuron_learnwindow_post.T

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
            print("Neuron learnwindow pre", neuron_learnwindow_pre)
            print("Neuron learnwindow post", neuron_learnwindow_post)
            print("Presyncpit:", learning_window_presynaptic)
            print("Postsynapitc:", learning_window_postsynaptic)
            print("Summe (pres): ", neuron_learnwindow_pre, neuron_learnwindow_pre.shape)
            print("Summe (post): ", neuron_learnwindow_post, neuron_learnwindow_post.shape)
            print("presynaptic learning window", learning_window_presynaptic)
            print("postsynaptic learning window", learning_window_postsynaptic)
            print("type of weight change:", type(weight_change))
            print("updated weights (function):", weights)
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

    import matplotlib.pyplot as plt
    x = np.linspace(-15, 15, 1000)
    y = np.array([learning_model.learning_window(xv) for xv in x])
    plt.plot(x,y)
    plt.show()
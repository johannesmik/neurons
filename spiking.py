#/usr/bin/python3
__author__ = 'johannes'

import numpy as np
import functools

class SRM:
    """ SRM_0 (Spike Response Model) """
    def __init__(self, neurons, threshold, t_current, t_membrane, nu_reset, simulation_window_size=100, verbose=False):
        """
        Neurons can have different t_current, t_membrane and nu_resets: Set those variables to 1D np.arrays of all the same size.

        :param neurons: Number of neurons
        :param threshold: Spiking threshold
        :param t_current: Current-time-constant (:math:`t_s`)
        :type t_current: Float or Numpy Float Array
        :param t_membrane: Membrane-time-constant (t_m)
        :param nu_reset: Reset constant
        :param simulation_window_size: Only look at the n last spikes
        :param verbose:
        :return: ``None``
        """

        # Check user input
        # TODO

        self.neurons = neurons
        self.threshold = threshold
        self.t_current = t_current
        self.t_membrane = t_membrane
        self.nu_reset = nu_reset
        self.simulation_window_size = simulation_window_size
        self.verbose = verbose
        self.last_spike = np.ones(self.neurons, dtype=float) * -1000000
        self.v_plot = np.empty((neurons, 0))

    def eta(self, spikes):
        return - self.nu_reset*np.exp(-spikes/self.t_membrane)

    @functools.lru_cache()
    def eps(self, s):
        r"""
        Evaluate the Epsilon function:

        .. math:: \epsilon (s) =  \frac{1}{1 - \frac{\tau_c}{\tau_m}} (\exp(\frac{s}{\tau_m}) - \exp(\frac{-s}{\tau_c}))
            :label: epsilon

        Returns a single Float Value if the time constants (current, membrane) are the same for each neuron.
        Returns a Float Vector with eps(s) for each neuron, if the time constants are different for each neuron.

        :param s: Time s
        :return: Function eps(s) at time s
        :rtype: Float or Vector of Floats
        """
        return (1/(1-self.t_current/self.t_membrane))*(np.exp(-s/self.t_membrane) - np.exp(-s/self.t_current))

    @functools.lru_cache()
    def eps_matrix(self, k, size):
        """
        **Examples**:
        Test

        :param k:
        :param size:
        :return:
        """

        matrix = np.zeros((self.neurons, size), dtype=float)

        for i in range(k):
            matrix[:, i] = self.eps(k-i)

        return matrix

    def simulate(self, spikes, weights, t):
        """
        Simulate one time step at time t. Changes the spiketrain in place at time t!
        Return the total current of all neurons.

        :param spikes: Spiketrain (Time indexing begins with 0)
        :param weights: Weights
        :param t: Evaluation time
        :return: total current of all neurons at time step t (vector), spikes at time t
        """

        # Check correct user input

        if type(spikes) != np.ndarray:
            raise ValueError("Spiketrain should be a numpy array")

        if type(weights) != np.ndarray:
            raise ValueError("Weights should be a numpy matrix")

        try: t = int(t)
        except: raise ValueError("Variable t should be int or convertible to int")

        if t < 0:
            raise ValueError("Time to be simulated is too small")

        if t >= spikes.shape[1]:
            raise ValueError("Spiketrain too short (0ms -- %dms) for simulating time %d" % (spikes.shape[1]-1, t))

        if weights.shape[0] != self.neurons or self.neurons != weights.shape[1]:
            raise ValueError("Weigths should be a quadratic matrix, with one row and one column for each neuron")

        if spikes.shape[0] != self.neurons:
            raise ValueError("Spikes should be a matrix, with one row for each neuron")

        # Work on a windowed view
        s_t = spikes[:, max(0, t+1-self.simulation_window_size):t+1]

        neurons, timesteps = s_t.shape

        epsilon_matrix = self.eps_matrix(min(self.simulation_window_size, t), timesteps)

        # Calculate current
        incoming_spikes = np.dot(weights.T, s_t)
        incoming_current = np.sum(incoming_spikes * epsilon_matrix, axis=1)
        total_current = self.eta(np.ones(neurons)*t - self.last_spike) + incoming_current

        # Any new spikes?
        neurons_high_current = np.where(total_current > self.threshold)
        spikes[neurons_high_current, t] = True

        # Update last_spike
        spiking_neurons = np.where(spikes[:, t])
        self.last_spike[spiking_neurons] = t

        if self.verbose:
            print("SRM Time step", t)
            print("Incoming current", incoming_current)
            print("Total current", total_current)
            print("Last spike", self.last_spike)
            print("")

        self.v_plot = np.hstack((self.v_plot, total_current.reshape(neurons, 1)))

        return total_current

class Izhikevich:
    """
     Izhikevich model

     http://www.izhikevich.org/publications/spikes.htm
    """
    # TODO this model looks wrong, because we receive far to less output spikes!
    def __init__(self, neurons, a, b, c, d, v0=-65, threshold=30, verbose=False):
        """

        :param neurons: total number of neurons
        :param a:
        :param b:
        :param c:
        :param d:
        :param v0: Initial voltage
        :param threshold:
        :param verbose:
        :return:
        """
        print("The Izhikevich model hasn't been tested yet -- use with care")
        self.a = a * np.ones((neurons, 1))
        self.b = b * np.ones((neurons, 1))
        self.c = c * np.ones((neurons, 1))
        self.d = d * np.ones((neurons, 1))
        self.threshold = threshold
        self.verbose = verbose

        # Initial u and v
        self.v = v0 * np.ones((neurons, 1))
        self.u = self.b*self.v
        self.v_plot = np.empty((neurons, 0))

    def simulate(self, spikes, weights, t):
        # Input
        # TODO: is this correct, or should I use another variable instead threshold?
        I = self.threshold * spikes[:, t]
        I = I[:, np.newaxis]
        fired = np.nonzero(self.v >= self.threshold)[0]  # Indices of neurons that spike
        print(fired)
        t = np.array([t+0*fired, fired])
        spikes[fired, t] = True
        self.v_plot = np.hstack((self.v_plot, self.v))

        self.v[fired] = self.c[fired]
        self.u[fired] = self.u[fired] + self.d[fired]

        I = I + np.sum(weights[fired, :], axis=0, keepdims=True)
        self.v = self.v + 0.5 * (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + I)  # step 0.5 ms
        self.v = self.v + 0.5 * (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + I)  # for numerical
        self.u = self.u + self.a * (np.multiply(self.b, self.v) - self.u)             # stability

        if self.verbose:
            print("Izhikevich Time step", t)
            print("Total current", self.v)
            print("Fired: ", fired)
            print("")

        return self.v


if __name__ == "__main__":

    srm_model = SRM(neurons=3, threshold=1, t_current=0.3, t_membrane=20, nu_reset=5, verbose=True)
    izhikevich_model = Izhikevich(neurons=3, a=0.02, b=0.2, c=-65, d=8, threshold=30, verbose=True)

    models = [srm_model, izhikevich_model]

    for model in models:
        print("-"*10)
        if isinstance(model, SRM):
            print('Demonstration of the SRM Model')
        elif isinstance(model, Izhikevich):
            print('Demonstration of the Izhikevich Model')

        s = np.array([[0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                      [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        w = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
        neurons, timesteps = s.shape

        for t in range(timesteps):
            total_current = model.simulate(s, w, t)
            print("Spiketrain:")
            print(s)
__author__ = 'johannes'

import learning
import pytest
import numpy as np

# TODO I quite don't like the fixtures yet, maybe I can parameterize them differently in future?

@pytest.fixture(scope="module")
def spiketrain_small():
    s = np.array([[0, 0, 1, 1, 1],
                  [0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1]], dtype=bool)
    return s

@pytest.fixture(scope="module")
def spiketrain_big():
    s = np.array([[0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 1, 0, 1, 0, 1, 0, 0, 0]], dtype=bool)
    return s

@pytest.fixture(scope="module", params=[spiketrain_small, spiketrain_big])
def spiketrain(request):
    return request.param()

@pytest.fixture(scope="function")
def weights_circle():
     w = np.array([[0, 1, 0],
                   [0, 0, 1],
                   [1, 0, 0]], dtype=float)
     return w

@pytest.fixture(scope="function")
def weights_full():
     w = np.array([[0, 1, 1],
                   [1, 0, 1],
                   [1, 1, 0]], dtype=float)
     return w

@pytest.fixture(scope="function")
def weights_arrow():
     w = np.array([[0, 0, 1],
                   [0, 0, 1],
                   [0, 0, 0]], dtype=float)
     return w

@pytest.fixture(scope="function", params=[weights_circle, weights_full, weights_arrow])
def weights(request):
    return request.param()

@pytest.fixture(scope="module")
def learning_model_empty():
    model = learning.STDP(eta=0.0, w_in=0.0, w_out=0.0, tau=0.0, window_size=5)
    return model

@pytest.fixture(scope="module")
def learning_model_standard():
    model = learning.STDP(eta=0.05, w_in=0.5, w_out=0.5, tau=10.0, window_size=5)
    return model

@pytest.fixture(scope="module", params=[learning_model_empty, learning_model_standard])
def learning_model(request):
    print(request.param)
    return request.param()


class TestNoChanges:

    def test_empty_spiketrain(self, learning_model, weights):
        """ The weights don't change for an empty spiketrain """
        weights_before = weights.copy()
        spikes = np.zeros((3, 1), dtype=bool)
        learning_model.weight_change(spikes, weights, 1)
        print("Weights before:", weights_before, "Weights after:", weights)
        assert not np.may_share_memory(weights_before, weights)
        assert np.array_equal(weights_before, weights)

    def test_empty_model(self, learning_model_empty, spiketrain, weights):
        """ The weights don't change for an empty model (all parameters set to zero) """
        weights_before = weights.copy()
        spikes = np.zeros((3, 1), dtype=bool)
        learning_model_empty.weight_change(spikes, weights, 1)
        print("Weights before:", weights_before, "Weights after:", weights)
        assert not np.may_share_memory(weights_before, weights)
        assert np.array_equal(weights_before, weights)

class TestSimpleChanges:
    """ Test simple changes """

    def test1(self, spiketrain, weights):
        " Only outgoing spikes change the weight "
        neurons, timesteps = spiketrain.shape
        weights_before = weights.copy()

        # Suggested weights
        last_spike = spiketrain[:,-1].reshape((neurons, 1))
        suggested_weights = np.where(weights_before != 0, weights_before + last_spike.T, weights_before)

        # Update weights by the model
        epsilon = np.finfo(float).eps
        model = learning.STDP(eta=1., w_in=0., w_out=1., tau=epsilon, window_size=5, verbose=True)
        model.weight_change(spiketrain, weights, timesteps)

        assert np.array_equal(suggested_weights, weights)

    def test2(self, spiketrain, weights):
        " Only incoming spikes change the weight "
        neurons, timesteps = spiketrain.shape
        weights_before = weights.copy()

        # Suggested weights
        last_spike = spiketrain[:,-1].reshape((neurons, 1))
        suggested_weights = np.where(weights_before != 0, weights_before + last_spike, weights_before)

        # Update weights by the model
        epsilon = np.finfo(float).eps
        model = learning.STDP(eta=1., w_in=1., w_out=0., tau=epsilon, window_size=5)
        model.weight_change(spiketrain, weights, timesteps)

        assert np.array_equal(suggested_weights, weights)

class TestPrecalculatedExample:
    """
    In the script "Theory of Synaptic Plasticity" on page 20 there's
    a concrete example of STDP learning. (I didn't include the spike t_i^2 so that the weights stay equal.)
    Here, I calculated the weight changes manually (with meaningful parameters)
    and compare them with the weight changes calculated by the program.
    """
    #TODO add a graphic in the documentation, for visualizing the times and so on
    #TODO parameterize tests nicely!


    @classmethod
    def setup_class(cls):
        """ setup any state specific to the execution of the given class (which
        usually contains tests).
        """
        cls.spiketrain = np.zeros((2, 51), dtype=bool)
        cls.spiketrain[0, (20, 24, 25, 40, 50)] = True
        cls.spiketrain[1, (10, 45)] = True
        cls.weights = np.array([[0, 1],
                                [0, 0]], dtype=float)
        cls.simulation_started = False
        cls.model = learning.STDP(eta=1, w_in=0.5, w_out=-0.5, tau=5., window_size=10)



    @pytest.mark.parametrize("time, expected_change", [
                             (10, -0.5),
                             (20, 0.5),
                             (40, 0.5),
                             (45, -0.132121),
                             (50, 0.132121)])
    def test_time(self, time, expected_change):

        self.simulation_started = True

        weights_before = self.weights.copy()

        # Expected weights
        expected_weights = np.array([[0, expected_change],[0, 0]])
        expected_weights += weights_before

        self.model.weight_change(self.spiketrain, self.weights, time)
        print("Time:", time)
        print("Weights before", weights_before)
        print("Weights after", self.weights)
        print("Expected weights", expected_weights)

        nullmatrix = np.zeros((2, 2))
        assert np.array_equal(nullmatrix, np.around(expected_weights - self.weights, 5))

    def test_after(self):
        '''
        Tests the weights after all previous tests have finished
        '''
        expected_weights = np.array([[0, 1.5], [0, 0]], dtype=float)
        nullmatrix = np.zeros((2, 2))
        assert np.array_equal(nullmatrix, np.around(expected_weights - self.weights, 5))
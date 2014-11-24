__author__ = 'johannes'

import stdp
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
    model = stdp.STDP(eta=0.0, w_in=0.0, w_out=0.0, tau=0.0, window_size=5)
    return model

@pytest.fixture(scope="module")
def learning_model_standard():
    model = stdp.STDP(eta=0.05, w_in=0.5, w_out=0.5, tau=10.0, window_size=5)
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
        model = stdp.STDP(eta=1., w_in=0., w_out=1., tau=epsilon, window_size=5, verbose=True)
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
        model = stdp.STDP(eta=1., w_in=1., w_out=0., tau=epsilon, window_size=5)
        model.weight_change(spiketrain, weights, timesteps)

        assert np.array_equal(suggested_weights, weights)

class TestSelected:
    pass
from neurons import spiking

__author__ = 'johannes'

import pytest
import numpy as np

class TestVarious:

    def test_neuron_no_spike(self):
        # Neuron should not spike

        timesteps = 20
        # Two neurons
        spiking_model = spiking.SRM(neurons=2, threshold=1.0, t_current=0.3,
                                    t_membrane=20, eta_reset=5, verbose=False)

        # Neuron 1 is connected to Neuron 2 with weight 1
        weights = np.array([[0, 1], [0, 0]])

        # Empty spiketrain of length 'timesteps'
        spiketrain = np.zeros((2, timesteps), dtype=bool)
        current = spiking_model.check_spikes(spiketrain, weights, 19)

        # The outcoming current on both neurons should be zero
        assert np.array_equal(current, np.array([0, 0]))

    def test_negative_weight(self):
        # If weights are negative enough, neuron should not spike
        # The same code as test_neuron_spike but with negative weight!
        timesteps = 20
        # Two neurons
        spiking_model = spiking.SRM(neurons=2, threshold=1.0, t_current=0.3,
                                    t_membrane=20, eta_reset=5, verbose=False)

        # Neuron 1 is connected to Neuron 2 with weight -1
        weights = np.array([[0, -1], [0, 0]])

        # Empty spiketrain of length 'timesteps'
        spiketrain = np.zeros((2, timesteps), dtype=bool)
        # Neuron 1 Spikes all the time :)
        spiketrain[0,:] = 1

        current = spiking_model.check_spikes(spiketrain, weights, 19)

        # The outcoming current on Neuron 1 should be 0
        # on Neuron 2 it should be negative
        assert current[0] == 0
        assert current[1] < 0

    def test_neuron_spike(self):
        # Neuron should spike
        timesteps = 20
        # Two neurons
        spiking_model = spiking.SRM(neurons=2, threshold=1.0, t_current=0.3,
                                    t_membrane=20, eta_reset=5, verbose=False)

        # Neuron 1 is connected to Neuron 2 with weight 1
        weights = np.array([[0, 1], [0, 0]], dtype=bool)

        # Empty spiketrain of length 'timesteps'
        spiketrain = np.zeros((2, timesteps))
        # Neuron 1 Spikes all the time :)
        spiketrain[0,:] = 1

        current = spiking_model.check_spikes(spiketrain, weights, 19)

        # The outcoming current on Neuron 1 should be 0
        # on Neuron 2 it should be positive
        assert current[0] == 0
        assert current[1] > 0

    def test_different_time_constants(self):
        # Each neuron has different time constants
        pass

class TestShouldFail:

    def test_wrong_spiketrain_size(self):

        spiking_model = spiking.SRM(neurons=2, threshold=1.0, t_current=0.3,
                                    t_membrane=20, eta_reset=5, verbose=False)

        # Empty spiketrain is too short
        spiketrain1 = np.zeros((2, 20))

        # Neuron 1 is connected to Neuron 2 with weight 1
        weights = np.array([[0, 1], [0, 0]], dtype=bool)

        with pytest.raises(ValueError) as e:
            current = spiking_model.check_spikes(spiketrain1, weights, 20)
        assert "Spiketrain too short (0ms -- 19ms) for simulating time 20" in str(e.value)

    def test_simulate_wrong_types(self):
        spiking_model = spiking.SRM(neurons=2, threshold=1.0, t_current=0.3,
                                    t_membrane=20, eta_reset=5, verbose=False)

        spiketrain1 = np.zeros((2, 21))
        weights = np.array([[0, 1], [0, 0]], dtype=bool)

        # Spiketrain is not a numpy array
        with pytest.raises(ValueError) as e:
            current = spiking_model.check_spikes([0,0,0], weights, 20)

        # Weights is not a matrix
        with pytest.raises(ValueError) as e:
            current = spiking_model.check_spikes(spiketrain1, [[0,1],[0,0]], 20)

        # Time is not a int
        with pytest.raises(ValueError) as e:
            current = spiking_model.check_spikes(spiketrain1, weights, [20, 13])
        assert "Variable t should be int or convertible to int" in str(e.value)

    def test_wrong_weight_size(self):
        spiking_model = spiking.SRM(neurons=2, threshold=1.0, t_current=0.3,
                                    t_membrane=20, eta_reset=5, verbose=False)

        spiketrain1 = np.zeros((2, 21))

        # Wrong weights
        weights = np.array([[0, 1], [0, 0], [0, 0]], dtype=bool)

        with pytest.raises(ValueError) as e:
            current = spiking_model.check_spikes(spiketrain1, weights, 20)
        assert "Weigths should be a quadratic matrix" in str(e.value)

    def test_wrong_time_too_small(self):
        # Simulate a time that is too small
        spiking_model = spiking.SRM(neurons=2, threshold=1.0, t_current=0.3,
                                    t_membrane=20, eta_reset=5, verbose=False)

        spiketrain1 = np.zeros((2, 20))

        weights = np.array([[0, 1], [0, 0]], dtype=bool)

        with pytest.raises(ValueError) as e:
            current = spiking_model.check_spikes(spiketrain1, weights, -1)
        assert "Time to be simulated is too small" in str(e.value)

    def test_wrong_number_of_constants(self):
        # 3 Neurons, 3 different t_s, but only 2 different t_m
        with pytest.raises(ValueError) as e:
            spiking_model = spiking.SRM(neurons=3, threshold=1.0, t_current=[0.3, 0.2, 0.3],
                                    t_membrane=[0.2, 0.5], eta_reset=[0.5, 0.5, 0.6], verbose=False)
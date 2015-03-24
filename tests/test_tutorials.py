__author__ = 'johannes'

import pytest
import numpy as np
from neurons import spiking, learning

class TestSRMNetwork:
    " The first tutorial: SRM network "

    def test_tutorial_works(self):

        model = spiking.SRM(neurons=3, threshold=1, t_current=0.3, t_membrane=20, eta_reset=5)

        weights = np.array([[0, 0, 1.], [0, 0, 1.], [0, 0, 0]])

        spiketrain = np.array([[0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                               [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)

        for time in range(10):
            total_potential = model.check_spikes(spiketrain, weights, time)

        print("Spiketrain:")
        print(spiketrain)

        expected_spiketrain = np.array([[0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                                        [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                                        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0]], dtype=bool)
        assert np.array_equal(spiketrain, expected_spiketrain)

class TestLearning:
    " The second tutorial: Learning "

    def test_tutorial_works(self):

        stdp_model = learning.STDP(eta=0.05, w_in=0.5, w_out=0.5, tau=10.0, window_size=5)

        weights = np.array([[0, 0, 1.], [0, 0, 1.], [0, 0, 0]])

        spiketrain = np.array([[0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                               [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                               [0, 0, 0, 1, 0, 0, 0, 0, 1, 0]], dtype=bool)

        for time in range(10):
            stdp_model.weight_change(spiketrain, weights, time)

        print("Weights after")
        print(weights)

        # That's the output that I got during my first run
        expected_weights = np.array([[0, 0, 1.18586337],
                                     [0, 0, 1.17766241],
                                     [0, 0, 0]])
        nullmatrix = np.zeros((3, 3))
        assert np.array_equal(nullmatrix, np.around(expected_weights - weights, 5))

class TestSpikeAndLearn:
    " The third tutorial: Spike and Learn "

    def test_tutorial_works(self):
        srm_model = spiking.SRM(neurons=3, threshold=1, t_current=0.3, t_membrane=20, eta_reset=5)

        stdp_model = learning.STDP(eta=0.05, w_in=0.5, w_out=0.5, tau=10.0, window_size=5)

        weights = np.array([[0, 0, 1.], [0, 0, 1.], [0, 0, 0]])

        spiketrain = np.array([[0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                               [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)

        for time in range(10):
            srm_model.check_spikes(spiketrain, weights, time)
            stdp_model.weight_change(spiketrain, weights, time)

        # Output that I got during my first run. There's a possibility that this is wrong calculations.
        expected_spiketrain = np.array([[0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                                        [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                                        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0]], dtype=bool)
        print(weights)
        expected_weights = np.array([[0, 0, 1.18586337],
                                     [0, 0, 1.17766241],
                                     [0, 0, 0]])

        assert np.array_equal(spiketrain, expected_spiketrain)
        nullmatrix = np.zeros((3, 3))
        assert np.array_equal(nullmatrix, np.around(expected_weights - weights, 5))
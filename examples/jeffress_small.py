"""
In this test we create a simple Jeffress model.
"""

import numpy as np

from neurons import spiking
from neurons import plotting
from neurons import tools


def test_jeffress():
    neurons = 11
    timesteps = 100

    ax_delays = np.array([0, 5, 15, 25, 0, 25, 15, 5, 0, 0, 0])
    model = spiking.SRM_X(neurons=neurons, threshold=np.array([1]*neurons), t_current=np.array([5]*neurons),
                          t_membrane=np.array([10]*neurons), eta_reset=np.array([2.0]*neurons), ax_delay=ax_delays)

    weights = np.zeros((neurons, neurons))

    # Connect input layer
    weights[0, (1, 2, 3)] = 1
    weights[4, (5, 6, 7)] = 1

    # Connect to output layer
    weights[(1, 5), 8] = 1.05
    weights[(2, 6), 9] = 1.05
    weights[(3, 7), 10] = 1.05

    print(weights)

    spiketrain = np.zeros((neurons, timesteps), dtype=bool)

    manual_spiketrain = True

    if manual_spiketrain:
        spiketrain[0, (20, 25, 30)] = 1
        spiketrain[4, (0, 5, 10)] = 1
    else:
        spiketrain[0,:] = tools.sound(timesteps, 85, 0.6, 4)
        spiketrain[4,:] = tools.sound(timesteps, 60, 0.6, 4)

    psth = plotting.PSTH(spiketrain, binsize=5)
    curr = plotting.CurrentPlot(4)

    for t in range(timesteps):
        current = model.check_spikes(spiketrain, weights, t)
        curr.add(current[[0, 3, 7, 10]])

    psth.show_plot()
    curr.show_plot()

if __name__ == "__main__":
    test_jeffress()
    plotting.show()

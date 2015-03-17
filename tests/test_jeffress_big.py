"""
In this test we create a bigger Jeffress model.

 - It consists of 180 output neurons.
 - The axonal delay between the input neurons and the inter-layer neurons are sampled from 0ms to 100ms.
 - We create the same spiketrain with 50 ms difference between the two input neurons.

Over several test runs, we adjust the weight which with the inter-layer is connected to the output layer.
One can see that the better the weights are adjusted, the more precise the outcome of the net is.
"""

import numpy as np
from neurons import spiking, plotting, tools


def test_jeffress(output_weight):

    neurons = 180 * 3 + 2
    timesteps = 300

    # Set the axonal delays
    ax_delays = np.zeros(neurons)
    ax_delays[1:181] = np.linspace(0, 100, 180)
    ax_delays[182:362] = np.linspace(100, 0, 180)

    model = spiking.SRM_X(neurons=neurons, threshold=np.array([1]*neurons), t_current=np.array([45]*neurons),
                          t_membrane=np.array([50]*neurons), eta_reset=np.array([5.0]*neurons), ax_delay=ax_delays, simulation_window_size=500)

    weights = np.zeros((neurons, neurons))

    # Connect to input layer
    weights[0, 1:181] = 1
    weights[181, 182:361] = 1

    # Connect inter-layer to output layer
    weights[1:181, 362:542] = output_weight * np.eye(180)
    weights[182:362, 362:542] = output_weight * np.eye(180)

    # Spiketrain: Same spiketrain on both input neurons, but one is sligthly delayed
    spiketrain = np.zeros((neurons, timesteps), dtype=bool)
    spiketrain[0, (50, 55, 60)] = 1
    spiketrain[181, (0, 5, 10)] = 1

    for t in range(timesteps):
        current = model.check_spikes(spiketrain, weights, t)

    print("output weight", output_weight)
    print("sum over the spikes in the output layer\n", np.sum(spiketrain, axis=1)[362:])

if __name__ == "__main__":

    output_weights = [2.5, 1.6, 1.5, 1.4, 1.3, 1.295]

    for output_weight in output_weights:
        test_jeffress(output_weight)

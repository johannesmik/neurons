Project overview
================


Two methods to rule them all
----------------------------

At the very core of our simulations we only need two methods:

1. `check_spikes()`
2. `learn_weights()`

The first one, `check_spikes()`, checks if the neurons generate new spikes at a given time `t`.
The second one, `learn_weights()`, updates the weights for a given spike train at time `t`.

This easy model of only two methods, makes the simulations very simple. For example:

.. code-block:: python

    timesteps = 3000
    spiketrain = np.array( ... , dtype=bool)
    weights = np.array( ... )

    spike_model = spiking.SRM( ... )
    learning_model = learning.STDP( ... )

    for t in range(timesteps):
        spike_model.check_spikes(spiketrain, weights, t)
        learning_model.learn_weights(spiketrain, weights, t)

This code is sufficient for SRM neurons with STPD learning.

In the next articles :doc:`Spiking models <spiking>` and :doc:`Learning models <learning>` we will have a closer look on those two methods.

.. note::
    Both methods operate in-place:

    * `check_spikes()` will change the spiketrain matrix in-place.
    * `learn_weights()` will change the weights matrix in-place.

Visualizing
-----------

In simulations that involve many neurons, we need methods to investigate our simulations.

Thus, we provide some plotting functions, such as:

# IMAGES

All plotting functions are well documented in the auto-reference in :doc:`Plotting.py <../reference/plotting>`.

We want to add more plots in future. Please give us your ideas for new plots, by writing a :doc:`feature request <feature_request>`.

Tools
-----

We also wrote some tools, for example to create Poisson distributed spiketrains.

Please see the auto-reference at :doc:`Tools.py <../reference/tools>`.

Also, in this section, your ideas are heartily appreciated (:doc:`feature request <feature_request>`).
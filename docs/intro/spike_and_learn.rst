Tutorial: Spike and learn
=========================

In this short tutorial we combine both the spiking and the learning model.


Setting everything up
---------------------

As in the last two tutorials, we

 - import the needed packages (lines 1 -- 2),
 - set up the SRM and STDP models (lines 4 & 6),
 - set up the initial spiketrain and weights (lines 8 & 10).

.. code-block:: python
    :linenos:

    import numpy as np
    from neurons import spiking, learning

    srm_model = spiking.SRM(neurons=3, threshold=1, t_current=0.3, t_membrane=20, eta_reset=5)

    stdp_model = learning.STDP(eta=0.05, w_in=0.5, w_out=0.5, tau=10.0, window_size=5)

    weights = np.array([[0, 0, 1.], [0, 0, 1.], [0, 0, 0]])

    spiketrain = np.array([[0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                           [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=bool)

Simulate
--------

The simulation is again similar. But now we calculate in each timestep first the spike change (line 2) and immediately afterwards
the weight change (line 3):

.. code-block:: python
    :linenos:

    for time in range(10):
        srm_model.check_spikes(spiketrain, weights, time)
        stdp_model.weight_change(spiketrain, weights, time)

That was the major step :) Not too huge, right?

Result
------

Print out the final spiketrain and the weight changes to the console:

.. code-block:: python

    print("Final Spiketrain:")
    print(spiketrain)

    print("Final weights:")
    print(weights)

Which gives us:

::

    Final Spiketrain:
    [[0 0 1 0 0 0 1 1 0 0]
     [1 0 0 0 0 0 1 1 0 0]
     [0 0 0 1 0 0 0 0 1 0]]
    Final weights:
    [[ 0.          0.          1.18586337]
     [ 0.          0.          1.17766241]
     [ 0.          0.          0.        ]]

We get the exact same spiketrain and weights as in the previous two tutorials.

However, it could be that the results are different when we run `check_spikes()` and `weight_change()` at the same time
rather than one after another. The learned weights can influence the check_spikes and produce more or less spikes!

Conclusion
----------

In this part of the tutorial we have seen that

In the next part, we want to :doc:`visualize the neuron currents and weight changes<visualizing>`.
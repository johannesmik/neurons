Spiking models
==============

SRM 0 Spiking Model
-------------------

.. math::
   :label: srm

   u_i(t) = \eta(t - \hat{t}_i) + \sum_j w_{ij} \sum_{t_j^f} \epsilon_0 (t - t_j^f)

TODO: description of w, and :math:`\hat{t}_i`.

Eta function
~~~~~~~~~~~~

The :math:`\eta(s)` function describes the after-potential of a neuron after a spike.

We approximate it like this (x-axis: time, y-axis: potential):

.. math::
    :label: eta

    \eta(s) = \delta(s) - \eta_0 \exp (-\frac{s}{\tau_m})


.. raw:: html

    <canvas id="eta-plot" width="400" height="150"></canvas>

    <p>Reset eta_0:</p>
    <div id="eta-eta0-slider"></div>

    <p>Membrane time constant</p>
    <div id="eta-membrane-slider"></div>

    <style>
      @import "../_static/jquery-ui.css";
      #eta-eta0-slider, #eta-membrane-slider {
        float: none;
        clear: none;
        width: 300px;
        margin: 15px;
      }
    </style>
    <script src="_static/jquery1.10.2.js"></script>
    <script src="../_static/jquery-ui.js"></script>
    <script src="../_static/plot-eta.js"></script>

Epsilon function
~~~~~~~~~~~~~~~~

:math:`\epsilon` describes the time course of a postsynaptic potential caused by spike arrival.

We approximated this function like this:

.. math:: \epsilon (s) =  \frac{1}{1 - \frac{\tau_c}{\tau_m}} (\exp(\frac{-s}{\tau_m}) - \exp(\frac{-s}{\tau_c}))
            :label: epsilon

Where

 * :math:`s` is the time after the arriving spike
 * :math:`\tau_c` is the **current** time constant.
 * :math:`\tau_m` is the **membrane** time constant.

When plotted, it looks like this (x-axis: time, y-axis: potential):

.. raw:: html

    <canvas id="epsilon-plot" width="400" height="150"></canvas>

    <p>Current time constant:</p>
    <div id="current-slider"></div>

    <p>Membrane time constant</p>
    <div id="membrane-slider"></div>

    <style>
      @import "../_static/jquery-ui.css";
      #current-slider, #membrane-slider {
        float: none;
        clear: none;
        width: 300px;
        margin: 15px;
      }
    </style>
    <script src="../_static/plot-eps.js"></script>

Linearization
~~~~~~~~~~~~~

Because we wanted to have a fast implementation of our model, we rewrote the original SRM equation :eq:`srm` as
 a operations on matrices. Those operations (dot-product, row-sum, element-wise product) are fast to compute.

We calculate the membrane potential for every neuron at time t :math:`u(t)`:

.. math::
   :label: srm_linearized

    u(t) = Z(t) + rowsum((W^T \cdot S ) \circ \mathcal{E}_t)

Where

 * :math:`n` is the number of neurons

 * :math:`W^T` the transposed Weight matrix
 * :math:`S` is the Spiketrain previous to time :math:`t` in binary notation
 * :math:`\mathcal{E}_n` is a helper matrix :math:`\begin{pmatrix} \epsilon_1(t) & \epsilon_1(t) & \dots & \epsilon_1(0) \\ \colon & & & \colon \\ \epsilon_n(t) & \epsilon_n(t) & \dots & \epsilon_n(0) \end{pmatrix}`
 * :math:`\epsilon_i(s)` is the epsilon function for neuron i at time s
 * :math:`Z(t) = \begin{pmatrix}\eta(t - \hat{t}_1 & \dots & \eta(t - \hat{t}_n \end{pmatrix}^T`
 * :math:`\circ` is element-wise product
 * *rowsum(M)* is the sum over the rows in a matrix M

To understand this equation better, let's have a look at it's components:

 1. :math:`W^T \cdot S` is a :math:`n \times t` matrix, that says us how many weighted spikes arrive at a neuron at each time
 2. :math:`rowsum((W^T \cdot S) \circ \mathcal{E}_t)` is a :math:`n`-dimensional vector, that gives us the membrane potential of a neuron caused by incoming spikes
 3. :math:`Z(t) + \dots` in the last step we add the after-potential of each neuron

.. note::

    Of course, those matrices could grow very large with time :math:`t`. So in practice, we use an approximation
    on a smaller time-slice. We don't look at the whole spiketrain, but only on a small fraction of it (maybe the last 500ms).

Other Spiking models
--------------------

Other spiking models can be easily integrated later.

All that you need is to implement a
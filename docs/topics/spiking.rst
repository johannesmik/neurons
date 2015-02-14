Spiking models
==============

SRM Spiking Model
-----------------

Epsilon function
----------------

:math:`\epsilon` describes the time course of a postsynaptic potential caused by spike arrival.

We defined epsilon to be a function of two constants, namely one for the current (:math:`\tau_c`) and membrane (:math:`\tau_m`).

.. math:: \epsilon (s) =  \frac{1}{1 - \frac{\tau_c}{\tau_m}} (\exp(\frac{-s}{\tau_m}) - \exp(\frac{-s}{\tau_c}))
            :label: epsilon

The epsilon function looks like this:

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
    <script src="../_static/jquery-ui.js"></script>
    <script src="../_static/plot-eps.js"></script>
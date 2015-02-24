.. _index:

Welcome to Neurons' documentation!
===================================

.. rubric:: *Neurons* is a simple simulation tool for neurons.

It currently supports the **SRM** model for calculating spike trains,
and the **STDP** model for learning synaptic weights.

It originated from a student project at the `Chair for Theoretical Biophysics at TU München`_ in summer term 2014.

.. _Chair for Theoretical Biophysics at TU München: http://www.t35.physik.tu-muenchen.de/


First steps
-----------

Read this if you are new to the project.

* **Installation:**
    :doc:`Linux <intro/linux>` |
    :doc:`Windows <intro/windows>`

* **Tutorials:**
    :doc:`Your first SRM network <intro/srm_network>` |
    :doc:`Learning the weights <intro/learn>` |
    :doc:`Combining spiking and learning <intro/spike_and_learn>` |
    :doc:`Visualizing the network <intro/visualizing>`

* **Topics:**
    :doc:`Project Overview <topics/overview>` |
    :doc:`Spiking <topics/spiking>` |
    :doc:`Learning <topics/learning>` |
    :doc:`Jeffress Model <topics/jeffress>`

* **Contribute:**
    :doc:`Write Bug Reports <topics/bug_report>` |
    :doc:`Request new features <topics/feature_request>` |
    :doc:`Extend the Code <topics/extending>`


Reference
---------

We have a :doc:`auto-generated reference<reference/index>`.


Licensing
---------

This project is licensed under the `BSD-2-Clause License`_.

The source code is freely available at Github_.

.. _Github: https://github.com/johannesmik/neurons

.. _BSD-2-Clause License: https://github.com/johannesmik/neurons/blob/master/LICENSE

Used open-source projects:

    `python 3 <http://python.org>`_ |
    `pytest <http://pytest.org/>`_ |
    `numpy <http://numpy.org/>`_ |
    `matplotlib <http://matplotlib.org/>`_ |
    `sphinx <http://sphinx-doc.org/>`_


.. toctree::
    :hidden:
    :maxdepth: 3

    contents
    intro/index
    topics/index
    reference/index
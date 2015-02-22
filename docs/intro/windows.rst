Windows Installation Guide
==========================

First install the requirements:

1. `Python3 <http://www.python.org/downloads/>`_

2. `Numpy <http://www.numpy.org/>`_

3. `Matplotlib <http://www.matplotlib.org/>`_

You can also use pre-packaged distributions like `Anaconda <http://continuum.io/downloads#py34>`_.

Next, download and unzip the `newest release <https://github.com/johannesmik/neurons/releases>`_ into a convenient location.

Open a terminal (Shortcut: `Windows + R`). Change into the directory you just unzipped and run the install script

::

    $ cd neurons-0.4
    $ python3 setup.py install --user

.. note::
    This copies all needed data into  in your home directory.

Now you can test if you can import the *neurons* package:

::

    $ python3
    >>> import neurons

This shouldn't lead to any error messages. Congratulations, you are done.

Feedback
--------

The installation does not work? Please write a :doc:`bug report<../topics/bug_report>`.
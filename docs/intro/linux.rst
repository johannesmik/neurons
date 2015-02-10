Ubuntu Installation guide
=========================

This is a step-by-step installation tutorial for *Ubuntu*.

Standard installation
---------------------

First install the requirements:

::

    $ sudo apt-get install python3 python3-numpy python3-matplotlib

Next, download and unzip the `zip-package <https://github.com/johannesmik/neurons/archive/master.zip>`_.

Change into the directory, and run the install script

::

    $ cd neurons-master
    $ python3 setup.py install --user

.. note::
    This copies all needed data into  `.local` in your home directory.

Now you can test if you can import the *neurons* package:

::

    $ python3
    >>> import neurons

This shouldn't lead to any error messages. Congratulations, you are done.

Installation using git
----------------------

First install the requirements:

::

    $ sudo apt-get install python3 python3-numpy python3-matplotlib

Next, clone the git repository:

::

    $ git clone https://github.com/johannesmik/neurons.git

Change into the directory, and run the install script

::

    $ cd neurons
    $ python3 setup.py install --user

.. note::
    This copies all needed data into  `.local` in your home directory..

Now you can test if you can import the *neurons* package:

::

    $ python3
    >>> import neurons

This shouldn't lead to any error messages. Congratulations, you are done.

Feedback
--------

The installation does not work? Please write a :doc:`bug report<../topics/bug_report>`.

__author__ = 'johannes'

import sys

from neurons import get_version
from distutils.core import setup

setup(name='neurons',
      version=get_version(),
      description='A simple simulation tool for neuron models',
      author='Johannes Mikulasch',
      url='http://github.com/johannesmik/neurons',
      license='BSD',
      packages=['neurons']
      )

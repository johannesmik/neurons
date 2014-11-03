#!/usr/bin/python3

'''
Convolution
'''

from __future__ import print_function

import numpy as np
import cProfile
import random
import matplotlib.pyplot as plt


def eps(s, t_membran):
    return np.exp(-s / t_membran)

def small_spiketrain():
    # 1000 timesteps
    # With 10 random spikes
    s = np.array([0]*1000)
    for i in range(10):
        index = random.randint(0, 1000)
        s[index] = 1
    return s

def big_spiketrain():
    # 1.000.000 timesteps Spiketrain
    # With 10.000 random spikes
    s = np.array([0]*1000000)
    for i in range(10000):
        index = random.randint(0, 1000000)
        s[index] = 1
    return s

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    t_current = 0.3
    t_membran = 20

    # Epsilon Function as a vector
    x = np.linspace(0, 200, 200)
    epsilon_vector = eps(x, t_membran)

    # Spiketrain
    s = big_spiketrain()

    # Convolute
    s = (np.convolve(s, epsilon_vector, 'same'))
    cProfile.run('np.convolve(s, epsilon_vector, "same")')

    plt.plot(s, label='Convoluted Spiketrain')
    plt.plot(x, epsilon_vector, label='epsilon vector')
    plt.legend()
    plt.show()

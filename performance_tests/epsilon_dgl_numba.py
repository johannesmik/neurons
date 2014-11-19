__author__ = 'johannes'

'''
Differential equation
'''

import numpy as np
import cProfile
import random
from numba import jit

@jit
def simulate():

    time=1000000
    u = np.zeros(time)

    # ~5% Spikes
    for i in range(time//20):
        index = random.randint(0, time-1)
        u[index] = 1

    t_membran = 20
    C = - 1.0 / t_membran

    # Calculate U
    for t in range(time):

        if t == 0:
            pass
        else:
            u[t] += u[t-1] * (C+1)

    return u

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    u = simulate()

    cProfile.run('u = simulate()')

    plt.plot(u)
    plt.show()
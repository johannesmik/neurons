# Based on the matplot script
# created by Eugene M. Izhikevich, February 25, 2003
# http://www.izhikevich.org/publications/net.m

import numpy as np
import matplotlib.pyplot as plt

# Excitatory and Inhibitatory synapses
Ne, Ni = 800, 200
re = np.random.rand(Ne, 1)
ri = np.random.rand(Ni, 1)

a = np.vstack((0.02*np.ones((Ne, 1)), 0.02 + 0.08 * ri))
b = np.vstack((0.2*np.ones((Ne, 1)), 0.25 - 0.05 * ri))
c = np.vstack((-65 + 15 * re**2, -65 * np.ones((Ni, 1))))
d = np.vstack((8 - 6 * re**2, 2 * np.ones((Ni, 1))))
S = np.hstack((0.5 * np.random.rand(Ne+Ni, Ne), - np.random.rand(Ne+Ni, Ni)))

# Initial values of u and v
v = -65 * np.ones((Ne+Ni, 1))
u = b*v
firings = np.empty((0, 2))

for t in range(1000):
    # Thalamic Input
    I = np.vstack((5 * np.random.randn(Ne, 1), 2 * np.random.randn(Ni, 1)))
    fired = np.nonzero(v>=30)[0] # Indices of spikes
    temp = np.array([t+0*fired, fired])
    firings = np.vstack((firings, temp.T))
    v[fired]=c[fired]
    u[fired]=u[fired]+d[fired]

    I = I + np.sum(S[:,fired], axis=1, keepdims=True)
    v = v + 0.5 * (0.04 * v ** 2 + 5 * v + 140 - u + I)  # step 0.5 ms
    v = v + 0.5 * (0.04 * v ** 2 + 5 * v + 140 - u + I)  # for numerical
    u = u + a * (np.multiply(b, v) - u)                  # stability

plt.scatter(firings[:,0], firings[:,1])
plt.show()

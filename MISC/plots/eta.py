from __future__ import division
from pylab import *

def eta(s, nu_reset, t_membran):

    ret = zeros(s.size)

    ret = - nu_reset * exp(-s/t_membran)

    ret[s < 0] = 0
    ret[s == 0] = 0.9 # Spike

    return ret

def plot_eta(ax, eta_reset, t_membran):
    x = linspace(-100, 500, num=601)
    if t_membran != 0:

        labelstr = r'$\eta_0 = %.1f, \tau_m = %.0f$' % (eta_reset, t_membran)

        ax.plot(x, eta(x, eta_reset, t_membran), label=labelstr)
        ax.legend(prop={'size':12})
        ax.set_xlabel('time in ms')
        ax.set_ylabel('current in mV')
        ax.set_ylim([-1, 1])
        ax.set_xlim([-10, 200])

eta_resets = [0.3, 0.7]
t_membranes = [10, 30]

for i, eta_reset in enumerate(eta_resets):
    for j, t_membrane in enumerate(t_membranes):
        ax = subplot2grid((len(eta_resets), len(t_membranes)), (i, j))
        plot_eta(ax, eta_reset, t_membrane)

suptitle(r'The $\eta(s)$ function for different values of $\eta_0$ and $\tau_m$', fontsize=16)
show()

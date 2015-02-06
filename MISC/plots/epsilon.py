from __future__ import division
from pylab import *


def eps(s, t_current, t_membran):
    return (1/(1-t_current/t_membran))*(np.exp(-s/t_membran) - np.exp(-s/t_current))

def plot_eps(ax, t_current, t_membran):
    x = linspace(0, 500, num=1000)
    print(t_current, t_membran)
    if t_current != t_membrane and t_current != 0 and t_membran != 0:
        ax.plot(x, eps(x, t_current, t_membran), label=r'$\tau_c = %.0f, \tau_m = %.0f$' % (t_current, t_membran))
        #ax.set_title('title')
        ax.legend(prop={'size':12})
    ax.set_xlabel('time in ms')
    ax.set_ylabel('current in mV')
    ax.set_ylim([0, 0.7])

t_currents = [20, 100]
t_membranes = [30, 110]

for i, t_current in enumerate(t_currents):
    for j, t_membrane in enumerate(t_membranes):
        ax = subplot2grid((len(t_currents), len(t_membranes)), (i, j))
        plot_eps(ax, t_current, t_membrane)

suptitle(r'The epsilon function $\epsilon(s)$ for different values of $\tau_c$ and $\tau_m$', fontsize=16)
show()

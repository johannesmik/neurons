from __future__ import division
from pylab import *


def eps(s, t_current, t_membran):
    return (1/(1-t_current/t_membran))*(np.exp(-s/t_membran) - np.exp(-s/t_current))

def plot_eps(ax, t_current, t_membran):
    x = linspace(0, 500, num=1000)
    print t_current, t_membran
    if t_current != t_membrane and t_current != 0 and t_membran != 0:
        ax.plot(x, eps(x, t_current, t_membran), label='C = %d, M = %d' % (t_current, t_membran))
        #ax.set_title('title')
        ax.legend(prop={'size':10})
    ax.set_ylim([0, 0.7])

#t_currents = arange(0, 1, 0.2)
#t_membranes = arange(0, 60, 10)

t_currents = range(0, 60, 10)
t_membranes = range(0, 60, 10)

for i, t_current in enumerate(t_currents):
    for j, t_membrane in enumerate(t_membranes):
        ax = subplot2grid((len(t_currents), len(t_membranes)), (i, j))
        plot_eps(ax, t_current, t_membrane)

suptitle('The epsilon function for different values of t_current and t_membran', fontsize=16)
show()

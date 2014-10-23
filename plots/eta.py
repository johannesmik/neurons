from __future__ import division
from pylab import *

def eta(s, nu_reset, t_membran):

    ret = zeros(s.size)

    ret = - nu_reset * exp(-s/t_membran)

    ret[s < 0] = 0
    ret[s == 0] = 10 # Spike

    return ret

def plot_eta(ax, nu_reset, t_membran):
    x = linspace(-100, 500, num=601)
    print nu_reset, t_membran
    if t_membran != 0:

        labelstr = ''
        #labelstr = 'nu_reset = %d, t_m = %d' % (nu_reset, t_membran)

        ax.plot(x, eta(x, nu_reset, t_membran), label=labelstr)
        #ax.set_title('title')
        ax.legend(prop={'size':10})
    ax.set_xlabel('time in ms')
    ax.set_ylabel('current in mV')
    ax.set_ylim([-1, 1])
    ax.set_xlim([-10, 200])

nu_resets = [0.50]
t_membranes = [0.30]

for i, nu_reset in enumerate(nu_resets):
    for j, t_membrane in enumerate(t_membranes):
        ax = subplot2grid((len(nu_resets), len(t_membranes)), (i, j))
        plot_eta(ax, nu_reset, t_membrane)

suptitle('The eta function', fontsize=16)
show()

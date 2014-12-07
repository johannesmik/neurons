import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

def poisson_homogenous(mu, timesteps):
    """
    Generate a spiketrain for a single neuron
    """
    size = (1, timesteps)
    spiketrain = np.random.poisson(lam=mu, size=size)
    spiketrain = np.array(spiketrain, dtype=bool)

    return spiketrain

def poisson_inhomogenous(mus, timesteps):
    """
    Generate a spiketrain for a single neuron
    using an inhomogenous poisson distribution
    :param mus: List or Tuple of the Lambdas
    """

    if timesteps % len(mus) != 0:
        raise Exception("Cannot divide the %d mu's on the %d timesteps equally." % (len(mus), timesteps))

    spiketrain = np.zeros((1, timesteps), dtype=bool)

    bucketsize = timesteps // len(mus)

    for i, mu in enumerate(mus):
        startindex = i*bucketsize
        spiketrain[0, startindex:startindex+bucketsize] = np.random.poisson(lam=mu, size=bucketsize)

    return spiketrain

def sound(timesteps, midpoint, maximum, variance):
    mu = np.arange(timesteps)
    mu = maximum * np.exp(-((mu - midpoint ) ** 2) / variance ** 2)
    s = poisson_inhomogenous(mu, timesteps)
    return s

def plot_current(current):

    plt.figure()
    plt.title("Neuron currents")
    plt.imshow(current, vmin=np.min(current), vmax=np.max(current),
           cmap=plt.cm.autumn)

def plot_spikes(spiketrain, plot_spikes_only=True):
    """ Gives a scatter plot of the spiketrain. """

    plt.figure()
    plt.title("Spiketrain plot")
    plt.ylabel("Neuron")
    plt.xlabel("Time")

    r, c = np.where(spiketrain)
    plt.scatter(c, r, c='r', s=20, marker="x")

    if not plot_spikes_only:
        r, c = np.where(spiketrain == False)
        plt.scatter(c, r, c='b', s=5, marker="_")


def plot_psth(spiketrain, binsize=20, neuron_indices=None):
    """ Plot a Peri-Stimulus Time Histogram """

    import matplotlib.pyplot as plt

    neurons, timesteps = spiketrain.shape
    bins = range(0, timesteps + binsize, binsize)

    if not neuron_indices:
        print ("Printing all neurons!!")
        neuron_indices = range(neurons)

    n_plots = len(neuron_indices) # Number of subplots

    fig, axes = plt.subplots(nrows=n_plots)
    plt.title("Peri Stimulus Time Histogram")

    for i, axis in enumerate(axes):
        neuron_index = neuron_indices[i]
        # Histogramm
        times = np.where(spiketrain[neuron_index])[0]
        axis.hist(times, bins, histtype='bar', stacked=True, fill=True, facecolor='green', alpha=0.5, zorder=0)

        # Scatter (Spikes)
        y = np.ones(times.shape)
        axis.scatter(times, y, c='r', s=20, marker="x", zorder=1)

        axis.set_title('Neuron %d' % neuron_index)
        axis.set_ylim(bottom=0)
        axis.set_xlim(0, timesteps)

    plt.tight_layout()
    plt.show()

def show_plots():
    plt.show()


class WeightPlotter:
    def __init__(self, save_interval):
        self.save_interval = save_interval
        self.weights = []

    def add(self, weights):
        """ Adds a snapshot of the current weights. """
        self.weights.append(weights.copy())

    def plot_weights(self):

        fig = plt.figure()

        ims = []
        for weight in self.weights:
            im = plt.imshow(weight, vmin=-1, vmax=1, interpolation='none', cmap=cm.coolwarm)
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=20, repeat_delay=3000)
#        ani.save('weightchange.mp4', writer='mencoder', fps=15)
        plt.show()


if __name__ == "__main__":
    spikes = np.zeros((5, 280), dtype=bool)
    for neuron in range(5):
        spikes[neuron] = sound(280, neuron*50, 0.4, 50)

    print(spikes)
    plot_psth(spikes, binsize=25)
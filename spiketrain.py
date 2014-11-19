import numpy as np

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

    # print("Generated inhomogenous spiketrain", spiketrain)

    return spiketrain

def sound(timesteps, midpoint, maximum, variance):
    mu = np.arange(timesteps)
    mu = maximum * np.exp(-((mu - midpoint ) ** 2) / variance ** 2)
    import matplotlib.pyplot as plt
    plt.plot(mu)
    s = poisson_inhomogenous(mu, timesteps)
    return s

def plot(spiketrain):

    import matplotlib.pyplot as plt

    plt.title("Spiketrain plot")
    plt.ylabel("Neuron")
    plt.xlabel("Time")

    r, c = np.where(spiketrain)
    plt.scatter(c, r, c='r', s=20, marker="x")

    r, c = np.where(spiketrain == False)
    plt.scatter(c, r, c='b', s=5, marker="_")
    plt.show()


def plot_psth(spiketrain, binsize=20):
    """ Plot a Peri-Stimulus Time Histogram """

    import matplotlib.pyplot as plt

    neurons, timesteps = spiketrain.shape

    bins=range(0, timesteps + binsize, binsize)
    print ("Bins:", bins)

    fig, axes = plt.subplots(nrows=neurons)

    for i, axis in enumerate(axes):
        # Histogramm
        times = np.where(spiketrain[i])[0]
        axis.hist(times, bins, histtype='bar', stacked=True, fill=True, facecolor='green', alpha=0.5, zorder=0)

        # Scatter (Spikes)
        y = np.ones(times.shape)
        axis.scatter(times, y, c='r', s=20, marker="x", zorder=1)

        axis.set_title('Neuron %d' % i)
        axis.set_ylim(bottom=0)
        axis.set_xlim(0, timesteps)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    s = np.zeros((5, 280), dtype=bool)
    for i in range(5):
        s[i] = sound(280, i*50, 0.4, 50)

    print(s)
    plot_psth(s, binsize=25)
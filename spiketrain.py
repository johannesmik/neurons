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

    print("Generated inhomogenous spiketrain", spiketrain)

    return spiketrain

def sound():
    mus = [0.02, 0.1, 0.4, 0.8, 0.4, 0.1, 0.02]
    timesteps = 280
    s = poisson_inhomogenous(mus, timesteps)
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


if __name__ == "__main__":
    #s = generate(0.1, (3, 100), [0, 1])
    #s = poisson_homogenous(0.1, 100)
    #s = poisson_inhomogenous([0.1, 2, 0.1, 2, 0.1], 500)
    s = np.zeros((5, 280), dtype=bool)
    for i in range(5):
        s[i] = sound()

    print(s)
    plot(s)
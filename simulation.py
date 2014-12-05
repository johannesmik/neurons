__author__ = 'johannes'


def simulate_srm_stdp(spiking_model, learning_model, spiketrain, weights, verbose=False, weightplotter=None):
    neurons, timesteps = spiketrain.shape

    for t in range(timesteps):

        if weightplotter and t % weightplotter.save_interval == 0:
            weightplotter.add(weights)

        spiking_model.simulate(spiketrain, weights, t)

        learning_model.weight_change(spiketrain, weights, t)
        if verbose:
            print("Time step: ", t)
            print("updated spikes", spiketrain)
            print("updated weights", weights)
            print("--------------")

    if verbose:
        print("Finished simulation")
        print("spikes:\n", spiketrain)
        print("weights", weights)


if __name__ == "__main__":
    pass
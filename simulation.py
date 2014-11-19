__author__ = 'johannes'

import numpy as np
import srm
import stdp

def simulate_srm_stdp(spiketrain, weights, **kwargs):
    """
    """
    neurons, timesteps = spiketrain.shape

    threshold = 1.0
    t_current = 0.3                 # Time of current (t_s)
    t_membran = 20                  # Membran-time-constant (t_m)
    nu_reset = 5

    spiking_model = srm.SRM(neurons=neurons, threshold=threshold, t_current=t_current,
                            t_membran=t_membran, nu_reset=nu_reset, verbose=False)

    t_l = 5                         # T_l (learning window size)
    eta = 0.05
    w_in = 0.05
    w_out = 0.05
    tau = 10.0

    learning_model = stdp.STDP(eta=eta, w_in=w_in, w_out=w_out, tau=tau, window_size=t_l, verbose=False)

    for t in range(timesteps):

        print("Timestep: ", t)

        spiking_model.simulate(spiketrain, weights, t)

        learning_model.weight_change(spiketrain, weights, t)
        print("updated weights", weights)

        print("--------------")

    print("Finished simulation")
    print("spikes:\n", spiketrain)
    print("weights", weights)

if __name__ == "__main__":

    s = np.array([[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    w = np.array([[0, 1, 1],
                  [0, 0, 1],
                  [0, 0, 0]], dtype=float)

    simulate_srm_stdp(s, w)
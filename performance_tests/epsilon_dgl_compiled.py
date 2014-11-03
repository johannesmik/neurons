import epsilon_dgl_compiled
import matplotlib.pyplot as plt
import cProfile

if __name__ == "__main__":
    u = epsilon_dgl_compiled.simulate()
    cProfile.run('epsilon_dgl_compiled.simulate()')

    plt.plot(u)
    plt.show()
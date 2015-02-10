__author__ = 'johannes'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm


class PSTH:
    """
    Peri-Stimulus Time Histogram

    .. image:: _images/psth_plot.png
        :alt: PSTH Plot
        :width: 400px

    **Example**

    The image above was created by following code:

    ::

        spikes = np.random.randint(0, 2, size=(2, 200))
        a = PSTH(spikes)
        a.show_plot()
        a.save_plot()
        plt.show()

    """

    def __init__(self, spiketrain, binsize=20):
        """
        Initialize the PSTH plot with a spiketrain

        :param spiketrain: The spiketrain to be plotted
        :type spiketrain: Numpy array
        :param binsize: The size of the bins
        :return: None
        """
        self.spiketrain = spiketrain
        self.binsize = binsize
        self.fig = None

    def show_plot(self, neuron_indices=None):
        """
        Shows the PSTH plot.

        :param neuron_indices: The indices of the neurons to be plotted
        :type neuron_indices: List or Tuple
        """
        neurons, timesteps = self.spiketrain.shape
        bins = range(0, timesteps + self.binsize, self.binsize)

        if not neuron_indices:
            print("Plotting all neurons!!")
            neuron_indices = range(neurons)

        n_plots = len(neuron_indices)  # Number of subplots

        fig, axes = plt.subplots(nrows=n_plots)
        plt.title("Peri Stimulus Time Histogram")

        print(type(axes))

        if not isinstance(axes, np.ndarray):
            axes = [axes]

        for i, axis in enumerate(axes):
            neuron_index = neuron_indices[i]
            # Histogramm
            times = np.where(self.spiketrain[neuron_index])[0]
            axis.hist(times, bins, histtype='bar', stacked=True, fill=True, facecolor='green', alpha=0.5, zorder=0)

            # Scatter (Spikes)
            y = np.ones(times.shape)
            axis.scatter(times, y, c='r', s=20, marker="x", zorder=1)

            axis.set_title('Neuron %d' % neuron_index)
            axis.set_ylim(bottom=0)
            axis.set_xlim(0, timesteps)

        plt.tight_layout()
        plt.show(block=False)

        self.fig = fig

    def save_plot(self, filename=None):
        """
        Saves the plot.

        :param filename: Name of the file. Default: 'plot.png'
        :type filename: String

        """

        if not self.fig:
            self.show_plot()

        if not filename:
            filename = 'plot.png'

        plt.figure(self.fig.number)
        plt.savefig(filename)
        print('saved psth plot as %s' % filename)


class HeatmapAnimation:
    def __init__(self, fps=30):
        self.values = []
        self.ani = None
        self.fps = fps

        self.vmin = -1
        self.vmax = 1

    def show_animation(self):
        """
        Shows the Animation
        :return: pyplot animation object
        """

        fig = plt.figure()

        ims = []
        for value in self.values:
            im = plt.imshow(value, vmin=self.vmin, vmax=self.vmax, interpolation='none', cmap=cm.coolwarm)
            ims.append([im])

        self.ani = animation.ArtistAnimation(fig, ims, interval=1000 / self.fps, repeat_delay=3000)
        plt.show(block=False)

        return self.ani

    def save_animation(self, filename=None):
        """
        Saves the animation
        :param filename: Name of the file. Default: 'heatmap_plot.mp4'
        :type filename: String
        :return:
        """

        if not filename:
            filename = 'heatmap_plot.mp4'

        if not self.ani:
            self.show_animation()

        self.ani.save(filename, writer='mencoder', fps=30)

        print("Saved heatmap animation as %s " % filename)


class WeightHeatmapAnimation(HeatmapAnimation):
    """
    Show an animation of the weights (as a heatmap).

    :Example:

    ::

        w = WeightHeatmapAnimation()
        for i in range(300):
            w.add(np.array([[i/300, 1], [1, 3]]))
        w.plot_weights()
        plt.show()

    """

    def __init__(self, fps=30):
        """
        Initialize the Weight Animation

        :param fps: How many frames-per-second the animation should be played. Default: 30
        :return: None
        """
        HeatmapAnimation.__init__(self, fps)

    def add(self, weights):
        """
        Add weights to the end of animation.

        :param weights: The weight matrix to be shown.
        :type weights: Symmetric Numpy array
        :return: None
        """
        self.values.append(weights.copy())


class CurrentsHeatmapAnimation(HeatmapAnimation):
    """
    :Example:

    ::

    cha = CurrentsHeatmapAnimation()
    for i in range(300):
        cha.add(np.random.randn(1, 20))
    cha.show_animation()
    plt.show()

    """

    def __init__(self, fps=30):
        """
        Initialize the Currents Animation
        :param fps: How many frames-per-second the animation should be played. Default: 30
        :return: None
        """
        HeatmapAnimation.__init__(self, fps)

    def add(self, currents):
        """
        Adds currents to the end of the animation
        :param currents:
        :return: None
        """
        if len(currents.shape) == 1:
            width = currents.shape[0]
            currents = currents.reshape((1, width))
        self.values.append(currents.copy())

class CurrentPlot:

    def __init__(self, neurons):
        self.values = np.empty((0, neurons))
        self.fig = None

    def add(self, currents):
        if len(currents.shape) == 1:
            width = currents.shape[0]
            currents = currents.reshape((1, width))

        self.values = np.append(self.values, currents.copy(), axis=0)

    def show_plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.values)

        plt.show(block=False)
        self.fig = fig

if __name__ == '__main__':
    c = CurrentPlot(3)
    c.add(np.array([1, 5, 4]))
    c.add(np.array([2, 4, 4]))
    c.show_plot()
    plt.show()
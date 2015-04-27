"""
This module contains functions to create nice plots.
"""

__author__ = 'johannes'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
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


        if not isinstance(axes, np.ndarray):
            axes = [axes]

        for i, axis in enumerate(axes):
            neuron_index = neuron_indices[i]
            # Histogramm
            times = np.where(self.spiketrain[neuron_index])[0]
            if len(times) >= 1:
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

class SpikePlot:
    """
    Most Simple SpikePlot

    .. image:: _images/spike_plot.png
        :alt: SpikePlot
        :width: 400px

    **Example**

    The image above was created by following code:

    ::

        spikes = np.random.randint(0, 2, size=(10, 200))
        a = SpikePlot(spikes)
        a.show_plot(neuron_indices=[1,2,6])
        a.save_plot()
        plt.show()

    """
    def __init__(self, spiketrain):
        """
        Initialize the Spike plot with a spiketrain

        :param spiketrain: The spiketrain to be plotted
        :type spiketrain: Numpy array
        :return: None
        """
        self.spiketrain = spiketrain
        self.fig = None

    def show_plot(self, neuron_indices=None):
        """
        Shows the Spike plot.

        :param neuron_indices: The indices of the neurons to be plotted
        :type neuron_indices: List or Tuple
        """
        neurons, timesteps = self.spiketrain.shape

        if not neuron_indices:
            print("Plotting all neurons!!")
            neuron_indices = range(neurons)

        n_plots = len(neuron_indices)  # Number of subplots

        fig = plt.figure()
        plt.title("SpikePlot")


        for i in range(len(neuron_indices)):
            print(i)
            neuron_index = neuron_indices[i]

            times = np.where(self.spiketrain[neuron_index])[0]

            # Scatter (Spikes)
            y = i * np.ones(times.shape)
            plt.scatter(times, y, c='r', s=40, marker="|", zorder=1)

            plt.xlim(0, timesteps)

        plt.ylabel('Neuron')
        plt.yticks(range(len(neuron_indices)), neuron_indices)

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

class HintonPlot(object):
    """
        Hinton diagrams are useful for visualizing the values of a 2D array (e.g.
        a weight matrix): Positive and negative values are represented by white and
        black squares, respectively, and the size of each square represents the
        magnitude of each value.

        Initial idea from David Warde-Farley on the SciPy Cookbook
    """

    def __init__(self, matrix, max_weight=None, ax=None):
        """
            Draw Hinton diagram for visualizing a weight matrix.

            http://matplotlib.org/examples/specialty_plots/hinton_demo.html
        """
        ax = ax if ax is not None else plt.gca()

        if not max_weight:
            max_weight = 2**np.ceil(np.log(np.abs(matrix).max())/np.log(2))

        ax.patch.set_facecolor('gray')
        ax.set_aspect('equal', 'box')
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

        for (x,y),w in np.ndenumerate(matrix):
            color = 'white' if w > 0 else 'black'
            size = np.sqrt(np.abs(w))
            rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                                 facecolor=color, edgecolor=color)
            ax.add_patch(rect)

        ax.autoscale_view()
        ax.invert_yaxis()
        plt.show(block=False)


class Histogram3DPlot(object):

    """
    .. image:: _images/threed_histogram_plot.png
        :alt: Histogram 3D Plot
        :width: 400px

    **Example**

    The image above was created by following code:

    ::

        hist = Histogram3DPlot(np.random.random((5, 5)))
        plt.show()
    """

    def __init__(self, matrix, xlimits=None, ylimits=None, width_factor=0.9, alpha=1.0, color='#00ceaa', ax=None):

        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')

        xlimits = xlimits if xlimits is not None else (0, 1)
        ylimits = ylimits if ylimits is not None else (0, 1)
        xsize, ysize = matrix.shape

        xpos, ypos = np.meshgrid(np.linspace(xlimits[0], xlimits[1], xsize), np.linspace(ylimits[0], ylimits[1], ysize))
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros(xsize * ysize)

        dx = width_factor * (abs(xlimits[0] - xlimits[1]) / xsize) * np.ones(xsize * ysize)
        dy = width_factor * (abs(ylimits[0] - ylimits[1]) / ysize) * np.ones(xsize * ysize)
        dz = matrix.flatten()

        ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color=color, alpha=alpha)
        plt.show(block=False)

class WeightHeatmapAnimation(HeatmapAnimation):
    """
    Show an animation of the weights (as a heatmap).

    .. raw:: html

        <iframe width="420" height="315" src="https://www.youtube.com/embed/R35L-m5Gxuw" frameborder="0" allowfullscreen></iframe>

    **Example**

    The video above was created by following code:

    ::

        w = WeightHeatmapAnimation()
        for i in range(300):
            w.add(np.array([[i/300, 1], [1, (300 - i) / 300]]))
        w.show_animation()
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
        :type weights: Numpy array
        :return: None
        """
        self.values.append(weights.copy())


class CurrentsHeatmapAnimation(HeatmapAnimation):
    """
    .. raw:: html

        <iframe width="560" height="150" src="https://www.youtube.com/embed/3vUZG_3eQ_I" frameborder="0" allowfullscreen></iframe>

    **Example**

    The video above was created by following code:

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
    """
    A plot of the current of neurons.

    Create the plot by adding the values of the currents at one timestep with `CurrentPlot,add(values)`.

    .. image:: _images/current_plot.png
        :alt: PSTH Plot
        :width: 400px

    **Example**

    The image above was created by following code:

    ::

        c = CurrentPlot(3)
        c.add(np.array([1, 5, 4]))
        c.add(np.array([2, 4, 4]))
        c.add(np.array([3, 5, 4]))
        c.add(np.array([4, 3, 4]))
        c.show_plot()
        plt.show()

    """

    def __init__(self, neurons):
        self.values = np.empty((0, neurons))
        self.fig = None

    def add(self, currents):
        if len(currents.shape) == 1:
            width = currents.shape[0]
            currents = currents.reshape((1, width))

        self.values = np.append(self.values, currents.copy(), axis=0)

    def show_plot(self):
        datapoints, neurons = self.values.shape
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.values)
        ax.legend(['Neuron % i' % i for i in range(neurons)])

        plt.show(block=False)
        self.fig = fig

    def save_plot(self, filename='plot.png'):
        """
        Saves the plot.

        :param filename: Name of the file. Default: 'plot.png'
        :type filename: String

        """

        if not self.fig:
            self.show_plot()

        plt.figure(self.fig.number)
        plt.savefig(filename)
        print('saved current plot as %s' % filename)

def show():
    plt.show()

if __name__ == '__main__':
    pass
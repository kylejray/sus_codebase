import numpy as np
from math import log
import matplotlib.pyplot as plt
from matplotlib import animation, transforms

from itertools import product

from .infospace_1D import plot_extrema_positions


# --------- Basic Plots ---------

def plot_running_quantity(quantity, sim=None,
                          start_plot_time=None, end_plot_time=None,
                          substage_times=None, substage_labels='numbered',
                          initial_time=None, final_time=None, ylim=None,
                          xlabel='time', ylabel=None, title=None, alpha=1,
                          fig=None, ax=None, figsize=(6, 5), savename=None):

    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            ax = fig.subplots()

    if title is not None:
        ax.set_title(title)

    nsteps = quantity.shape[-1] - 1

    if initial_time is None:
        initial_time = 0.

    if final_time is None:
        final_time = sim.system.protocol.total_t if sim is not None else 1.

    if start_plot_time is None:
        start_plot_time = initial_time

    if end_plot_time is None:
        end_plot_time = final_time

    if substage_times is None and sim is not None:
        substage_times = sim.system.protocol.substage_times

    times = np.linspace(initial_time, final_time, nsteps + 1)

    total_time = final_time - initial_time

    start = round((start_plot_time - initial_time) / total_time * nsteps)
    end = round((end_plot_time - initial_time) / total_time * nsteps)

    if quantity.ndim > 1:
        for trial in range(quantity.shape[0]):
            ax.plot(times[start:end], quantity[trial, start:end], alpha=alpha)
    else:
        ax.plot(times[start:end], quantity[start:end], alpha=alpha)

    if ylim is None:
        ylim = ax.get_ylim()

    if substage_times is not None:
        ax.vlines(substage_times, *ylim, linestyle='dashed', alpha=0.5)

        if substage_labels is not None:

            yval = 0.9 * (ylim[1] - ylim[0]) + ylim[0]

            if substage_labels is 'numbered':

                substage_labels = range(len(substage_times) - 1)

            for substage, substage_label in enumerate(substage_labels):

                xval = (substage_times[substage] \
                         + substage_times[substage + 1]) / 2

                if start_plot_time < xval and xval < end_plot_time:
                    ax.text(xval, yval, f'{substage_label}')

    ax.set_xlim(start_plot_time, end_plot_time)
    ax.set_ylim(ylim)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if savename is not None:
        fig.savefig(savename)
        plt.close(fig)

    return ax


def plot_pos_over_time(sim, trial_indices,
                       start_plot_time=None, end_plot_time=None,
                       substage_times=None, substage_labels='numbered',
                       initial_time=None, final_time=None, ylim=None,
                       xlabel='time', ylabel = 'position',
                       title='position over time', alpha=1,
                       fig=None, ax=None, figsize=(6,5), savename=None):

    quantity = sim.output.all_state['states'][trial_indices, :, 0]

    if substage_times is None \
            and hasattr(sim.system.protocol, 'substage_times'):
        substage_times = sim.system.protocol.substage_times

    return plot_running_quantity(quantity, sim,
                                 start_plot_time, end_plot_time,
                                 substage_times, substage_labels,
                                 initial_time, final_time, ylim,
                                 xlabel, ylabel, title, alpha,
                                 fig, ax, figsize, savename)


def plot_dist_over_time(pos_dists):
    pass


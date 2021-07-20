from math import inf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def pot_at_time(system, t, xlim, ax=None, ylim=None,
                to_label_yvalues=True,
                label_axes=True, add_guidelines=True, **kwargs):

    if ax is None:
        fig, ax = plt.subplots()

    if label_axes:
        ax.set_xlabel('x')
        ax.set_ylabel('V(x)')

    X = np.linspace(*xlim, 101)
    Y = [system.potential(x, t) for x in X]

    if ylim:
        ax.set_ylim(ylim)
    else:
        ylow = min(min(Y), 0) if add_guidelines else min(Y)
        yhigh = max(max(Y), 0) if add_guidelines else max(Y)
        ydelta = yhigh - ylow
        ylim = ylow - ydelta/10, yhigh + ydelta/10

    if add_guidelines:
        ax.hlines(0, *xlim, linestyle='dashed', alpha=0.25)
        ax.vlines(0, *ylim, linestyle='dashed', alpha=0.25)

    if not to_label_yvalues:
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

    ax.set_ylim(ylim)

    return ax.plot(X, Y, **kwargs)[0]


def params_onto_ax(params, ax=None, substage_times=None, tlim=None,
                   set_xticks=True, set_yticks=True, savename=None,
                   Vlim=None, to_plot_legend=True, param_labels=None,
                   xticks=None, xticklabels=None, legend_loc='best'):

    '''params is a list of tuples of param names and functions over time'''

    if ax is None:
        fig, ax = plt.subplots()

    if param_labels is not None:
        old_params = params
        params = {}
        for pfunc, label in zip(old_params.values(), param_labels):
            params[label] = pfunc

    if tlim is None:
        if substage_times is not None:
            tlim = substage_times[0], substage_times[-1]
        else:
            tlim = (0., 1.)

    T = np.linspace(*tlim, 200)

    param_plots = []
    for plabel, pfunc in params.items():

        param_list = [pfunc(t) for t in T]
        param_scale = max(map(abs, param_list))

        param_scale = param_scale if param_scale !=0 else 1

        if param_scale == 1:
            param_plot, = ax.plot(T, param_list, label=plabel)
        else:
            param_plot, = ax.plot(T, param_list,
                                  label=plabel + f' ({round(param_scale, 2)})')

        param_plots.append(param_plot)

    if xticks:
        ax.set_xticks(xticks)
    elif not set_xticks:
        ax.set_xticks([])

    if xticklabels:
        ax.set_xticklabels(xticklabels)

    if not set_yticks:
        ax.set_yticks([])

    ax.set_xlabel('t')

    if Vlim:
        ax.set_ylim(Vlim)
    else:
        Vlim = ax.get_ylim()

    if substage_times:
        ax.vlines(substage_times, *Vlim, linestyle='dashed', alpha=0.5);

    if to_plot_legend:
        ax.legend(loc=legend_loc)

    if savename:
        ax.get_figure().savefig(savename)
        plt.close(ax.get_figure())

    return param_plots


def pot_at_times(system, times, xlim, axes=None, Vlim=None,
                 to_label_axes=True, to_label_yvalues=True,
                 to_add_guidelines=True, savename=None,
                 to_label_figure=True, axes_labels=None,
                 figure_label=None, **kwargs):
    """Plot potential at specific times."""

    if axes is None:
        fig, ax = plt.subplots()
        axes = [ax]*len(times)

    # if colors is None:
    #     cmap = matplotlib.cm.get_cmap('gist_rainbow')
    #     colors = cmap(np.linspace(0, 1, len(times)))

    if Vlim is None:
        ymin = inf
        ymax = -inf

    if not axes_labels:
        axes_labels = [f't: {t:.3}' for t in times]

    for time, ax, ax_label in zip(times, axes, axes_labels):

        pot_at_time(system, time, xlim, ax=ax, label_axes=False,
                    add_guidelines=False)

        if not to_label_yvalues:
            ax.yaxis.set_ticks([])

        if to_label_axes:
            ax.set_title(ax_label)

        if Vlim is None:
            ax_ymin, ax_ymax = ax.get_ylim()
            if ax_ymin < ymin:
                ymin = ax_ymin
            if ax_ymax > ymax:
                ymax = ax_ymax

    ylim = Vlim if Vlim is not None else (ymin, ymax)

    for ax in axes:
        ax.set_ylim(ylim)
        if to_add_guidelines:
            ax.hlines(0, *xlim, linestyle='dashed', alpha=0.5)
            ax.vlines(0, *ylim, linestyle='dashed', alpha=0.5)
        ax.set_xlim(xlim)

    if to_label_figure:
        if not figure_label:
            figure_label = 'Potential'
        axes[0].get_figure().suptitle(figure_label)

    axes[0].get_figure().tight_layout()

    if savename is not None:
        axes[0].get_figure().savefig(savename)
        plt.close(axes[0].get_figure())


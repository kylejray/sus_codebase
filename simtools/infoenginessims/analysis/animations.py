import numpy as np
from math import ceil, inf
from numpy import linspace, s_, full
from numpy import abs as np_abs
import matplotlib.pyplot as plt
from matplotlib import animation, transforms
from matplotlib.ticker import EngFormatter

from ..gregtools import get_rough_minima_from_array
from .protocols import params_onto_ax, pot_at_time
from .infospace_1D import plot_extrema_positions, plot_extrema_potentials
from .state_dists_1D import state_hist_onto_ax


# -------- Fundamental Animation Functions

def animate_features(features, tf=None, sim=None,
                     nframes=20, an_time=5_000, t0=0,
                     fig=None, axes=None, figsize=None):
    """Animate features of a simulation.

    Example features:
     - protocol params
     - potential extrema
     - potential
     - potential + position histogram
     - potential + trajectories
     - potential + position hist + trajectories
     - 2D state hist
     """

    nfeatures = len(features)

    if axes is None:
        fig, axes = get_fig_axes(nfeatures, figsize)
    else:
        fig = axes[0].get_figure()

    if tf is None:
        tf = sim.system.protocol.total_t

    tvals = linspace(t0, tf, nframes)

    an_dt_per_frame = an_time / (nframes)

    for feature, ax in zip(features, axes):
        feature.setup(ax, tvals)

    def update(frame):

        time = tvals[frame]

        outputs = []
        for feature in features:
            output = feature.update(time)
            outputs.append(output)

        return outputs

    # fig.tight_layout()

    plt.close(fig)

    return animation.FuncAnimation(fig, update, frames=nframes,
                                   interval=an_dt_per_frame)


def animate_simulation(feature_classes, feature_classes_args, sim,
                       nframes=20, an_time=5_000, t0=0, tf=None,
                       fig=None, axes=None, figsize=None):
    """Animate a simulation from feature classes."""

    if tf is None:
        tf = sim.system.protocol.total_t

    features = []
    for feature_class, feature_class_args, \
            in zip(feature_classes, feature_classes_args):

        if isinstance(feature_class_args, dict):
            feature = feature_class(sim, **feature_class_args)
        else:
            feature = feature_class(sim, *feature_class_args)

        features.append(feature)

    return animate_features(features, tf, sim, nframes, an_time, t0,
                            fig, axes, figsize)


# -------- Helper Functions

def get_fig_axes(nfeatures, figsize=None):

    if figsize is None:

        ncols = ceil(nfeatures / 2)
        nrows = 1 if nfeatures < 2 else 2

        figsize = (5 * ncols, 4 * nrows)

    if nfeatures == 1:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        axes = [ax]

    elif nfeatures == 2:
        fig, axes = plt.subplots(2, 1, figsize=figsize)

    elif nfeatures == 3:
        fig = plt.figure(figsize=figsize)
        plt.subplot2grid((2, 2), (0, 0))
        plt.subplot2grid((2, 2), (1, 0))
        plt.subplot2grid((2, 2), (0, 1))
        axes = fig.get_axes()

    elif nfeatures == 4:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

    elif nfeatures == 5:
        fig = plt.figure(figsize=figsize)
        plt.subplot2grid((2, 3), (0, 0))
        plt.subplot2grid((2, 3), (1, 0))
        plt.subplot2grid((2, 3), (0, 1))
        plt.subplot2grid((2, 3), (1, 1))
        plt.subplot2grid((2, 3), (0, 2))
        axes = fig.get_axes()

    elif nfeatures == 6:
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

    else:
        raise ValueError(f"Received {nfeatures} feature_classes,"
                         + "can only handle 1-6.")

    return fig, axes


def get_Vlim_from_potential(potential, tvals, xlim, nxsteps=10_000):

    xvals = linspace(*xlim, nxsteps)

    Varray = np.array([[potential(x, t) for t in tvals] for x in xvals])

    Vmin = inf
    Vmax = -inf
    for tstep, t in enumerate(tvals):

        minima = get_rough_minima_from_array(Varray[:, tstep])
        maxima = get_rough_minima_from_array(-Varray[:, tstep])

        if len(minima) > 0:
            Vmin_at_t = Varray[minima, tstep].min()
            Vmin = min(Vmin_at_t, Vmin)

        if len(maxima) > 0:
            Vmax_at_t = Varray[maxima, tstep].max()
            Vmax = max(Vmax_at_t, Vmax)

    if Vmin == inf:
        Vmin = Varray.min()
    if Vmax == -inf:
        Vmax = Varray.max()

    delta_V = Vmax - Vmin

    Vmax += 0.2 * delta_V
    Vmin -= 0.2 * delta_V

    return Vmin, Vmax


# -------- Feature Classes

class Feature:

    def setup(self, ax, tvals):
        pass

    def update(self, time):
        pass


class Potential(Feature):

    def __init__(self, sim,
                 xlim=None, Vlim=None, nxsteps=200,
                 potential_color='blue',
                 show_x_ticks=True, show_V_ticks=True):

        self.sim = sim
        self.xlim = xlim
        self.Vlim = Vlim
        self.nxsteps = nxsteps
        self.potential_color = potential_color
        self.show_x_ticks = show_x_ticks
        self.show_V_ticks = show_V_ticks

    def setup(self, ax, tvals):

        sim = self.sim
        xlim = self.xlim
        Vlim = self.Vlim
        nxsteps = self.nxsteps
        potential_color = self.potential_color
        show_x_ticks = self.show_x_ticks
        show_V_ticks = self.show_V_ticks

        potential = sim.system.potential

        if xlim is None:
            xbins = sim.initial_dist.pos_hist[1]
            xlim = xbins[0], xbins[-1]

        xvals = linspace(*xlim, nxsteps)

        if Vlim is None and tvals is not None:
            Vlim = get_Vlim_from_potential(potential, tvals, xlim)

        line = ax.plot([], [], color=potential_color)[0]

        ax.yaxis.label.set_color(potential_color)
        ax.tick_params(axis='y', colors=potential_color)

        ax.set_xlim(xlim)
        ax.set_ylim(Vlim)

        ax.set_xlabel('x')
        ax.set_ylabel('V(x)')

        if not show_x_ticks:
            ax.xaxis.set_ticks([])
        if not show_V_ticks:
            ax.yaxis.set_ticks([])

        self.potential = potential
        self.xvals = xvals
        self.line = line

    def update(self, time):

        potential = self.potential
        xvals = self.xvals
        line = self.line

        line.set_data(xvals, [potential(x, time) for x in xvals])

        return line


class PotentialPoshist:

    def __init__(self, sim,
                 xlim=None, Vlim=None, nxsteps=200,
                 poshist_ymax=None,
                 potential_color='blue', poshist_color='LimeGreen',
                 show_x_ticks=True, show_V_ticks=True, show_Pr_ticks=True):

        self.sim = sim
        self.xlim = xlim
        self.Vlim = Vlim
        self.nxsteps = nxsteps
        self.poshist_ymax = poshist_ymax
        self.potential_color = potential_color
        self.poshist_color = poshist_color
        self.show_x_ticks = show_x_ticks
        self.show_V_ticks = show_V_ticks
        self.show_Pr_ticks = show_Pr_ticks

    def setup(self, ax, tvals):

        sim = self.sim
        xlim = self.xlim
        Vlim = self.Vlim
        nxsteps = self.nxsteps
        poshist_ymax = self.poshist_ymax
        potential_color = self.potential_color
        poshist_color = self.poshist_color
        show_x_ticks = self.show_x_ticks
        show_V_ticks = self.show_V_ticks
        show_Pr_ticks = self.show_Pr_ticks

        ax_potential = ax
        ax_poshist = plt.twinx(ax_potential)

        potential = sim.system.potential

        all_statedists = sim.output.all_state_dists
        statehists = all_statedists['hists']
        poshist_indices = np.array(all_statedists['step_indices'])

        nsteps = sim.nsteps
        initial_t = 0
        final_t = sim.system.protocol.total_t

        dt = (final_t - initial_t) / nsteps

        poshist_times = dt * poshist_indices

        poshist_bins = statehists[0][1][0]
        poshist_xvals = (poshist_bins[0:-1] + poshist_bins[:-1]) / 2

        poshists = [(hist[0].sum(axis=1), poshist_bins) for hist in statehists]

        if xlim is None:

            xbins = sim.initial_dist.pos_hist[1]

            xmin = min(xbins[0], poshist_bins[0])
            xmax = max(xbins[-1], poshist_bins[-1])

            xlim = xmin, xmax

        xvals = linspace(*xlim, nxsteps)

        if Vlim is None and tvals is not None:
            Vlim = get_Vlim_from_potential(potential, tvals, xlim)

        if poshist_ymax is None:
            poshist_ymax = 0
            for poshist in poshists:
                poshist_ymax = max(poshist_ymax, poshist[0].max())
        poshist_ymax *= 1.05

        potential_line = ax_potential.plot([], [], color=potential_color)[0]

        ax_potential.set_xlim(xlim)
        ax_potential.set_ylim(Vlim)
        ax_poshist.set_ylim(0, poshist_ymax)

        ax_potential.yaxis.label.set_color(potential_color)
        ax_potential.tick_params(axis='y', colors=potential_color)
        ax_poshist.yaxis.label.set_color(poshist_color)
        ax_poshist.tick_params(axis='y', colors=poshist_color)

        ax_poshist.yaxis.set_major_formatter(EngFormatter())

        ax_potential.set_xlabel('x')
        ax_potential.set_ylabel('V(x)')
        ax_poshist.set_ylabel('Counts')

        if not show_x_ticks:
            ax_potential.xaxis.set_ticks([])
        if not show_V_ticks:
            ax_potential.yaxis.set_ticks([])
        if not show_Pr_ticks:
            ax_poshist.yaxis.set_ticks([])

        self.potential = potential
        self.xvals = xvals
        self.potential_line = potential_line
        self.ax_poshist = ax_poshist
        self.poshists = poshists
        self.poshist_xvals = poshist_xvals
        self.poshist_times = poshist_times

    def update(self, time):

        potential = self.potential
        xvals = self.xvals
        potential_line = self.potential_line
        ax_poshist = self.ax_poshist
        poshists = self.poshists
        poshist_xvals = self.poshist_xvals
        poshist_times = self.poshist_times
        poshist_color = self.poshist_color

        potential_line.set_data(xvals, [potential(x, time) for x in xvals])

        ax_poshist.collections.clear()

        poshist = poshists[np_abs(poshist_times - time).argmin()]
        poshist_vals = poshist[0]

        ax_poshist.fill_between(poshist_xvals, poshist_vals, step='mid',
                                color=poshist_color, alpha=0.7,
                                where=(poshist_vals > 0))

        return potential_line


class PosExtrema:

    def __init__(self, sim,
                 infospace, xlim=None, to_show_legend=False,
                 legend_loc='lower left', time_line_color='blue',
                 **kwargs):

        self.sim = sim
        self.infospace = infospace
        self.xlim = xlim
        self.to_show_legend = to_show_legend
        self.legend_loc = legend_loc
        self.time_line_color = time_line_color
        self.kwargs = kwargs

    def setup(self, ax, tvals):

        sim = self.sim
        infospace = self.infospace
        xlim = self.xlim
        to_show_legend = self.to_show_legend
        legend_loc = self.legend_loc
        time_line_color = self.time_line_color
        kwargs = self.kwargs

        t0, tf = tvals[0], tvals[-1]

        substage_times = sim.system.protocol.substage_times

        vline = ax.plot([], [], lw=2, color=time_line_color)[0]
        plot_extrema_positions(infospace, 0.1,
                               start_plot_time=t0, end_plot_time=tf,
                               substage_times=substage_times, ax=ax,
                               to_show_legend=to_show_legend, **kwargs)

        if to_show_legend:
            ax.legend(loc=legend_loc)

        ymin, ymax = ax.get_ylim()

        yrange = np.linspace(ymin * 0.95, ymax * 0.95, 100)

        ax.set_yticks([])
        ax.set_xticks(substage_times)

        self.vline = vline
        self.yrange = yrange

    def update(self, time):

        vline = self.vline
        yrange = self.yrange

        vline.set_data(full(100, time), yrange)

        return vline


class PotExtrema:

    def __init__(self, sim,
                 infospace, Vlim=None, to_show_legend=False,
                 legend_loc='lower left', time_line_color='blue',
                 **kwargs):

        self.sim = sim
        self.infospace = infospace
        # self.Vlim = Vlim
        self.to_show_legend = to_show_legend
        self.legend_loc = legend_loc
        self.time_line_color = time_line_color
        self.kwargs = kwargs

    def setup(self, ax, tvals):

        sim = self.sim
        infospace = self.infospace
        # Vlim = self.Vlim
        to_show_legend = self.to_show_legend
        legend_loc = self.legend_loc
        time_line_color = self.time_line_color
        kwargs = self.kwargs

        t0, tf = tvals[0], tvals[-1]

        substage_times = sim.system.protocol.substage_times

        vline = ax.plot([], [], lw=2, color=time_line_color)[0]
        plot_extrema_potentials(infospace, 0.1,
                               start_plot_time=t0, end_plot_time=tf,
                               substage_times=substage_times, ax=ax,
                               to_show_legend=to_show_legend, **kwargs)

        if to_show_legend:
            ax.legend(loc=legend_loc)

        ymin, ymax = ax.get_ylim()

        yrange = np.linspace(ymin * 0.95, ymax * 0.95, 100)

        ax.set_yticks([])
        ax.set_xticks(substage_times)

        self.vline = vline
        self.yrange = yrange

    def update(self, time):

        vline = self.vline
        yrange = self.yrange

        vline.set_data(full(100, time), yrange)

        return vline


class State2DHist:

    def __init__(self, sim,
                 xlim=None, vlim=None, log=True, cmap_name=None):

        self.sim = sim
        self.xlim = xlim
        self.vlim = vlim
        self.log = log
        self.cmap_name = cmap_name

    def setup(self, ax, tvals):

        sim = self.sim
        xlim = self.xlim
        vlim = self.vlim
        log = self.log
        cmap_name = self.cmap_name

        all_state_dists = sim.output.all_state_dists
        statehists = all_state_dists['hists']
        hist_indices = np.array(all_state_dists['step_indices'])

        nsteps = sim.nsteps
        initial_t = 0
        final_t = sim.system.protocol.total_t

        dt = (final_t - initial_t) / nsteps

        hist_times = dt * hist_indices

        # title = ax.set_title('', ha='left')
        im = state_hist_onto_ax(statehists[0], log=log, cmap_name=cmap_name,
                                ax=ax, plot_color_bar=False)

        ax.set_xlim(xlim)
        ax.set_ylim(vlim)

        ax.set_xlabel('x')
        ax.set_ylabel('v', labelpad=0)

        self.statehists = statehists
        self.hist_times = hist_times
        # self.title = title
        self.im = im

    def update(self, time):

        statehists = self.statehists
        hist_times = self.hist_times
        # title = self.title
        im = self.im

        hist = statehists[np_abs(hist_times - time).argmin()]

        im.set_array(hist[0].transpose()[::-1, ::])
        # title.set_text(f't: {time:.4}')

        # return im, title
        return im



# -------- OLD CODE BELOW, STILL USED THOUGH --------- #

def animate_protocol(system, xlim, nxsteps=200,
                     nframes=20, an_time=5_000,
                     Vlim=None, tlim=None, figsize=(6, 10)):

    pot_color = 'blue'

    protocol = system.protocol
    interval = int(round(an_time / nframes))

    fig, (ax_params, ax_pot) = plt.subplots(2, 1, figsize=figsize);

    X = np.linspace(*xlim, nxsteps)
    if tlim is None:
        tlim = 0, protocol.total_t

    vline, = ax_params.plot([], [], lw=2, color=pot_color)
    params_onto_ax(protocol.params, ax_params, protocol.substage_times, tlim)

    ax_params.hlines(0, *tlim, linestyle='dashed', alpha=0.5)

    ax_params.legend(loc='lower right')
    ax_params.set_yticks([])

    pot_line = pot_at_time(system, 0, xlim, ax_pot, Vlim)
    pot_line.set_color(pot_color)

    ax_pot.yaxis.label.set_color(pot_color)
    ax_pot.tick_params(axis='y', colors=pot_color)

    def update(frame):
        t = frame / nframes * protocol.total_t

        vline.set_data([t] * 100, np.linspace(-1, 1, 100))
        pot_line.set_data(X, [system.potential(x, t) for x in X])

        return pot_line

    plt.close()

    return animation.FuncAnimation(fig, update, frames=nframes,
                                   interval=interval)


def animate_state_hists(hists, tlim=(0, 1), an_time=10_000, log=False,
                        cmap_name=None):
    """Assumes hists are evenly spread in time."""

    nframes = len(hists)
    interval = round(an_time / nframes)

    fig, ax = plt.subplots()

    vmax = max(hist[0].max() for hist in hists)

    title = fig.suptitle('', ha='left')
    im = state_hist_onto_ax(hists[0], log=log, cmap_name=cmap_name,
                            fig=fig, ax=ax, vmax=vmax)

    def update(frame):

        t = frame / (nframes - 1) * (tlim[1] - tlim[0]) + tlim[0]

        im.set_array(hists[frame][0].transpose()[::-1,::])
        title.set_text(f't: {t:.4}')

        return im

    plt.close()

    return animation.FuncAnimation(fig, update, frames=nframes,
                                   interval=interval)


def OLD_animate_simulation(system, xhists, xlim=None, nxsteps=200,
                       an_time=5_000,
                       Vlim=None,
                       xhist_bins=None, xhist_range=None,
                       xhist_ymax=None,
                       figsize=(6, 5)):

    pot_color = 'blue'
    xhist_color = 'LimeGreen'

    protocol = system.protocol
    # nsteps = simulation.nsteps
    total_t = protocol.total_t
    potential = lambda x,t: system.potential(x, t)
    nframes = len(xhists)
    interval = int(round(an_time / nframes))

    fig, ax_pot = plt.subplots(1, 1, figsize=figsize)
    ax_xhist = plt.twinx(ax_pot)

    if xlim is None:
        xlow, xhigh = xhists[0][1][0], xhists[0][1][-1]
        # xdelta = xhigh - xlow
        xlim = xlow, xhigh
        # xlim = xlow - xdelta/10, xhigh + xdelta/10

    X = np.linspace(*xlim, nxsteps)
    # tlim = 0, protocol.total_t
    #
    # vline, = ax_params.plot([], [], lw=2, color=pot_color)
    # params_onto_ax(protocol.params, ax_params,
    #                tlim,
    #                protocol.substage_times)
    #
    # ax_params.hlines(0, *tlim, linestyle='dashed', alpha=0.5)
    #
    # ax_params.legend(loc='upper right')
    # ax_params.set_yticks([])

    pot_line = pot_at_time(system, 0, xlim, ax_pot, Vlim)
    pot_line.set_color(pot_color)

    ax_pot.yaxis.label.set_color(pot_color)
    ax_pot.tick_params(axis='y', colors=pot_color)

    # if xhist_bins is None:
    #     xhist_bins = ensemble.initial_dist.pos_hist[1]
    #
    # if xhist_range is None:
    #     try:
    #         xhist_range = xhist_bins[0], xhist_bins[-1]
    #     except TypeError:
    #         init_bins = ensemble.initial_dist.pos_hist[1]
    #         xhist_range = init_bins[0], init_bins[-1]
    #
    # xhist = find_pos_hist(ensemble, 0, bins=xhist_bins, range=xhist_range)

    xhist = xhists[0]
    xhist_rects = ax_xhist.bar(xhist[1][:-1], xhist[0],
                               xhist[1][1] - xhist[1][0], align='edge',
                               color=xhist_color, alpha=0.7)

    ax_xhist.set_ylabel('Pr(x)')
    ax_xhist.yaxis.label.set_color(xhist_color)
    ax_xhist.tick_params(axis='y', colors=xhist_color)

    if xhist_ymax is None:
        xhist_ymax = max(max(xhist[0]) for xhist in xhists) * 1.05

    ax_xhist.set_ylim((0, xhist_ymax))

    def update(frame):

        t = frame / nframes * total_t
        # step = round(frame / nframes * nsteps)

        # vline.set_data([t] * 100, np.linspace(-1, 1, 100))
        pot_line.set_data(X, [potential(x, t) for x in X])

        # xhist = find_pos_hist(ensemble, step, bins=xhist_bins,
        #                       range=xhist_range)
        xhist = xhists[frame]
        for i, rect in enumerate(xhist_rects):
            rect.set_height(xhist[0][i])

        return xhist_rects, pot_line#, vline

    plt.close()

    return animation.FuncAnimation(fig, update, frames=nframes,
                                   interval=interval)


def animate_simulation_with_params(system, xhists, xlim=None, nxsteps=200,
                                   an_time=5_000,
                                   Vlim=None,
                                   xhist_ymax=None,
                                   figsize=(6, 10)):

    pot_color = 'blue'
    xhist_color = 'LimeGreen'

    protocol = system.protocol
    # nsteps = simulation.nsteps
    total_t = protocol.total_t
    potential = lambda x,t: system.potential(x, t)
    nframes = len(xhists)
    interval = int(round(an_time / nframes))

    fig, (ax_params, ax_pot) = plt.subplots(2, 1, figsize=figsize);

    ax_xhist = plt.twinx(ax_pot)

    if xlim is None:
        xlow, xhigh = xhists[0][1][0], xhists[0][1][-1]
        # xdelta = xhigh - xlow
        xlim = xlow, xhigh
        # xlim = xlow - xdelta/10, xhigh + xdelta/10

    X = np.linspace(*xlim, nxsteps)
    tlim = 0, protocol.total_t

    vline, = ax_params.plot([], [], lw=2, color=pot_color)
    params_onto_ax(protocol.params, ax_params, protocol.substage_times, tlim)

    ax_params.hlines(0, *tlim, linestyle='dashed', alpha=0.5)

    ax_params.legend(loc='lower right')
    ax_params.set_yticks([])

    pot_line = pot_at_time(system, 0, xlim, ax_pot, Vlim)
    pot_line.set_color(pot_color)

    ax_pot.yaxis.label.set_color(pot_color)
    ax_pot.tick_params(axis='y', colors=pot_color)

    # if xhist_bins is None:
    #     xhist_bins = ensemble.initial_dist.pos_hist[1]
    #
    # if xhist_range is None:
    #     try:
    #         xhist_range = xhist_bins[0], xhist_bins[-1]
    #     except TypeError:
    #         init_bins = ensemble.initial_dist.pos_hist[1]
    #         xhist_range = init_bins[0], init_bins[-1]
    #
    # xhist = find_pos_hist(ensemble, 0, bins=xhist_bins, range=xhist_range)

    xhist = xhists[0]
    xhist_rects = ax_xhist.bar(xhist[1][:-1], xhist[0],
                               xhist[1][1] - xhist[1][0], align='edge',
                               color=xhist_color, alpha=0.7)

    ax_xhist.set_ylabel('Pr(x)')
    ax_xhist.yaxis.label.set_color(xhist_color)
    ax_xhist.tick_params(axis='y', colors=xhist_color)

    if xhist_ymax is None:
        xhist_ymax = max(max(xhist[0]) for xhist in xhists) * 1.05

    ax_xhist.set_ylim((0, xhist_ymax))

    def update(frame):

        t = frame / nframes * total_t
        # step = round(frame / nframes * nsteps)

        vline.set_data([t] * 100, np.linspace(-1, 1, 100))
        pot_line.set_data(X, [potential(x, t) for x in X])

        # xhist = find_pos_hist(ensemble, step, bins=xhist_bins,
        #                       range=xhist_range)
        xhist = xhists[frame]
        for i, rect in enumerate(xhist_rects):
            rect.set_height(xhist[0][i])

        return xhist_rects, pot_line#, vline

    plt.close()

    return animation.FuncAnimation(fig, update, frames=nframes,
                                   interval=interval)


def animate_simulation_with_extrema(system, infospace, xhists, xlim=None,
                                    nxsteps=200,
                                    an_time=5_000,
                                    Vlim=None, log=False,
                                    xhist_ymax=None,
                                    figsize=(6, 10),
                                    to_show_pot_ext_legend=False):

    pot_color = 'blue'
    xhist_color = 'LimeGreen'

    protocol = system.protocol
    # nsteps = simulation.nsteps
    total_t = protocol.total_t
    potential = lambda x,t: system.potential(x, t)
    nframes = len(xhists)
    interval = int(round(an_time / nframes))

    fig, (ax_extrema, ax_pot) = plt.subplots(2, 1, figsize=figsize);

    ax_xhist = plt.twinx(ax_pot)

    if xlim is None:
        xlow, xhigh = xhists[0][1][0], xhists[0][1][-1]
        # xdelta = xhigh - xlow
        xlim = xlow, xhigh
        # xlim = xlow - xdelta/10, xhigh + xdelta/10

    X = np.linspace(*xlim, nxsteps)
    tlim = 0, protocol.total_t

    vline, = ax_extrema.plot([], [], lw=2, color=pot_color)
    plot_extrema_positions(infospace, 0.1, end_plot_time=total_t,
                           substage_times=system.protocol.substage_times,
                           ax=ax_extrema,
                           to_show_legend=to_show_pot_ext_legend)

    # ax_extrema.hlines(0, *tlim, linestyle='dashed', alpha=0.5)

    if to_show_pot_ext_legend:
        ax_extrema.legend(loc='lower left')

    ax_extrema.set_yticks([])

    pot_line = pot_at_time(system, 0, xlim, ax_pot, Vlim)
    pot_line.set_color(pot_color)

    ax_pot.yaxis.label.set_color(pot_color)
    ax_pot.tick_params(axis='y', colors=pot_color)

    # if xhist_bins is None:
    #     xhist_bins = ensemble.initial_dist.pos_hist[1]
    #
    # if xhist_range is None:
    #     try:
    #         xhist_range = xhist_bins[0], xhist_bins[-1]
    #     except TypeError:
    #         init_bins = ensemble.initial_dist.pos_hist[1]
    #         xhist_range = init_bins[0], init_bins[-1]
    #
    # xhist = find_pos_hist(ensemble, 0, bins=xhist_bins, range=xhist_range)

    xhist = xhists[0]
    xhist_rects = ax_xhist.bar(xhist[1][:-1], xhist[0],
                               xhist[1][1] - xhist[1][0], align='edge',
                               color=xhist_color, alpha=0.7, log=log)

    ax_xhist.set_ylabel('Pr(x)')
    ax_xhist.yaxis.label.set_color(xhist_color)
    ax_xhist.tick_params(axis='y', colors=xhist_color)

    if xhist_ymax is None:
        xhist_ymax = max(max(xhist[0]) for xhist in xhists)
        xhist_ymax = xhist_ymax * 1.05 if not log else xhist_ymax * 2

    xhist_ymin = 0 if not log else 0.5 * min(xhist[0].min() for xhist in xhists)

    ax_xhist.set_ylim((xhist_ymin, xhist_ymax))

    def update(frame):

        t = frame / nframes * total_t
        # step = round(frame / nframes * nsteps)

        vline.set_data([t] * 100, np.linspace(-1, 1, 100))
        pot_line.set_data(X, [potential(x, t) for x in X])

        # xhist = find_pos_hist(ensemble, step, bins=xhist_bins,
        #                       range=xhist_range)
        xhist = xhists[frame]
        for i, rect in enumerate(xhist_rects):
            rect.set_height(xhist[0][i])

        return xhist_rects, pot_line#, vline

    plt.close()

    return animation.FuncAnimation(fig, update, frames=nframes,
                                   interval=interval)
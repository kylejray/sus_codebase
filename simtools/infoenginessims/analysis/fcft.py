import sys
import os

# sys.path.insert(0, os.environ['INFOENGINES_PATH'])

from math import log, inf, nan
from numpy import convolve, zeros, isclose
from numpy import negative as np_negative, log as np_log
from itertools import product
from matplotlib import pyplot as plt
from matplotlib import cm, colors

from infoenginessims.analysis.hists_1D import \
    plot_hist, plot_hists_with_total, plot_hist_with_vlines

def plot_initial_is_hists(sim, val_name, output_folder=None,
                          plot_dimensions=(7,7)):
    """Plots hists for each final info state."""

    try:

        hists = sim.output.initial_is_objs['hists'][val_name]
        nhists = len(hists)

        ymin, ymax = inf, -inf

        for is_f in range(nhists):

            hist = hists[is_f]

            if hist[0].sum() > 0:

                hmin, hmax = hist[0][hist[0] > 0].min(), hist[0].max()
                ymin = hmin if hmin < ymin else ymin
                ymax = hmax if hmax > ymax else ymax

        ymin = 1 if ymin == inf else ymin
        ymax = 1 if ymax == -inf else ymax

        ymin /= 2
        ymax *= 2

        for is_f in range(nhists):

            hist = hists[is_f]

            fig, ax = plt.subplots()

            plot_hist(hist, title=f'initial is {is_f} {val_name}',
                           log=True, ax=ax)
            ax.set_xlabel(val_name)
            ax.set_ylabel('Counts')

            ax.set_ylim((ymin, ymax))

            if output_folder is not None:
                savename = f'initial_is_{val_name}_hist_{is_f}.png'
                fig.savefig(output_folder + savename, dpi=300)
                plt.close()

    except AttributeError:
        print(f'Couldn\'t plot initial is {val_name} hist, missing attribute.')


def plot_final_is_hists(sim, val_name, output_folder=None,
                        plot_dimensions=(7,7)):
    """Plots hists for each final info state."""

    try:

        hists = sim.output.final_is_objs['hists'][val_name]
        nfinal_info_states = len(hists)

        ymin, ymax = inf, -inf

        for is_f in range(nfinal_info_states):

            hist = hists[is_f]

            if hist[0].sum() > 0:

                hmin, hmax = hist[0][hist[0] > 0].min(), hist[0].max()
                ymin = hmin if hmin < ymin else ymin
                ymax = hmax if hmax > ymax else ymax

        ymin = 1 if ymin == inf else ymin
        ymax = 1 if ymax == -inf else ymax

        ymin /= 2
        ymax *= 2

        for is_f in range(nfinal_info_states):

            hist = hists[is_f]

            fig, ax = plt.subplots()

            plot_hist(hist, title=f'final is {is_f} {val_name}', log=True,
                           ax=ax)
            ax.set_xlabel(val_name)
            ax.set_ylabel('Counts')

            ax.set_ylim((ymin, ymax))

            if output_folder is not None:
                savename = f'final_is_{val_name}_hist_{is_f}.png'
                fig.savefig(output_folder + savename, dpi=300)
                plt.close()

    except AttributeError:
        print(f'Couldn\'t plot final is {val_name} hist, missing attribute.')


def plot_substage_hists(sim, val_name, output_folder=None, figsize=(7, 7)):
    """Plot substage work hists.

    Makes a figure and axes for each substage.
    """

    hists = sim.output.substage_objs['hists'][val_name]
    nsubstages = len(hists)

    for ss in range(nsubstages):

        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(f'substage: {ss}')

        hist = hists[ss]

        plot_hist(hist, ax=ax, log=True)

        ax.set_xlabel(val_name)
        ax.set_ylabel('Counts')

        if output_folder is not None:

            savename = f'ss_{val_name}_hist_{ss}.png'
            fig.savefig(output_folder + savename, dpi=300)
            plt.close(fig)


def plot_substage_is_hists(sim, val_name, output_folder=None,
                           plot_dimensions=(7, 7)):
    """Plot substage work info state hists in a compact manner.

    Makes a figure for each substage and a subplot for each beginning and
    ending info state pair for the substage.
    """

    # hists_attribute = getattr(sim.output, f'substage_is_trans_{val_name}_hists')
    # hists = hists_attribute['hists']
    # trans_lims = hists_attribute['trans_lims']
    hists = sim.output.substage_is_objs['hists'][val_name]
    nss_info_states = sim.infospace.nss_info_states

    # bins = hists[0][0][0][0][1]

    for ss in range(len(nss_info_states) - 1):

        nis_0 = nss_info_states[ss]
        nis_1 = nss_info_states[ss + 1]

        fig, axes = plt.subplots(nis_0, nis_1,
                                 figsize=(plot_dimensions[0] * nis_1,
                                          plot_dimensions[1] * nis_0),
                                 squeeze=False)

        fig.suptitle(f'substage: {ss}')

        ymin, ymax = inf, -inf

        for is_0 in range(nis_0):

            for is_1 in range(nis_1):

                ax = axes[is_0][is_1]
                ax.set_title(f'info states: {is_0}-> {is_1}')

                ax.set_xlabel(val_name)
                # ax.set_ylabel(f'Pr({val_name}, {is_1} | {is_0})')
                ax.set_ylabel('Counts')

                hist = hists[ss][is_0][is_1]

                nonzero_vals = hist[0][hist[0] > 0]
                if nonzero_vals.sum() > 0:

                    plot_hist(hist, ax=ax, log=True)

                    hmin, hmax = nonzero_vals.min(), nonzero_vals.max()
                    ymin = hmin if hmin < ymin else ymin
                    ymax = hmax if hmax > ymax else ymax

        ymin = 1 if ymin == inf else ymin
        ymax = 1 if ymax == -inf else ymax

        ymin /= 2
        ymax *= 2

        for is_0, is_1 in product(range(nis_0), range(nis_1)):

            ax = axes[is_0][is_1]
            ax.set_ylim((ymin, ymax))

        if output_folder is not None:

            savename = f'ss_is_{val_name}_hist_{ss}.png'
            fig.savefig(output_folder + savename, dpi=300)
            plt.close(fig)


def plot_substage_is_trans_hists(sim, val_name, output_folder=None,
                                 plot_dimensions=(7, 7), to_show_legend=True):
    """Plots substage work hists in compact, overlayed manner.

    Makes a figure for each substage, and an axes for each beginning and
    ending infostate pair for the substage.  Then adds a plot for each
    subplot for each of the transition limits, overlaying each on top of the
    previous.
    """

    substage_is_trans_objs = sim.output.substage_is_trans_objs
    hists = substage_is_trans_objs['hists'][val_name]
    trans_lims = substage_is_trans_objs['trans_lims']
    nss_info_states = sim.infospace.nss_info_states
    ntrials = sim.ntrials

    for ss in range(len(nss_info_states) - 1):

        nis_0 = nss_info_states[ss]
        nis_1 = nss_info_states[ss + 1]

        fig, axes = plt.subplots(nis_0, nis_1,
                                 figsize=(plot_dimensions[0] * nis_1,
                                          plot_dimensions[1] * nis_0),
                                 squeeze=False)

        fig.suptitle(f'substage: {ss}')

        for is_0 in range(nis_0):

            ymin, ymax = inf, -inf

            for is_1, trans_index in product(range(nis_1),
                                             range(len(trans_lims))):

                hist = hists[ss][is_0][is_1][trans_index]
                good_vals = hist[0][hist[0] > 0]

                if good_vals.sum() > 0:
                    hmin, hmax = good_vals.min(), good_vals.max()

                    ymin = hmin if hmin < ymin else ymin
                    ymax = hmax if hmax > ymax else ymax

            if ymin == inf:
                ymin = 1

            if ymax == -inf:
                ymax = 1

            ymin /= 2
            ymax *= 2

            for is_1 in range(nis_1):

                ax = axes[is_0][is_1]
                ax.set_title(f'info states: {is_0}-> {is_1}')

                ax.set_xlabel(val_name)
                # ax.set_ylabel(f'Pr({val_name}, {is_1} | {is_0})')
                ax.set_ylabel('Counts')

                total = 0

                for trans_index, trans_lim in enumerate(trans_lims):

                    hist = hists[ss][is_0][is_1][trans_index]
                    bins = hist[1]

                    prob = hist[0].sum() / ntrials
                    if prob > 0:
                        label = f'trans_lim {trans_lim}\n prob: {prob:.2}'
                    else:
                        label = None

                    if (hist[0] > 0).sum() > 0:

                        xvals = bins[:-1] + (bins[1] - bins[0]) / 2
                        ax.plot(xvals, hist[0], alpha=0.8, label=label)

                    total += hist[0]

                if total.sum() > 0:

                    total_prob = total.sum() / ntrials

                    xvals = bins[:-1] + (bins[1] - bins[0]) / 2
                    ax.plot(xvals, total, '.', mec='None', c='gray', alpha=0.5,
                            label=f'total\n prob: {total_prob:.3}')

                ax.set_ylim((ymin, ymax))
                ax.set_yscale('log', nonposy='clip')

                if to_show_legend:
                    ax.legend()

        if output_folder is not None:
            savename = 'ss_is_trans_{}_hist_{}.png'.format(val_name, ss)
            fig.savefig(output_folder + savename, dpi=300)
            plt.close(fig)


def plot_substage_is_trans_hists_separated(sim, val_name, output_folder,
                                           figsize=(7, 7)):
    """Plots each ss, is_0, is_1, and trans_lim separately."""

    substage_is_trans_objs = sim.output.substage_is_trans_objs
    hists = substage_is_trans_objs['hists'][val_name]
    trans_lims = substage_is_trans_objs['trans_lims']
    nss_info_states = sim.infospace.nss_info_states
    nsubstages = len(nss_info_states) - 1
    ntrials = sim.ntrials

    for ss in range(nsubstages):

        nis_0 = nss_info_states[ss]
        nis_1 = nss_info_states[ss + 1]

        for is_0 in range(nis_0):

            ymin, ymax = inf, -inf

            for is_1, trans_index in product(range(nis_1),
                                             range(len(trans_lims))):

                hist = hists[ss][is_0][is_1][trans_index]
                good_vals = hist[0][hist[0] > 0]

                if good_vals.sum() > 0:
                    hmin, hmax = good_vals.min(), good_vals.max()

                    ymin = hmin if hmin < ymin else ymin
                    ymax = hmax if hmax > ymax else ymax

            if ymin == inf:
                ymin = 1

            if ymax == -inf:
                ymax = 1

            ymin /= 2
            ymax *= 2

            for is_1 in range(nis_1):

                for trans_index, trans_lim in enumerate(trans_lims):

                    fig, ax = plt.subplots(figsize=figsize)
                    ax.set_title(f'ss: {ss}, info states: {is_0}-> {is_1}'
                                 + f'\ntrans_lim:{trans_lim}')

                    ax.set_xlabel(val_name)
                    # ax.set_ylabel(f'Pr({val_name}, {is_1}, trans_lim | {is_0})')
                    ax.set_ylabel('Counts')

                    hist = hists[ss][is_0][is_1][trans_index]

                    prob = hist[0].sum() / ntrials
                    if prob > 0:
                        label = f'trans_lim {trans_lim}: {prob:.1e}'
                    else:
                        label = None

                    plot_hist(hist, ax=ax, log=True,
                                                     label=label)

                    ax.set_ylim((ymin, ymax))

                    ax.legend()

                    if output_folder is not None:
                        savename = 'ss_is_trans_{}_hist_sep_{}_{}_{}_{}.png'\
                                        .format(val_name, ss, is_0, is_1,
                                                trans_index)
                        fig.savefig(output_folder + savename, dpi=300)
                        plt.close(fig)


def plot_substage_time_hists(sim, val_name, output_folder=None,
                             figsize=(7,7), to_show_legend=True,
                             to_show_colorbar=False):

    colormap = cm.get_cmap('viridis')

    hists = sim.output.substage_time_objs['hists'][val_name]
    substage_times = sim.system.protocol.substage_times
    nsubstages = len(substage_times) - 1
    ntrials = sim.ntrials

    # bins = hists[0][0][0][0][1]

    for ss in range(nsubstages):

        ss_total_time = substage_times[ss + 1] - substage_times[ss]
        ntime_bins = len(hists[ss])
        bins = hists[ss][0][1]

        fig, ax = plt.subplots(figsize=figsize)

        ax.set_xlabel(val_name)
        ax.set_ylabel(f'Counts')

        fig.suptitle(f'substage: {ss}')

        total = zeros(len(bins) - 1)

        for tbin in range(ntime_bins):

            t0 = ss_total_time * (tbin / ntime_bins)
            t1 = ss_total_time * ((tbin + 1) / ntime_bins)

            hist = hists[ss][tbin]

            prob = hist[0].sum()
            if prob > 0:
                label = ''  # f'tbin: {tbin}'
                if tbin < ntime_bins - 1:
                    label += f'\n{t0:.2} <= is_0 time < {t1:.2}'
                else:
                    label = f'\n{t0:.2} <= is_0 time <= {t1:.2}'
                # label += f'\n prob: {prob:.2}'
            else:
                label = None

            color = colormap(tbin / (ntime_bins - 1))

            if (hist[0] > 0).sum() > 0:
                xvals = bins[:-1] + (bins[1] - bins[0]) / 2
                ax.plot(xvals, hist[0], alpha=0.8,# label=label,
                        color=color)

            total += hist[0]

        if total.sum() > 0:
            xvals = bins[:-1] + (bins[1] - bins[0]) / 2
            ax.plot(xvals, total, '.', mec='None', c='gray', alpha=0.5)

        ax.set_yscale('log', nonposy='clip')

        if to_show_legend:
            ax.legend()

        if to_show_colorbar:
            sm = plt.cm.ScalarMappable(
                cmap=colormap)  # , norm=plt.Normalize(vmin=0, vmin=1))
            sm._A = []
            plt.colorbar(sm)

        if output_folder is not None:
            savename = 'ss_time_{}_hist_{}.png'.format(val_name, ss)
            fig.savefig(output_folder + savename, dpi=300)
            plt.close(fig)


def plot_substage_is_time_hists(sim, val_name, output_folder=None,
                                plot_dimensions=(7, 7), to_show_legend=True,
                                to_show_colorbar=False):

    colormap = cm.get_cmap('viridis')

    # hists = getattr(sim.output, f'substage_is_time_{val_name}_hists')
    hists = sim.output.substage_is_time_objs['hists'][val_name]
    nss_info_states = sim.infospace.nss_info_states
    substage_times = sim.system.protocol.substage_times
    ntime_bins = len(hists[0][0][0])
    ntrials = sim.ntrials

    # bins = hists[0][0][0][0][1]

    for ss in range(len(nss_info_states) - 1):

        nis_0 = nss_info_states[ss]
        nis_1 = nss_info_states[ss + 1]

        ss_total_time = substage_times[ss + 1] - substage_times[ss]

        fig, axes = plt.subplots(nis_0, nis_1,
                                 figsize=(plot_dimensions[0] * nis_1,
                                          plot_dimensions[1] * nis_0),
                                 squeeze=False)

        fig.suptitle(f'substage: {ss}')

        for is_0 in range(nis_0):

            ymin, ymax = inf, -inf

            for is_1, tbin in product(range(nis_1), range(ntime_bins)):

                hist = hists[ss][is_0][is_1][tbin]
                good_vals = hist[0][hist[0] > 0]

                if good_vals.sum() > 0:

                    hmin, hmax = good_vals.min(), good_vals.max()

                    # print(f'ss: {ss}, is_0: {is_0}, is_1: {is_1}'
                    #        + f'\n  hmin, hmax: {hmin}, {hmax}')

                    ymin = hmin if hmin < ymin else ymin
                    ymax = hmax if hmax > ymax else ymax

            if ymin == inf:
                ymin = 1

            if ymax == -inf:
                ymax = 1

            ymin /= 2
            ymax *= 2

            # print(ymin, ymax)

            for is_1 in range(nis_1):

                ax = axes[is_0][is_1]
                ax.set_title(f'info states: {is_0}-> {is_1}')

                ax.set_xlabel(val_name)
                # ax.set_ylabel(f'Pr({val_name}, {is_1} | {is_0})')
                ax.set_ylabel('Counts')

                total = 0

                for tbin in range(ntime_bins):

                    t0 = ss_total_time * (tbin / ntime_bins)
                    t1 = ss_total_time * ((tbin + 1) / ntime_bins)

                    # tfrac = (tbin + 1) / ntime_bins

                    hist = hists[ss][is_0][is_1][tbin]
                    bins = hist[1]

                    count = hist[0].sum()
                    if count > 0:
                        label = '' # f'tbin: {tbin}'
                        if tbin < ntime_bins - 1:
                            label += f'\n{t0:.2} <= is_0 time < {t1:.2}'
                        else:
                            label = f'\n{t0:.2} <= is_0 time <= {t1:.2}'
                        # label += f'\n prob: {prob:.2}'
                    else:
                        label = None

                    color = colormap(tbin / (ntime_bins - 1))

                    if (hist[0] > 0).sum() > 0:
                        xvals = bins[:-1] + (bins[1] - bins[0]) / 2
                        ax.plot(xvals, hist[0], alpha=0.8,# label=label,
                                color=color)

                    total += hist[0]

                total_prob = total.sum() / ntrials

                if total.sum() > 0:
                    xvals = bins[:-1] + (bins[1] - bins[0]) / 2
                    ax.plot(xvals, total, '.', mec='None', c='gray', alpha=0.5,
                            label=f'total, prob: {total_prob:.2}')

                ax.set_ylim((ymin, ymax))
                ax.set_yscale('log', nonposy='clip')

                if to_show_legend:
                    ax.legend()

        if to_show_colorbar:
            sm = plt.cm.ScalarMappable(
                cmap=colormap)  # , norm=plt.Normalize(vmin=0, vmin=1))
            sm._A = []
            plt.colorbar(sm)

        if output_folder is not None:
            savename = 'ss_is_time_{}_hist_{}.png'.format(val_name, ss)
            fig.savefig(output_folder + savename, dpi=300)
            plt.close(fig)


def plot_miracle_at_untilt_hists(sim, val_name, output_folder=None):

    hists = sim.output.miracle_at_untilt_objs['hists'][val_name]

    labels = ['always 0', 'always 1', 'all others']

    fig, ax = plot_hists_with_total(hists, labels=labels, to_plot_fill=True,
                                    log=True)

    ax.set_xlabel(val_name)
    ax.set_ylabel('Counts')

    if output_folder is not None:
        savename = output_folder + f'miracle_at_untilt_{val_name}_hists.png'
        fig.savefig(savename, dpi=300)
        plt.close(fig)


def plot_octime_at_untilt_hists(sim, val_name, output_folder=None,
                                to_show_legend=True,
                                rev_sim=None):

    hists = sim.output.octime_at_untilt_objs['hists'][val_name]
    nhists = len(hists)

    colormap = cm.viridis
    colors = [colormap(i / (nhists - 1)) for i in range(nhists)]

    labels = [None] * nhists

    fig, ax = plot_hists_with_total(hists, colors=colors, labels=labels,
                                    to_plot_fill=False, to_show_legend=False,
                                    log=True)

    ax.set_xlabel(val_name)
    ax.set_ylabel('Counts')

    ylim = ax.get_ylim()
    ax.set_ylim(ylim)

    if rev_sim is not None:

        avg_vals = sim.output.octime_at_untilt_objs['avg_vals'][val_name]
        for_probs = sim.output.octime_at_untilt_objs['probs']
        rev_probs = rev_sim.output.octime_at_untilt_objs['probs']

        for hist, color, avg_val, for_prob, rev_prob \
                in zip(hists, colors, avg_vals, for_probs, rev_probs):

            if for_prob > 0:

                avg_val_bound = -log(rev_prob / for_prob)

                ax.vlines(avg_val_bound, ylim[0], hist[0].max()*4,
                          colors=color, label=f'mean bound: {avg_val_bound:.2}',
                          linestyle='--', alpha=0.5)

                ax.vlines(avg_val, ylim[0], hist[0].max() * 4,
                          colors=color, label=f'mean: {avg_val:.2}',
                          linestyle=':')

        if to_show_legend:
            ax.legend()

        savename = output_folder + f'octime_at_untilt_{val_name}_hists_wb.png'

    else:

        savename = output_folder + f'octime_at_untilt_{val_name}_hists.png'

    if output_folder is not None:
        fig.savefig(savename, dpi=300)
        plt.close(fig)


def plot_crooks(forward_hist, reverse_hist, val_name, xlim=None,
                output_folder=None, ax=None, figsize=(6, 5),
                save_extension='.png'):

    bins = forward_hist[1]
    xvals = (bins[:-1] + bins[1:]) / 2

    for_vals = forward_hist[0]
    rev_vals = reverse_hist[0]

    prob_ratio = rev_vals[::-1] / for_vals
    log_ratio = np_negative(np_log(prob_ratio))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(xvals, log_ratio)

    if xlim is None:
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
    else:
        ax.set_xlim(xlim)
        ax.set_ylim(xlim)

    ax.plot(xvals, xvals, '--', alpha=0.5)

    ax.set_xlabel(val_name)
    # ylabel_string = r'$\log \frac{R(-' + val_name \
    #                  + r')}{P(' + val_name + r')}$'
    ylabel_string = r'$\log \, R(-' + val_name \
                    + r') / P(' + val_name + r')$'
    ax.set_ylabel(ylabel_string)

    if output_folder is not None:
        savename = f'crooks_{val_name}_hist' + save_extension
        fig.savefig(output_folder + savename, dpi=300)
        plt.close(fig)

    return ax


def plot_TC_crooks(forward_hists, reverse_hists, val_name, xlim=None,
                   output_folder=None, figsize=(6, 5), save_extension='.png'):

    if xlim is None:
        xmin, xmax = inf, -inf
    else:
        xmin, xmax = xlim

    figs = []
    axes = []
    for forward_hist, reverse_hist in zip(forward_hists, reverse_hists):

        fig, ax = plt.subplots(figsize=figsize)

        plot_crooks(forward_hist, reverse_hist, val_name, ax=ax)

        if xlim is None:
            ax_xmin, ax_xmax = ax.get_xlim()
            xmin = ax_xmin if ax_xmin < xmin else xmin
            xmax = ax_xmax if xmin < ax_xmax else xmax

        figs.append(fig)
        axes.append(ax)

    for ax in axes:
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((xmin, xmax))

    if output_folder is not None:
        for C, fig in enumerate(figs):
            savename = f'crooks_{val_name}_hist_{C}' + save_extension
            fig.savefig(output_folder + savename, dpi=300)
            plt.close(fig)


def plot_all_partition_TC_crooks(for_sim, eq_rev_sim, noneq_rev_sim,
                                 output_folder, xlim=None, figsize=(6, 5),
                                 save_extension='.png'):

    val_names = ['W', 'ep', 'W']
    rev_sims = [eq_rev_sim, noneq_rev_sim, noneq_rev_sim]
    rev_names = ['eq', 'noneq', 'noneq']

    for partition in ['whole_', 'miracle_at_untilt_', 'octime_at_untilt_']:

        for val_name, rev_sim, rev_name in zip(val_names, rev_sims, rev_names):

            for_hists = getattr(for_sim.output, f'{partition}objs')\
                                    ['hists'][val_name]

            rev_hists = getattr(rev_sim.output, f'{partition}objs')\
                                    ['hists'][val_name]

            sub_folder  = output_folder + f'{partition}{rev_name}_{val_name}/'

            if not os.path.exists(sub_folder):
                os.makedirs(sub_folder)

            plot_TC_crooks(for_hists, rev_hists, val_name, xlim, sub_folder,
                           figsize, save_extension)

            # Needed for proper suffixes
            partition1 = partition if partition != 'whole_' else ''

            for_hists = getattr(for_sim.output, f'final_is_{partition1}objs')\
                ['hists'][val_name]

            rev_hists = getattr(rev_sim.output,
                                f'initial_is_{partition1}objs')\
                                    ['hists'][val_name]

            sub_folder = output_folder \
                          + f'final_is_{partition1}{rev_name}_{val_name}/'

            if not os.path.exists(sub_folder):
                os.makedirs(sub_folder)

            plot_TC_crooks(for_hists, rev_hists, val_name, xlim, sub_folder,
                           figsize, save_extension)


def get_TCFT_quantities(avg_vals, expavg_vals, for_probs, rev_probs):

    ratios = []
    log_ratios = []
    discreps = []
    bound_diffs = []

    total_avg_val = 0
    DKL = 0
    avg_discrep = 0

    for C, (avg_val, expavg_val, p, r) \
            in enumerate(zip(avg_vals, expavg_vals, for_probs, rev_probs)):

        # if isclose(p, 0.0) and isclose(r, 0.0):
        #     print('empty')
        #     continue

        ratio = r / p
        log_ratio = -log(ratio) if ratio > 0 else nan

        discrep = expavg_val - ratio
        bound_diff = avg_val - log_ratio

        if p > 0:
            total_avg_val += p * avg_val
            DKL += p * log_ratio
            avg_discrep += p * discrep

        ratios.append(ratio)
        log_ratios.append(log_ratio)
        discreps.append(discrep)
        bound_diffs.append(bound_diff)

    total_bound_diff = total_avg_val - DKL

    TCFT_relations = {'avg_vals': avg_vals,
                      'expavg_vals': expavg_vals,
                      'ratios': ratios,
                      'log_ratios': log_ratios,
                      'discreps': discreps,
                      'bound_diffs': bound_diffs,
                      'avg_discrep': avg_discrep,
                      'total_avg_val': total_avg_val,
                      'DKL': DKL,
                      'total_bound_diff': total_bound_diff}

    return TCFT_relations


def print_TCFT_quantities(TCFT_quantities, file=sys.stdout):

    avg_vals = TCFT_quantities['avg_vals']
    expavg_vals = TCFT_quantities['expavg_vals']

    ratios = TCFT_quantities['ratios']
    log_ratios = TCFT_quantities['log_ratios']

    discreps = TCFT_quantities['discreps']
    bound_diffs = TCFT_quantities['bound_diffs']

    avg_discrep = TCFT_quantities['avg_discrep']
    total_avg_val = TCFT_quantities['total_avg_val']
    DKL = TCFT_quantities['DKL']
    total_bound_diff = TCFT_quantities['total_bound_diff']

    print(f'----------------\n\n Partition Summary\n', file=file)
    print(f'   avg discrep: {avg_discrep}', file=file)
    print(f'   DKL:         {DKL}', file=file)
    print(f'   avg:         {total_avg_val}', file=file)
    print(f'   diff:        {total_bound_diff}', file=file)

    for C, (avg_val, expavg_val, ratio, log_ratio,
            discrep, bound_diff) \
            in enumerate(zip(avg_vals, expavg_vals, ratios,
                             log_ratios, discreps, bound_diffs)):

        print(f'\n-----\nClass {C}\n', file=file)

        print('\tratio == expavg?:', file=file)
        print(f'ratio:\t{ratio}', file=file)
        print(f'expavg:\t{expavg_val}', file=file)
        print(f'discrp:\t{discrep}', '\n', file=file)

        print('\tlogratio <= than avg?', file=file)
        print(f'lratio:\t{log_ratio}', file=file)
        print(f'avg:\t{avg_val}', file=file)
        print(f'diff:\t{bound_diff}', file=file)
        print(f'\t{avg_val > log_ratio}', file=file)


def assign_all_partition_TCFT_quantities(for_sim, eq_rev_sim, noneq_rev_sim):

    for partition in ['whole_', 'miracle_at_untilt_', 'octime_at_untilt_']:

        # First do partition without boundary infostate specificiation

        try:

            for_objs = getattr(for_sim.output, partition + 'objs')
            for_probs = for_objs['probs']

            TCFT_quantities = {}
            for_objs['TCFT_quantities'] = TCFT_quantities

            for val_name, rev_sim in zip(['W', 'ep'],
                                         [eq_rev_sim, noneq_rev_sim]):

                try:

                    rev_probs = getattr(rev_sim.output, partition + 'objs') \
                        ['probs']

                    avg_vals = for_objs['avg_vals'][val_name]
                    expavg_vals = for_objs['expavg_vals'][val_name]

                    these_TCFT_quantities \
                        = get_TCFT_quantities(avg_vals, expavg_vals,
                                              for_probs, rev_probs)

                    TCFT_quantities[val_name] = these_TCFT_quantities

                except AttributeError:

                    print(f'Couldn\'t assign {val_name} TCFT_quantities to',
                          f'{partition}objs')

            # Now do partition with final infostate specification

            # Needed for proper suffixes
            partition1 = partition if partition != 'whole_' else ''

            for_objs = getattr(for_sim.output, f'final_is_{partition1}objs')
            for_probs = for_objs['probs']

            TCFT_quantities = {}
            for_objs['TCFT_quantities'] = TCFT_quantities

            for val_name, rev_sim in zip(['W', 'ep'],
                                         [eq_rev_sim, noneq_rev_sim]):

                try:

                    rev_probs \
                        = getattr(rev_sim.output,
                                  f'initial_is_{partition1}objs')['probs']

                    avg_vals = for_objs['avg_vals'][val_name]
                    expavg_vals = for_objs['expavg_vals'][val_name]

                    these_TCFT_quantities \
                        = get_TCFT_quantities(avg_vals, expavg_vals,
                                              for_probs, rev_probs)

                    TCFT_quantities[val_name] = these_TCFT_quantities

                except AttributeError:

                    print(f'Couldn\'t assign {val_name} TCFT_quantities to',
                          f'final_is_{partition1}objs')

        except AttributeError:
            print('Couldn\'t assign TCFT_quantities to',
                  f'final_is_{partition}objs')


def print_all_partition_TCFT_quantities(for_sim, output_folder):

    for partition in ['whole_', 'miracle_at_untilt_', 'octime_at_untilt_']:

        for_objs = getattr(for_sim.output, partition + 'objs')

        for val_name in ['W', 'ep']:

            TCFT_quantities = for_objs['TCFT_quantities'][val_name]

            filename = f'{partition}{val_name}.txt'

            with open(output_folder + filename, 'w') as file:
                print_TCFT_quantities(TCFT_quantities, file)

        # Needed for proper suffixes
        partition1 = partition if partition != 'whole_' else ''

        for_objs = getattr(for_sim.output, f'final_is_{partition1}objs')

        for val_name in ['W', 'ep']:

            TCFT_quantities = for_objs['TCFT_quantities'][val_name]

            filename = f'final_is_{partition1}{val_name}.txt'

            with open(output_folder + filename, 'w') as file:
                print_TCFT_quantities(TCFT_quantities, file)


# # HAVEN'T UPDATED CONVOLUTION PLOTTING
#
# def plot_convolved_total_W_hist(sim, output_folder):
#     """Plots the total work via convolving substage work dists."""
#
#     hists = sim.output.substage_is_time_W_hists
#     nss_info_states = sim.infospace.nss_info_states
#     nsubstages = len(nss_info_states) - 1
#     final_W_hist = sim.output.final_W_hist
#
#     bins = hists[0][0][0][0][1]
#
#     b = sim.output.initial_avg_bit_value
#     initial_is_probs = [1 - b, b]
#
#     total_vals = 0
#
#     for all_is_traj in product(*[range(nis) for nis in nss_info_states]):
#
#         is_traj_vals = zeros(bins.size - 1)
#
#         initial_is_prob = initial_is_probs[all_is_traj[0]]
#         is_traj_vals[round(bins.size / 2 - 1)] = initial_is_prob
#
#         for ss in range(nsubstages):
#
#             is_0 = all_is_traj[ss]
#             is_1 = all_is_traj[ss + 1]
#
#             ss_work_vals = npsum(hist[0] for hist in hists[ss][is_0][is_1])
#
#             is_traj_vals = convolve(is_traj_vals, ss_work_vals, mode='same')
#
#         total_vals += is_traj_vals
#
#     total_hist = total_vals, bins
#
#     diff_vals = abs(total_vals - final_W_hist[0])
#     Wvals = bins[:-1] + (bins[1] - bins[0]) / 2
#
#     fig, ax = hists_1d.plot_work_hist(total_hist, log=True)
#
#     ax.plot(Wvals, diff_vals, color='k', label='discrep')
#
#     ymin = final_W_hist[0][final_W_hist[0] > 0].min() * 0.5
#     ymax = final_W_hist[0].max() * 2
#
#     ax.set_ylim(ymin, ymax)
#
#     ax.legend()
#
#     savename = output_folder + 'convolved_final_W_hist.png'
#     fig.savefig(savename, dpi=300)
#     plt.close(fig)
#
#
# def plot_cumulative_convolved_W_hists(sim, output_folder):
#     """Plots cumulative convolved work dists.
#
#     Makes a work dist for total work generated up to each substage.
#     """
#
#     ss_hists = sim.output.substage_is_time_W_hists
#     total_W_hist = sim.output.final_W_hist
#     nss_info_states = sim.infospace.nss_info_states
#
#     nsubstages = len(nss_info_states) - 1
#     bins = ss_hists[0][0][0][0][1]
#
#     b = sim.output.initial_avg_bit_value / sim.ntrials
#     initial_is_probs = [1 - b, b]
#
#     # Cumulative values of work up to a substage for ending in an info state
#     cum_vals = [[0] * nis for nis in nss_info_states]
#
#     # Set initial values of the cumulative work hist values
#     for istate in range(nss_info_states[0]):
#
#         cum_vals[0][istate] = zeros(bins.size - 1)
#         cum_vals[0][istate][round(bins.size / 2 - 1)] = initial_is_probs[istate]
#
#     # Set the cumulative work hists for each subsequent substage boundary
#     for ss in range(nsubstages):
#
#         initial_iss = range(nss_info_states[ss])
#         final_iss = range(nss_info_states[ss + 1])
#
#         for is_0, is_1 in product(initial_iss, final_iss):
#
#             ss_work_vals = npsum(hist[0] for hist in ss_hists[ss][is_0][is_1])
#
#             cum_vals[ss + 1][is_1] = cum_vals[ss + 1][is_1] \
#                                      + convolve(cum_vals[ss][is_0],
#                                                 ss_work_vals, mode='same')
#
#         cum_hists = []
#         labels = []
#         for is_1 in final_iss:
#             cum_hists.append((cum_vals[ss + 1][is_1], bins))
#             labels.append(f'is: {is_1}')
#
#         # Wvals = bins[1:] + (bins[1] - bins[0]) / 2
#
#         fig, ax = hists_1d.plot_work_hists_with_total(cum_hists,
#                                                       labels=labels)
#
#         ymin = total_W_hist[0][total_W_hist[0] > 0].min() * 0.5
#         ymax = ax.get_ylim()[1] #total_W_hist[0].max() * 2
#
#         ax.set_ylim((ymin, ymax))
#
#         savename = output_folder + f'cum_W_hist_{ss}.png'
#         fig.savefig(savename, dpi=300)
#         plt.close(fig)
#
#
#         # fig, ax = plt.subplots()
#         # ax.plot(Wvals, cum_vals[ss + 1][0])
#         # ax.plot(Wvals, cum_vals[ss + 1][1])
#         #
#         # cum_total_vals = cum_vals[ss + 1][0] + cum_vals[ss + 1][1]
#         #
#         # ax.plot(Wvals, cum_total_vals, linestyle = '--', color='k', alpha=0.5)
from math import log
import matplotlib.pyplot as plt

from itertools import product


# --------- Basic Work Dists ---------

def plot_hist(hist, title=None, fig=None, ax=None, figsize=(6, 5),
              savename=None, to_tick_top=True,
              to_plot_fill=True,
              to_show_legend=False, to_close_on_save=True, log=False,
              step='mid',
              **kwargs):

    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            ax = fig.subplots()

    # if to_label_axes:
    #     ax.set_xlabel('Work')
    #     ax.set_ylabel('Counts')

    yvals, bins = hist
    # width = bins[1:] - bins[:-1]
    # xvals = bins[:-1] + (width / 2)
    xvals = (bins[:-1] + bins[1:]) / 2

    if to_plot_fill:
        ax.fill_between(xvals, yvals, step=step, **kwargs)
        # ax.bar(hist[1][:-1], hist[0], (hist[1][1] - hist[1][0]),
        #        bottom, label=label, align='edge', **kwargs)
    else:
        ax.plot(xvals, yvals, **kwargs)

    if log:
        ax.set_yscale('log', nonposy='clip')

    if to_tick_top:
        ax.xaxis.set_tick_params(top=True, direction='in')

    if title is not None:
        ax.set_title(title)

    if to_show_legend:
        ax.legend()

    if savename is not None:
        fig.savefig(savename)
        if to_close_on_save:
            plt.close(fig)
        else:
            return fig, ax
    else:
        return fig, ax


def plot_hists_with_total(hists, colors=None, alphas=None, labels=None,
                          total_color='gray', total_alpha=None,
                          total_linestyle='--',
                          fig=None, ax=None, savename=None,
                          figsize=(6, 5), dpi=300,
                          to_plot_fill=False,
                          to_show_legend=True,
                          to_close_on_save=True, log=True, **kwargs):
    """Plots several hists via overlap and their total as a line."""

    to_plot_outline = not to_plot_fill

    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            ax = fig.subplots()

    if colors is None:
        colors = [f'C{i}' for i in range(len(hists))]

    if alphas is None:
        if to_plot_fill:
            alphas = [0.5] * len(hists)
        else:
            alphas = [1.] * len(hists)

    if labels is None:
        labels = range(len(hists))

    # if to_label_axes:
    #     ax.set_xlabel('Work')
    #     ax.set_ylabel('Counts')

    bins = hists[0][1]
    # xvals = bins[:-1] + (bins[1] - bins[0]) / 2

    total = 0

    for hist, color, alpha, label in zip(hists, colors, alphas, labels):

        # plot_work_hist(hist)

        if hist[0].sum() > 0:
            plot_hist(hist, ax=ax,
                      to_plot_fill=to_plot_fill,
                      log=log, alpha=alpha, color=color, label=label,
                      **kwargs)
            # if to_plot_fill:
            #     plot_hist(hist, ax=ax, alpha=alpha, color=color,
            #               label=label, log=log,
            #               **kwargs)
        else:
            print(f'hist {label} is empty.')

        total += hist[0]

    total_hist = total, bins

    plot_hist(total_hist, ax=ax, to_plot_fill=False,
              color=total_color, linestyle=total_linestyle, alpha=total_alpha)

    # if to_plot_fill:
    #     plot_hist(total_hist, to_plot_fill=True,
    #               c=total_color, ls='--')
    # elif to_plot_outline:
    #     plot_hist(total_hist, to_plot_outline=True,
    #               c=total_color, ms=3, alpha=0.3)

    # plot_hist(total_hist, '.', c=total_color, ms=3, alpha=0.3)

    if to_show_legend:
        ax.legend()

    if savename is not None and fig is not None:
        fig.savefig(savename, dpi=dpi)
        if to_close_on_save:
            plt.close(fig)
        else:
            return fig, ax
    else:
        return fig, ax


def plot_hist_with_vlines(hist,
                          line_xvals, line_colors, line_styles,
                          line_labels, line_alphas,
                          line_ymins=None, line_ymaxs=None,
                          fig=None, ax=None, savename=None,
                          figsize=(6, 5), dpi=300,
                          to_plot_fill=False,
                          to_show_legend=True,
                          to_close_on_save=True, **kwargs):
    """Plots a histogram along with vertical lines."""

    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            ax = fig.subplots()

    # if to_label_axes:
    #     ax.set_xlabel('Work')
    #     ax.set_ylabel('Counts')

    plot_hist(hist, log=True, to_plot_fill=to_plot_fill, to_label_axes=False,
              ax=ax, **kwargs)

    ylim = ax.get_ylim()
    ax.set_ylim(ylim)

    if line_ymins is None:
        line_ymins = [ylim[0]] * len(line_xvals)
    if line_ymaxs is None:
        line_ymaxs = [ylim[1]] * len(line_xvals)

    for xval, color, style, label, alpha, ymin, ymax \
        in zip(line_xvals, line_colors, line_styles, line_labels,
               line_alphas, line_ymins, line_ymaxs):

        ax.vlines(xval, ymin, ymax,
                  color=color, linestyle=style, label=label, alpha=alpha)

    if to_show_legend:
        ax.legend()

    if savename is not None and fig is not None:
        fig.savefig(savename, dpi=dpi)
        if to_close_on_save:
            plt.close(fig)
        else:
            return fig, ax
    else:
        return fig, ax


# # --------- Old, I think ---------
#
# def make_work_plots(hists, ymaxs=None, titles=None, xlims=None, axes=None,
#                     figsize=None, **kwargs):
#
#     nhists = len(hists)
#
#     if axes is None:
#         if figsize is None:
#             figsize = (10, 4 * nhists)
#         fig, axes = plt.subplots(nhists, 1, figsize=figsize)
#         if nhists == 1:
#             axes = [axes]
#
#     if ymaxs is None:
#         ymaxs = [None] * nhists
#
#     if titles is None:
#         titles = [None] * nhists
#
#     if xlims is None:
#
#         xmin = min(hist[1][0] for hist in hists)
#         xmax = max(hist[1][-1] for hist in hists)
#
#         margin = (xmax - xmin) * 0.05  # 0.05 seems to be default margin
#
#         xlims = [(xmin - margin, xmax + margin)] * nhists
#
#     for hist, title, ax, ymax, xlim in zip(hists, titles, axes, ymaxs, xlims):
#
#         plot_work_hist(hist, title=title, ax=ax, **kwargs)
#         ax.tick_params(top=True)
#
#         ax.set_ylabel('Counts')
#
#         if ymax is None:
#             ymax = hist[0].max()
#
#         if 'log' in kwargs and kwargs['log'] is True:
#             ax.set_ylim((1/2, ymax * 2))
#         else:
#             ax.set_ylim((0, ymax * 1.05))
#
#         ax.vlines(log(2), *ax.get_ylim(), color='cyan', linestyle='dashed',
#                   alpha=0.5, label='kBT ln2')
#
#         ax.vlines(0, *ax.get_ylim(), linestyle='dashed', alpha=0.2)
#
#         ax.set_xlim(xlim)
#
#     axes[0].legend()
#
#     axes[-1].set_xlabel('Energy (kBT)')
#
#     if axes is None:
#         fig.tight_layout()
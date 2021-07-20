import numpy as np
from matplotlib import pyplot as plt


def plot_extrema(infospace, ext_index, *args,
                 start_plot_time=None, end_plot_time=None,
                 substage_times=None, ax=None,
                 minima_color='steelblue', maxima_color='darkorange',
                 bifurcation_color='red', bifurcation_alpha=1.0,
                 xticks=None, xticklabels=None,
                 to_label_axes=True, to_draw_key_times=True,
                 to_show_legend=False,
                 **kwargs):
    """Plots either the potential extrema positions or values.

    Plots potential extrema positions if ext_index is 0, and the extrema
    values if ext_index is 1."""

    if ax is None:
        fig, ax = plt.subplots()

    ext_colors = {'minima': minima_color, 'maxima': maxima_color}

    nsteps = infospace.nsteps
    initial_time = infospace.initial_time
    final_time = infospace.final_time

    for index in range(len(infospace.cont_extrema_blocks)):

        cont_extrema = infospace.cont_extrema_blocks[index]
        block_start_time = infospace.key_times[index]
        block_end_time = infospace.key_times[index + 1]
        block_nsteps = infospace.key_steps[index + 1] \
                        - infospace.key_steps[index]

        # The final block has an extra point
        if index == len(infospace.cont_extrema_blocks) - 1:
            block_nsteps += 1

        for ext_type in ['minima', 'maxima']:

            for thread in cont_extrema[ext_type + '_threads']:

                T = np.linspace(block_start_time, block_end_time, block_nsteps)
                Y = [extrema[ext_index] for extrema in thread]

                ax.scatter(T, Y, c=ext_colors[ext_type], *args, **kwargs)

    if to_label_axes:
        ax.set_xlabel('t')
        ylabel = 'x' if ext_index is 0 else 'V'
        ax.set_ylabel(ylabel)

    if xticks:
        ax.set_xticks(xticks)

    if xticklabels:
        ax.set_xticklabels(xticklabels)

    if to_show_legend:
        ax.legend()

    # ymin = ax.get_ylim()[0] * 0.9
    # ymax = ax.get_ylim()[1] * 0.9

    ymin = ax.get_ylim()[0]
    ymax = ax.get_ylim()[1]

    if substage_times is not None:
        ax.vlines(substage_times, ymin, ymax, linestyle='--', alpha=0.5)

    if to_draw_key_times:
        ax.vlines(infospace.key_times[1:-1], ymin, ymax, linestyle='--',
                  alpha=bifurcation_alpha, color=bifurcation_color)

    ax.set_ylim(ymin, ymax)

def plot_extrema_positions(infospace, *args,
                           start_plot_time=None, end_plot_time=None,
                           substage_times=None, ax=None,
                           minima_color='steelblue', maxima_color='darkorange',
                           to_label_axes=True, to_draw_key_times=True,
                           to_show_legend=False,
                           **kwargs):

    ext_index = 0

    return plot_extrema(infospace, ext_index, *args,
                        start_plot_time=start_plot_time,
                        end_plot_time=end_plot_time,
                        substage_times=substage_times, ax=ax,
                        minima_color=minima_color, maxima_color=maxima_color,
                        to_label_axes=to_label_axes,
                        to_draw_key_times=to_draw_key_times,
                        to_show_legend=to_show_legend,
                        **kwargs)

def plot_extrema_potentials(infospace, *args,
                            start_plot_time=None, end_plot_time=None,
                            substage_times=None, ax=None,
                            minima_color='steelblue', maxima_color='darkorange',
                            to_label_axes=True, to_draw_key_times=True,
                            to_show_legend=False,
                            **kwargs):

    ext_index = 1

    return plot_extrema(infospace, ext_index, *args,
                        start_plot_time=start_plot_time,
                        end_plot_time=end_plot_time,
                        substage_times=substage_times, ax=ax,
                        minima_color=minima_color, maxima_color=maxima_color,
                        to_label_axes=to_label_axes,
                        to_draw_key_times=to_draw_key_times,
                        to_show_legend=to_show_legend,
                        **kwargs)


def plot_extrema_pos_and_pot(infospace, *args, start_plot_time=None,
                             end_plot_time=None,
                             substage_times=None, axes=None,
                             minima_color='steelblue',
                             maxima_color='darkorange',
                             to_label_axes=True, to_draw_key_times=True,
                             to_show_legend=False,
                             **kwargs):

    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(6, 10))

    ax_pos, ax_pot = axes

    plot_extrema_positions(infospace, *args,
                           start_plot_time=start_plot_time,
                           end_plot_time=end_plot_time,
                           substage_times=substage_times, ax=ax_pos,
                           minima_color=minima_color, maxima_color=maxima_color,
                           to_label_axes=to_label_axes,
                           to_draw_key_times=to_draw_key_times,
                           to_show_legend=to_show_legend,
                           **kwargs)

    plot_extrema_potentials(infospace, *args,
                            start_plot_time=start_plot_time,
                            end_plot_time=end_plot_time,
                            substage_times=substage_times, ax=ax_pot,
                            minima_color=minima_color,
                            maxima_color=maxima_color,
                            to_label_axes=to_label_axes,
                            to_draw_key_times=to_draw_key_times,
                            to_show_legend=False,
                            **kwargs)

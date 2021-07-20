import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.colors import LogNorm

def hist_onto_ax(hist, ax=None):

    if ax is None:
        fig, ax = plt.subplots()

    return ax.bar(hist[1][:-1], hist[0], width=hist[1][1]-hist[1][0],
                  align='edge')


def state_hist_onto_ax(state_hist, vmax=None, log=False, fig=None, ax=None,
                       cmap_name=None, plot_color_bar=True):

    if ax is None:
        fig, ax = plt.subplots()

    pos_bins = state_hist[1][0]
    vel_bins = state_hist[1][1]
    prob_array = state_hist[0].transpose()[::-1,::]

    extent = (pos_bins[0], pos_bins[-1], vel_bins[0], vel_bins[-1])

    if log is True:
        norm = LogNorm(vmin=prob_array[prob_array>0].min(),
                       vmax=prob_array.max())
    else:
        norm = None

    if cmap_name is not None:
        cmap = plt.cm.get_cmap(cmap_name)
    else:
        cmap = None

    im = ax.imshow(prob_array, interpolation='nearest', extent=extent,
                   aspect='auto', vmax=vmax, cmap=cmap, norm=norm)

    if plot_color_bar:
        colorbar = fig.colorbar(im, ax=ax, use_gridspec=True, shrink=0.8)
        colorbar.ax.tick_params(width=1)

    return im


def full_plot(dist, fig=None, figsize=(10, 10), pos_lim=None, vel_lim=None):

    if fig is None:
        fig = plt.figure(figsize=figsize)

    ax_pos = fig.add_axes([0.13, 0.6, 0.37, 0.4])
    ax_vel = fig.add_axes([0.57, 0.1, 0.37, 0.4])
    ax_sta = fig.add_axes([0.13, 0.1, 0.37, 0.4], sharex=ax_pos, sharey=ax_vel)

    if pos_lim:
        ax_pos.set_xlim(pos_lim)
    if vel_lim:
        ax_vel.set_ylim(vel_lim)

    ax_sta.set_xlabel('position')
    ax_sta.set_ylabel('velocity')

    # pos_hist = dist.pos_hist
    # vel_hist = dist.vel_hist
    pos_hist = dist.get_marginal_hist([0])
    vel_hist = dist.get_marginal_hist([1])

    # Convert to histogram1d from histogramdd forms
    pos_hist = pos_hist[0], pos_hist[1][0]
    vel_hist = vel_hist[0], vel_hist[1][0]

    prob_array = dist.probs.transpose()[::-1, :]

    hist_onto_ax(pos_hist, ax_pos)

    base = ax_vel.transData
    rot = transforms.Affine2D().rotate_deg(-90)

    ax_vel.bar(vel_hist[1][:-1], vel_hist[0],
               width=vel_hist[1][1]-vel_hist[1][0],
               align='edge', transform=rot+base)


    extent = (pos_hist[1][0], pos_hist[1][-1], vel_hist[1][0], vel_hist[1][-1])
    im = ax_sta.imshow(prob_array, interpolation='nearest', extent=extent,
                     aspect='auto')
    #colorbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    #colorbar.ax.tick_params(width=1)

    return im
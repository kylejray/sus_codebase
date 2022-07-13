import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from .info_space import separate_by_state


rc_dict = {'font.size':16, 'axes.labelsize':'large', 'ytick.right':False,'legend.loc':'upper right', 'legend.fontsize':'xx-small', 'figure.autolayout':True, 'figure.figsize': (10,10), 'mathtext.fontset':'stix', 'font.family':'STIXGeneral'}


def animate_sim_old(all_state, total_time, frame_skip=30, which_axes=None, axes_names=None, color_by_state=None, key_state=None, color_key=None, legend=True, alpha=None, fig_ax=None):

    if color_by_state is not None:
        if key_state is not None:
            state_lookup = separate_by_state(key_state)
        else:
            state_lookup = separate_by_state(all_state[:, 0, ...])

    N, nsteps, N_dim = np.shape(all_state)[0], np.shape(all_state)[1], np.shape(all_state)[2]

    if which_axes is None:
        assert np.size(np.shape(all_state)) in (3, 4), 'not a recognized all_state format, use which_axes kwarg or all_state of dimension [N, Nsteps, D, 2]/[N, Nsteps, D]'
        for i in range(N_dim):
            if np.size(np.shape(all_state)) == 4:
                which_axes = [np.s_[..., i, 0] for i in range(N_dim)]
            if np.size(np.shape(all_state)) == 3:
                which_axes = [np.s_[..., i] for i in range(N_dim)]

    assert len(which_axes) <= 3 and len(which_axes) > 1, 'can only plot 2 or 3 coordinates at once, use 1D histogram animation'

    x_array = [all_state[item] for item in which_axes]

    if fig_ax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax

    samples = np.linspace(0, nsteps, nsteps + 1)[::frame_skip]
    time = np.linspace(0, total_time, nsteps + 1)
    opacity=alpha
    if opacity is None:
        opacity = min(1, 500/N)

    if len(x_array) == 2:
        x = x_array[0]
        y = x_array[1]

        names = axes_names
        if axes_names is None:
            names = ('x1', 'v1')

        x_lim = (np.min(x), np.max(x))
        y_lim = (np.min(y), np.max(y))

        ax.set(xlim=x_lim, ylim=y_lim, xlabel=names[0], ylabel=names[1])
        txt = ax.set_title('t={:.2f}'.format(0))

        if color_by_state is None:
            scat = ax.scatter(x[:, 0], y[:, 0], alpha=min(1, 300/N))
        else:
            if color_key is not None:
                color_lookup = dict(zip(state_lookup, color_key))
                scat = [plt.scatter(x[state_lookup[key], 0], y[state_lookup[key], 0], c=color_lookup[key], alpha=opacity) for key in state_lookup]
            else:
                scat = [plt.scatter(x[state_lookup[key], 0], y[state_lookup[key], 0], alpha=min(1, 300/N)) for key in state_lookup]
            if legend:
                fig.legend(state_lookup)

        def animate(i):
            index = int(samples[i])
            t_c = time[index]
            x_i = x[:, index]
            y_i = y[:, index]
            if color_by_state is None:
                scat.set_offsets(np.c_[x_i, y_i])
            else:
                for i, item in enumerate(state_lookup):
                    scat[i].set_offsets(np.c_[x_i[state_lookup[item]], y_i[state_lookup[item]]])
            txt.set_text('t={:.2f}'.format(t_c))

    if len(x_array) == 3:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        x = x_array[0]
        y = x_array[1]
        z = x_array[2]

        if color_by_state is None:
            scat = ax.scatter(x[:, 0], y[:, 0], z[:, 0], alpha=min(1, 300/N))
        else:
            if color_key is not None:
                color_lookup = dict(zip(state_lookup, color_key))
                scat = [ax.scatter(x[state_lookup[key], 0], y[state_lookup[key], 0], z[state_lookup[key], 0], c=color_lookup[key], alpha=min(1, 300/N)) for key in state_lookup]
            else:
                scat = [ax.scatter(x[state_lookup[key], 0], y[state_lookup[key], 0], z[state_lookup[key], 0], alpha=opacity) for key in state_lookup]
            if legend:
                fig.legend(state_lookup)
        names = axes_names
        if names is None:
            names = ('x1', 'x2', 'x3')
        x_lim = (np.min(x), np.max(x))
        y_lim = (np.min(y), np.max(y))
        z_lim = (np.min(z), np.max(z))

        ax.set(xlim=x_lim, ylim=y_lim, zlim=z_lim, xlabel=names[0], ylabel=names[1], zlabel=names[2])
        #txt = ax.suptitle('t={:.2f}'.format(0))

        def animate(i):
            index = int(samples[i])
            t_c = time[index]
            x_i = x[:, index]
            y_i = y[:, index]
            z_i = z[:, index]
            if color_by_state is None:
                scat._offsets3d = (x_i, y_i, z_i)
            else:
                for i, key in enumerate(state_lookup):
                    scat[i]._offsets3d = (x_i[state_lookup[key]], y_i[state_lookup[key]], z_i[state_lookup[key]])

            #txt.set_text('t={:.2f}'.format(t_c))

    ani = animation.FuncAnimation(fig, animate, interval=100, frames=len(samples), blit=False)

    return ani, fig, ax

def animate_sim(all_state, times=[0,1], system=None, frame_skip=30, which_axes=None, axes_names=None, color_by_state=None, key_state=None, color_key=None, legend=True, alpha=None, fig_ax=None, **pot_kwargs):

    if color_by_state is not None:
        if key_state is not None:
            state_lookup = separate_by_state(key_state)
        else:
            state_lookup = separate_by_state(all_state[:, 0, ...])

    N, nsteps, N_dim = np.shape(all_state)[0], np.shape(all_state)[1], np.shape(all_state)[2]

    if which_axes is None:
        assert np.size(np.shape(all_state)) in (3, 4), 'not a recognized all_state format, use which_axes kwarg or all_state of dimension [N, Nsteps, D, 2]/[N, Nsteps, D]'
        for i in range(N_dim):
            if np.size(np.shape(all_state)) == 4:
                which_axes = [np.s_[..., i, 0] for i in range(N_dim)]
            if np.size(np.shape(all_state)) == 3:
                which_axes = [np.s_[..., i] for i in range(N_dim)]

    assert len(which_axes) == 2, 'can only plot 2 coordinates at once'

    x_array = [all_state[item] for item in which_axes]

    if fig_ax is None:
        fig, ax = plt.subplots()

    else:
        fig, ax = fig_ax
        
    samples = np.linspace(0, nsteps-1, nsteps)[::frame_skip]
    time = np.linspace(times[0], times[1], nsteps + 1)
    opacity=alpha
    if opacity is None:
        opacity = min(1, 300/N)

    x = x_array[0]
    y = x_array[1]

    names = axes_names

    if axes_names is None:
        names = ('x', 'y')


    x_lim = (np.min(x), np.max(x))
    y_lim = (np.min(y), np.max(y))
    
    txt = fig.suptitle('t={:.2f}'.format(times[0]))

    scat_kwargs = {'alpha':opacity, 'zorder':10}

    if color_by_state is None:
        scat = ax.scatter(x[:, 0], y[:, 0], **scat_kwargs)
    else:
        if color_key is not None:
            color_lookup = dict(zip(state_lookup, color_key))
            scat = [ax.scatter(x[state_lookup[key], 0], y[state_lookup[key], 0], c=color_lookup[key], **scat_kwargs) for key in state_lookup]
        else:
            scat = [ax.scatter(x[state_lookup[key], 0], y[state_lookup[key], 0], **scat_kwargs) for key in state_lookup]
        if legend:
            fig.legend(state_lookup)
    
    if system is not None:
        pot, pout = system.show_potential(times[0], ax=ax, cbar=False, surface=False, **pot_kwargs)
        pot.title.set_visible(False)
    ax.set(xlim=x_lim, ylim=y_lim, xlabel=names[0], ylabel=names[1])

    def animate(i):
        index = int(samples[i])
        t_c = time[index]
        x_i = x[:, index]
        y_i = y[:, index]

        if system is not None:
            new_pot = system.show_potential(t_c, cbar=False, ax=ax, surface=False, **pot_kwargs)
            #pot.collections = new_pot[0].collections

        if color_by_state is None:
            scat.set_offsets(np.c_[x_i, y_i])
        else:
            for i, item in enumerate(state_lookup):
                scat[i].set_offsets(np.c_[x_i[state_lookup[item]], y_i[state_lookup[item]]])
        txt.set_text('t={:.2f}'.format(t_c))


    ani = animation.FuncAnimation(fig, animate, interval=100, frames=len(samples), blit=False)

    return ani, fig, ax


def animate_hist_2D(all_state, total_time, which_axes=None, frame_skip=30, nbins=64, lims=None):

    N, nsteps, N_dim, _ = np.shape(all_state)
    if which_axes is None:
        which_axes = []
        for i in range(N_dim):
            which_axes.append(np.s_[:, :, i, 0])
        if N_dim == 1:
            which_axes.append(np.s_[:, :, 0, 1])

    samples = np.s_[::frame_skip]

    time = np.linspace(0, total_time, nsteps)

    time = time[samples]
    all_state = all_state[:, samples, :, :]

    x, y = all_state[which_axes[0]], all_state[which_axes[1]]
    if lims is None:
        lims = [np.min(x), np.max(x)], [np.min(y), np.max(y)]

    fig, ax = plt.subplots(figsize=(10, 10))
    txt = ax.text(0, 2, '{:.2f}'.format(0), verticalalignment='bottom')
    hist = ax.hist2d(x[:, 0], y[:, 0], bins=nbins, range=lims)

    def animate(i):
        t_c = time[i]
        x_i = x[:, i]
        y_i = y[:, i]
        hist = ax.hist2d(x_i, y_i, bins=nbins, range=lims)
        txt.set_text('t={:.2f}'.format(t_c))

    ani = animation.FuncAnimation(fig, animate, interval=100, frames=len(time), blit=True)
    return ani


def plot_state(state, which_axes=None, axes_names=None, color_by_state=None, initial_state=None, color_key=None, legend=True):

    if color_by_state is not None:
        if initial_state is None:
            state_lookup = separate_by_state(state)
        else:
            state_lookup = separate_by_state(initial_state)

    N, N_dim = np.shape(state)[0], np.shape(state)[1]

    if which_axes is None:
        assert np.size(np.shape(state)) in (2, 3), 'not a recognized state format, use which_axes kwarg or state of dimension [N, D, 2]/[N, D]'
        if np.size(np.shape(state)) == 3:
            which_axes = [np.s_[:, i, 0] for i in range(N_dim)]
        if np.size(np.shape(state)) == 2:
            which_axes = [np.s_[:, i] for i in range(N_dim)]

    if N_dim == 1 and len(which_axes) == 1:
        which_axes.append(np.s_[:, 0, 1])

    assert len(which_axes) <= 3 and len(which_axes) > 1, 'can only plot 2 or 3 coordinates at once, use 1D histogram animation'

    x_array = []
    for item in which_axes:
        x_array.append(state[item])

    fig, ax = plt.subplots()
    plt.close()
    names = axes_names

    if len(x_array) == 2:
        fig, ax = plt.subplots(figsize=(5, 5))
        x = x_array[0]
        y = x_array[1]

        if names is None and N_dim == 1:
            names = ('$x$', '$v_x$')
        if names is None and N_dim == 2:
            names = ('$x$', '$y$')

        x_lim = (np.min(x), np.max(x))
        y_lim = (np.min(y), np.max(y))

        ax.set(xlim=x_lim, ylim=y_lim, xlabel=names[0], ylabel=names[1])

        if color_by_state is None:
            scat = ax.scatter(x, y, alpha=min(1, 300/N))
        else:
            if color_key is not None:
                color_lookup = dict(zip(state_lookup, color_key))
                scat = [plt.scatter(x[state_lookup[key]], y[state_lookup[key]], c=color_lookup[key], alpha=min(1, 300/N)) for key in state_lookup]
            else:
                scat = [plt.scatter(x[state_lookup[key]], y[state_lookup[key]], alpha=min(1, 300/N)) for key in state_lookup]
            if legend:
                fig.legend(state_lookup)

    if len(x_array) == 3:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        x = x_array[0]
        y = x_array[1]
        z = x_array[2]

        if color_by_state is None:
            scat = ax.scatter(x[:], y[:], z[:], alpha=min(1, 300/N))
        else:
            if color_key is not None:
                color_lookup = dict(zip(state_lookup, color_key))
                scat = [ax.scatter(x[state_lookup[key]], y[state_lookup[key]], z[state_lookup[key]], c=color_lookup[key], alpha=min(1, 300/N)) for key in state_lookup]
            else:
                scat = [ax.scatter(x[state_lookup[key]], y[state_lookup[key]], z[state_lookup[key]], alpha=min(1, 300/N)) for key in state_lookup]
            if legend:
                fig.legend(state_lookup)

        if names is None:
            names = ('$x$', '$y$', '$z$')
        x_lim = (np.min(x), np.max(x))
        y_lim = (np.min(y), np.max(y))
        z_lim = (np.min(z), np.max(z))

        ax.set(xlim=x_lim, ylim=y_lim, zlim=z_lim, xlabel=names[0], ylabel=names[1], zlabel=names[2])

    return fig, ax


def animate_hist_1D(all_state, total_time, which_axes=None, frame_skip=20, nbins=64, lims=None):

    N, nsteps, N_dim, *_ = np.shape(all_state)
    if which_axes is None:
        which_axes = []
        for i in range(N_dim):
            which_axes.append(np.s_[:, :, i, 0])

    samples = np.s_[::frame_skip]

    time = np.linspace(0, total_time, nsteps)

    time = time[samples]

    all_state = all_state[:, samples, ...]

    coords = []
    for item in which_axes:
        coords.append(all_state[item])

    if lims is None:
        lims = [np.min(np.array(coords)), np.max(np.array(coords))]

    fig, ax = plt.subplots(figsize=(10, 10))
    txt = ax.text(0, 2, '{:.2f}'.format(0), verticalalignment='bottom')

    for j in range(len(which_axes)):
        counts, bins = np.histogram(coords[j][:, 0], bins=nbins)
        h1 = ax.hist(bins[:-1], bins, weights=counts)
        ax.set_xlim(lims)
        y_max = np.max(h1[0])
        
    ax.set_ylim([0, 1.2 * y_max])

    def animate(i):
        plt.cla()
        t_c = time[i]
        for j in range(len(which_axes)):
            counts, bins = np.histogram(coords[j][:, i], bins=nbins)
            hj = ax.hist(bins[:-1], bins, weights=counts)
            ax.set_xlim(lims)
            y_max = np.max(hj[0])
            if ax.get_ylim()[1] < y_max:
                ax.set_ylim([0, 1.2 * y_max])

        txt.set_text('t={:.2f}'.format(t_c))

    ani = animation.FuncAnimation(fig, animate, interval=100, frames=len(time), blit=False)
    return ani, fig, ax


    '''
    function to do crooks analysis for a list of works that come frmo a time symmeteic protocol. does some plots, returns some info

    Arguments
    --------
    work: ndarray of dimension [N_samples]
        the works, this is for time symmetric protocols, so no reverse process is needed
    nbins: int
        desired number of work bins
    beta: float
        1/(kB*T)
    low_stats: boolean
        if set to True (default), the function will atempt to look only in the subspace of work where we have both +W and -W realizations

    Returns
    -------
    works: ndarray of dimension [nbins,]
        array of the works asspcoated with the...
    counts: ndarray of dimensions [2, nbins]
        array of the log of the counts associate with the works above counts[0]/counts[1] is the counts for negative/positive works
    '''

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].hist(work, bins=2*nbins+1, log=True)
    ax[0].set_xlabel('$W$')
    ax[0].set_title('bare work histogram')

    naive_jar = np.mean(np.exp(-work))
    total_count = len(work)
    zero_work_count = sum(work == 0)
    print('naive jarzynski: average of exp -Beta W is {}'.format(naive_jar))
    work = work[work != 0]

    w_min, w_max = np.min(work), np.max(work)

    if low_stats:
        limit = min(abs(w_min), abs(w_max))
    else:
        limit = max(abs(w_min), abs(w_max))

    bins = np.linspace(-limit, limit, 2*nbins+1)

    counts = np.histogram(work, bins=bins)[0]

    neg_counts = counts[:nbins]
    neg_counts = neg_counts[::-1]
    pos_counts = counts[nbins:]

    step_size = limit/nbins
    works = np.linspace(step_size/2, limit-step_size/2, nbins)

    trunc_exp_work = (zero_work_count + sum(neg_counts*np.exp(beta*works)) + sum(pos_counts*np.exp(-beta*works)))/(sum(counts)+zero_work_count)

    print('binned jarzynski: binned average of exp -Beta W using only values of work where we have +W and -W realizations: {}'.format(trunc_exp_work))
    ignored_ratio = (total_count-zero_work_count-sum(counts))/total_count
    print('this means ignoring {:.1f} percent of trials'.format(100*ignored_ratio))

    ax[1].hist(work, bins, log=True)
    ax[1].set_xlabel('$W$')
    ax[1].set_title('histogram with truncated data')

    log_ratio = np.log(np.divide(pos_counts, neg_counts))

    ax[2].scatter(beta*works, log_ratio)
    ax[2].plot(beta*works, beta*works, '--')
    ax[2].set_xlabel('$\\beta W$')
    ax[2].set_ylabel('$\\ln \\frac{{P(W)}}{{P(-W)}}$')
    ax[2].set_title('Crooks for truncated data')
    plt.show()

    return works, [neg_counts, pos_counts]

def heatmap(data, col_labels, row_labels, label_slices=[np.s_[:],np.s_[:]], ax=None,
            cbar_kw={}, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data.transpose(), **kwargs, origin='lower')

    # Create colorbar
    cbar = ax.figure.colorbar(im, **cbar_kw)
    #cbar.ax.set_ylabel(cbar_kw['label'], rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[0])[label_slices[0]])
    ax.set_yticks(np.arange(data.shape[1])[label_slices[1]])
    # ... and label them with the respective list entries if not too many
    ax.set_xticklabels(col_labels[label_slices[0]])
    ax.set_yticklabels(row_labels[label_slices[1]])

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
    #         rotation_mode="anchor")

    # Turn spines off and create white grid.

    #ax.spines[:].set_visible(False)
    #ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    #ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    #ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    #ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    '''
    # Normalize the threshold to the images color range.
    try:
        threshold = im.norm(threshold)
    except TypeError:
        pass
    '''

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = plticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            try: kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            except:
                kw.update(color=textcolors)
            try:
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)
            except:
                pass

    return texts

def pcolor_diagram(field, ax=None, size=10, colorbar=False, y_min=0, y_max=None,
                x_min=0, x_max=None, ticks=True, xtick_spacing=10, ytick_spacing=10,
                tick_size=None, cmap=plt.cm.Dark2, edgecolors='none', return_pcolor=False,
                **pcolormesh_kwargs):
    '''
    Plots the given 2D field using matplotlib pcolormesh. Returns a matplotlib
    Axes object.

    Parameters
    ----------
    field: ndarray (2-D)
        2-D array of data to be plotted.
    ax: matplotlib Axes object, optional (default=None)
        An external Axes object that may be passed in to be further manipulated
        after calling this plotting function.
    size: int, optional (default=16)
        Sets the size of the Figure object.
    colorbar: bool, optional (default=False)
        Set to True to include a colorbar in the plot, False to not include a colorbar.
    y_min: int, optional (default=0)
        Lower limit of the y-axis to be plotted.
    y_max: int, optional (default=None)
        Upper limit of the y-axis to be plotted. If None, is set to the size of the
        y-axis for the input field.
    x_min: int, optional (default=0)
        Lower limit of the x-axis to be plotted.
    x_max: int, optional (default=None)
        Upper limit of the x-axis to be plotted. If None, is set to the size of the
        x-axis for the input field.
    ticks: bool, optional (default=None)
        Determines whether to display the axis tick marks and corresponding labels or not.
    xtick_spacing: int, optional (default=10)
        Sets the interval for ticks on along the x-axis.
    ytick_spacing: int, optional (default=10)
        Sets the interval for ticks along the y-axis.
    tick_size: int, optional (default=None)
        Sets the size of the tick labels for the axes. If None, defaults to the value
        of the 'size' parameter.
    cmap: matplotlib colormap, optional (default=plt.cm.Greys)
        Colormap used by pcolormesh for plotting the input field.
    edgecolors: valid matplotlib color, optional (default='black')
        Sets the color of the gird lines outlining the cells of the field.
        If set to 'none' does not display any grid lines.
    **pcolormesh_kwargs:
        Additional keyword arguments for the matplotlib pcolormesh plotting function.

    Returns
    -------
    ax: matplotlib Axes object
        Axes object that has either been passed in or created, then updated with
        this function.
    '''
    h,w = np.shape(field)
    if y_max is None:
        y_max = h
    if x_max is None:
        x_max = w
    H = y_max - y_min
    W = x_max - x_min
    if ax is None:
        figax = plt.subplots(figsize = (size, (H/W)*size))
        ax=figax[1]

    cropped_field = field[y_min:y_max, x_min:x_max]
    im = ax.pcolormesh(cropped_field, cmap=cmap, edgecolors=edgecolors, **pcolormesh_kwargs)
    ax.invert_yaxis()

    if colorbar:
        plt.colorbar(im,fraction=0.046, pad=0.04)


    # various code for tick control
    if ticks:
        ax.xaxis.tick_top()
        ax.xaxis.set_major_locator(plticker.MultipleLocator(xtick_spacing))
        ax.yaxis.set_major_locator(plticker.MultipleLocator(ytick_spacing))
        if tick_size is None:
            tick_size = size
        x_labels = [str(int(label)+x_min+xtick_spacing) for label in ax.get_xticks().tolist()]
        y_labels = [str(int(label)+y_min+ytick_spacing) for label in ax.get_yticks().tolist()]
        x_ticks = [tick+0.5 for tick in ax.get_xticks()]
        y_ticks = [tick +0.5 for tick in ax.get_yticks()]
        if W % xtick_spacing == 0:
            x_trim = 2
        else:
            x_trim = 1
        if H % ytick_spacing == 0:
            y_trim = 2
        else:
            y_trim = 1
        ax.set_xticks(x_ticks[1:len(x_ticks)-x_trim], minor=True)
        ax.set_yticks(y_ticks[1:len(y_ticks)-y_trim], minor=True)
        ax.set_xticklabels(x_labels, fontsize=tick_size, minor=True)
        ax.set_yticklabels(y_labels, fontsize=tick_size, minor=True)
        ax.xaxis.set_major_formatter(plticker.NullFormatter())
        ax.yaxis.set_major_formatter(plticker.NullFormatter())
        ax.tick_params(
                axis='both',
                which='minor',
                direction='out',
                top='off',
                right='off',
                pad=8)
        ax.tick_params(
                axis='both',
                which='major',
                top='off',
                bottom='off',
                left='off',
                right='off')
    else:
        ax.tick_params(axis='both',
                        which='both',
                        bottom='off',
                        top='off',
                        left='off',
                        right='off',
                        labelleft='off',
                        labelbottom='off')
    if return_pcolor:
        return(im,ax)
    else:
        return ax

'''
def equilibrated_state(eq_system, T=1, N=5000, initial_state=None, eq_period=1, what_time=0, max_iterations=4):

    delta_E = 1
    i = 0
    nsteps = 1000
    gamma = 1
    theta = 1
    eta = 1 * np.sqrt(T)
    dynamic = langevin_underdamped.LangevinUnderdamped(theta, gamma, eta,
                                                       eq_system.get_external_force)

    integrator = rkdeterm_eulerstoch.RKDetermEulerStoch(dynamic)

    procedures = [sp.ReturnFinalState()]

    trivial_protocol = eq_system.potential.trivial_protocol()
    trivial_protocol.time_stretch(eq_period)

    for i, item in enumerate(eq_system.protocol.get_params(what_time)):
        trivial_protocol.change_params(which_params=i+1, new_params=item)

    system = eq_system.copy()
    system.protocol = trivial_protocol
    total_time = system.protocol.t_f - system.protocol.t_i

    if initial_state is None:
        initial_state = system.eq_state(N, resolution=100, damping=None)
        sys.stdout.write("\033[K")

    while delta_E >= .001 and i <= max_iterations:
        dt = total_time / nsteps
        sim = simulation.Simulation(integrator.update_state, procedures, nsteps, dt, initial_state)

        sim.system = system
        sim.output = sim.run(verbose=True)
        sys.stdout.write("\033[K")
        sys.stdout.write("\033[K")
        equilibrated_state = sim.output.final_state
        last_delta_E = delta_E
        delta_E = (sum(system.get_energy(equilibrated_state, 0)) - sum(system.get_energy(initial_state, 0)))/sum(system.get_energy(initial_state, 0))
        delta_E = abs(delta_E)
        if (last_delta_E - delta_E)/last_delta_E < .15:
            nsteps += 500
        initial_state = equilibrated_state
        print(i, delta_E)
        i += 1
    return(equilibrated_state)
    '''

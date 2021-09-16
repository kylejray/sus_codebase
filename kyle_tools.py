
import sys
import os



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


is_path = os.path.expanduser('~/source/informational_states/')
sys.path.insert(0, is_path)

from measure import MeasurementDevice, Measurement



def separate_by_state(state, **kwargs):
    kwargs['trajectory_mode'] = False
    measurement_device = MeasurementDevice(**kwargs)

    _, bools = measurement_device.apply(state)

    return measurement_device.get_lookup(bools)


def animate_sim(all_state, total_time, frame_skip=30, which_axes=None, axes_names=None, color_by_state=None, key_state=None, color_key=None, legend=True, alpha=None):

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

    fig, ax = plt.subplots()
    samples = np.linspace(0, nsteps, nsteps + 1)[::frame_skip]
    time = np.linspace(0, total_time, nsteps + 1)
    opacity=alpha
    if opacity is None:
        opacity = min(1, 500/N)

    if len(x_array) == 2:
        fig, ax = plt.subplots(figsize=(5, 5))
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
        txt = ax.set_title('t={:.2f}'.format(0))

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

            txt.set_text('t={:.2f}'.format(t_c))

    ani = animation.FuncAnimation(fig, animate, interval=100, frames=len(samples), blit=False)

    return ani


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

    ani = animation.FuncAnimation(fig, animate, interval=100, frames=len(time), blit=False)
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

    N, nsteps, N_dim, _ = np.shape(all_state)
    if which_axes is None:
        which_axes = []
        for i in range(N_dim):
            which_axes.append(np.s_[:, :, i, 0])

    samples = np.s_[::frame_skip]

    time = np.linspace(0, total_time, nsteps)

    time = time[samples]

    all_state = all_state[:, samples, :, :]

    coords = []
    for item in which_axes:
        coords.append(all_state[item])

    if lims is None:
        lims = [np.min(coords), np.max(coords)]

    fig, ax = plt.subplots(figsize=(10, 10))
    txt = ax.text(0, 2, '{:.2f}'.format(0), verticalalignment='bottom')

    for j in range(len(which_axes)):
        counts, bins = np.histogram(coords[j][:, 0], bins=nbins)
        h1 = ax.hist(bins[:-1], bins, weights=counts, density=True)
        ax.set_xlim(lims)
        y_max = np.max(h1[0])
        
    ax.set_ylim([0, 1.2 * y_max])

    def animate(i):
        plt.cla()
        t_c = time[i]
        ax.set_ylim([0, 1.2 * y_max])
        for j in range(len(which_axes)):
            counts, bins = np.histogram(coords[j][:, i], bins=nbins)
            ax.hist(bins[:-1], bins, weights=counts, density=True)
            ax.set_xlim(lims)
            ax.set_ylim([0, 1.2 * y_max])

        txt.set_text('t={:.2f}'.format(t_c))

    ani = animation.FuncAnimation(fig, animate, interval=100, frames=len(time), blit=False)
    return ani


def szilard_accuracy_init_final(init_s, fin_s):
    tfs = fin_s
    tis = init_s
    suc_L2U = sum(tfs[tis[:, 0, 0] > 0][:, 1, 0] > 0)
    suc_R2D = sum(tfs[tis[:, 0, 0] < 0][:, 1, 0] < 0)
    fail_L2D = sum(tfs[tis[:, 0, 0] > 0][:, 1, 0] < 0)
    fail_R2U = sum(tfs[tis[:, 0, 0] < 0][:, 1, 0] > 0)
    accuracy = (suc_L2U+suc_R2D)/len(tis)
    failure = (fail_L2D+fail_R2U)/len(tis)
    return accuracy, failure


def szilard_accuracy_all_state(system, all_state, offsets=None, return_trajectories=False):
    positions = all_state[..., 0]
    N, steps, _ = np.shape(positions)

    if offsets is not None:
        positions = positions - offsets
    
    fs_lookup = separate_by_state(positions[:,-1])
    fs_distribution=[]
    target_distribution = .25 * np.ones(4)
    for key in fs_lookup:
        fs_distribution.append(sum(fs_lookup[key]))
    fs_distribution = np.divide( fs_distribution, N)
    dkl = sum(fs_distribution * np.log( fs_distribution/ target_distribution))

    binary_device = MeasurementDevice()
    binary_measurement = Measurement(binary_device, dataset=positions)
    traj = binary_measurement.trajectories_by_number()
    numb_lookup = binary_device.get_lookup(binary_measurement.outcome_numbers)

    bound = int(steps/2)
    measure = np.s_[:,:bound]
    control = np.s_[:,bound:]

    L0L1_measure = ((traj[measure] == numb_lookup['00']) | (traj[measure] == numb_lookup['01'])).all(axis=1) & (traj[:, bound] == numb_lookup['00'])
    L0L1_control = ((traj[control] == numb_lookup['00']) | (traj[control] == numb_lookup['10'])).all(axis=1)
    L0L1_succ = L0L1_measure & L0L1_control

    R0R1_measure = ((traj[measure] == numb_lookup['10']) | (traj[measure] == numb_lookup['11'])).all(axis=1) & (traj[:, bound] == numb_lookup['11'])
    R0R1_control = ((traj[control] == numb_lookup['01']) | (traj[control] == numb_lookup['11'])).all(axis=1)
    R0R1_succ = R0R1_measure & R0R1_control

    accuracy = (sum(L0L1_succ) + sum(R0R1_succ))/N, dkl
    print("success ratio, dkl:", accuracy)

    if return_trajectories:
        bools = ~(L0L1_succ | R0R1_succ), L0L1_succ, R0R1_succ
        keys = 'failures', 'right_sucess', 'left_success'
        print('second return is a dictionary with the following keys:',keys)
        trajectory_dict = dict(zip(keys, bools))
        return accuracy, trajectory_dict 
    else:
        return accuracy


def fredkin_fidelity(initial_state, final_state, verbose=False):
    trials = len(initial_state)

    is_lookup = separate_by_state(initial_state)
    fs_lookup = separate_by_state(final_state)

    storage_fixed_gates = ['000', '001', '010', '011']
    comp_fixed_gates = ['100', '111']

    sfg_succ = 0
    sfg_total = 0
    for key in storage_fixed_gates:
        sfg_total += sum(is_lookup[key])
        sfg_succ += sum(is_lookup[key] & fs_lookup[key])

    cfg_succ = 0
    cfg_total = 0
    for key in comp_fixed_gates:
        cfg_total += sum(is_lookup[key])
        cfg_succ += sum(is_lookup[key] & fs_lookup[key])

    csg_succ = 0
    csg_total = sum(is_lookup['101']) + sum(is_lookup['110'])
    csg_succ += sum(is_lookup['101'] & fs_lookup['110'])
    csg_succ += sum(is_lookup['110'] & fs_lookup['101'])

    marginal_success = [[csg_succ, csg_total], [cfg_succ, cfg_total], [sfg_succ, sfg_total]]
    success, total = np.sum(marginal_success, axis=0)
    marginal_success.append([success, total])

    if verbose is True:
        print('swap gates:{} success out of {}'.format(csg_succ, csg_total))
        print('computational fixed gates:{} success out of {}'.format(cfg_succ, cfg_total))
        print('storage fixed gates:{} success out of {}'.format(sfg_succ, sfg_total))
        names = ['swap gates', 'computational fixed gates', 'storage fixed gates', 'overall fidelity']
        return(dict(zip(names, marginal_success)))
    else:
        return success/total


def crooks_analysis_tsp(work, nbins=25, beta=1, low_stats=True):
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


import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../informational_states/")

import measure

MeasurementDevice, Measurement = measure.MeasurementDevice, measure.Measurement


def separate_by_state(state, **kwargs):
    kwargs['trajectory_mode'] = False
    measurement_device = MeasurementDevice(**kwargs)

    _, bools = measurement_device.apply(state)

    return measurement_device.get_lookup(bools)

def is_bundle(state_ensemble, state_bools, n_points):
    coords=[]
    weights=[]
    for key in state_bools:
        info_state = state_ensemble[state_bools[key]]
        c, w = representative_bundle(info_state, n_points)
        coords.append(c)
        weights.append(w)
    output = coords[0]
    out_weight = weights[0]
    for item in zip(coords[1:], weights[1:]):
        output = np.append(output,item[0], axis=0)
        out_weight = np.append(out_weight, item[1], axis=0)
    
    return output, out_weight

def representative_bundle(state_ensemble, n_points):
    state_ensemble =np.squeeze(state_ensemble)
    shape = np.shape(state_ensemble)
    N, state_shape = shape[0], shape[1:]
    state = state_ensemble.reshape(N, np.prod(state_shape))


    state_means = np.mean(state, axis=0)
    state_stds = np.std(state, axis=0)
    bins = np.linspace(state_means-3*state_stds, state_means+3*state_stds, n_points+1).transpose()
    hist = np.histogramdd(state, bins=bins, density=True)
    values = bins[:,:-1] + np.diff(bins)/2
    mesh = np.meshgrid(*values)
    mesh = [item.ravel() for item in mesh]
    coords = np.array([ np.reshape(item, state_shape) for item in list(zip(*mesh))])
    weights = np.ravel(hist[0])

    return coords[weights != 0], weights[weights != 0]

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

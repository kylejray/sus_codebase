import os
import sys
import json
import numpy as np

def jsonify(data):
    json_data = dict()
    for key, value in data.items():
        if isinstance(value, list) or isinstance(value, tuple): # for lists and tuples
            value = [ jsonify(item) if isinstance(item, dict) else item for item in value ]
        if isinstance(value, dict): # for nested lists
            value = jsonify(value)
        if isinstance(key, int): # if key is integer: > to string
            key = str(key)
        if type(value).__module__=='numpy': # if value is numpy.*: > to python list
            value = value.tolist()
        if type(value)==range: # if value is range > to python list
            value = str(value)
        if isinstance(value, slice):
            value = str(value)
            
        json_data[key] = value

    return json_data

def open_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def file_list(directory, prefix_list=[''], extension_list=['.json']):
    '''
    returns a list of strings representing filenames in dir that have extensions in extension_list and start with prefixes in prefix list
    '''
    def filter_func(f):
        valid_ext = os.path.splitext(f)[1] in extension_list
        valid_pref = any([f.startswith(item) for item in prefix_list])
        return valid_ext & valid_pref

    file_list = list(filter(lambda f: filter_func(f), os.listdir(directory)))
    return [directory+f for f in file_list]


'''
WIP
def numpify(data):
    np_data = dict()
    for key, value in data.items():
        if isinstance(value, list): # for lists
            value = [ numpify(item) if isinstance(item, dict) else item for item in value ]
        if isinstance(value, dict): # for nested lists
            value = numpify(value)
        if isinstance(key, int): # if key is integer: > to string
            key = str(key)
        if type(value)==float: # if value is float: > to numpy array
            value = value.tolist()
        np_data[key] = value
    return json_data
'''

'''

def check_jsonable(d, dictionary=True):
     ...:     if dictionary:
     ...:         for key, value in d.items():
     ...:             if isinstance(value, list) or isinstance(value, tuple):
     ...:                 for item in value:
     ...:                     if isinstance(item, dict):
     ...:                         check_jsonable(item)
     ...:                     else:
     ...:                         check_jsonable(dictionary=False)
     ...:      if isinstance(value, dict):
     ...:         check_jsonable(value)
     ...:      if isinstance(value, list):
     ...:      if type(value).__module__ is not 'builtins':
     ...:         print(key, type(value), type(value).__module__)
'''

def get_size(obj, seen=None):
    """Recursively finds size of objects"""

    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])

    return size/ 1024**2

def inv_xtanhx(arg, tol=.001, max_iterations=10):
    if np.isclose(abs(arg), arg*np.tanh(arg), atol=tol):
        return abs(arg)

    done=False
    steps = int(1/tol)
    i=0

    while not done and i <= max_iterations:
        x = np.linspace(arg-1, arg+1, 2*steps)
        x = x[x>0]
        y = x * np.tanh(x) - arg
        absy = np.sign(y)
        decider = np.diff(absy, prepend=absy[0])
        i_g = np.where(decider>0)[0]
        assert len(i_g)==1, 'inv xtanx  algorithm failed'
        i_l = i_g-1
        ratio = abs(y[i_g]/y[i_l])
        output = (ratio*x[i_l] +x[i_g])/(ratio+1)

        if np.isclose(output, arg*np.tanh(arg), atol=tol):
            done=True
        else:
            steps = steps*2
            i+=1
    return output  
'''
from sklearn.neighbors import KernelDensity


def kde(train_data, bandwidth=1, kernel='gaussian'):
    n_dim = len(train_data)
    
    train_data = np.c_[*train_data]
    
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)

        
    kde.fit(train_data)
    kde.n = n_dim

    return kde

def kde_prob(kde, data)
    original_shape = np.shape(data)
    assert len(original_shape) == 2


    plot_data = np.c_[Y.ravel(),X.ravel()]
    
    logprob = kde.score_samples(plot_data)
    rs_log = np.reshape(logprob, (mesh, mesh))
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_wireframe(X, Y, np.exp(rs_log), alpha=.5)
    return(fig, ax)

    xlims = [50,100]
    ylims = [0,100]
    x_d = np.linspace(*xlims, mesh)
    y_d = np.linspace(*ylims, mesh)

    X, Y = np.meshgrid(x_d, y_d)

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

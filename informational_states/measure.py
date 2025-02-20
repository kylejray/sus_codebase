import numpy as np

def binary_partition(positions, boundary=0):
    '''
    takes a set of position coordinates and sets each value to either 0 or 1 depending on if it is below or above the boundary
    '''
    return (np.sign(positions-boundary)+1)/2


class MeasurementDevice:
    def __init__(self, outcome_names=None, outcome_values=None, transformation_function=binary_partition, trajectory_mode=True):
        self.trajectory_mode = trajectory_mode
        self.outcome_names = outcome_names
        self.outcome_values = outcome_values
        self.transform = transformation_function

        self.default_names = False
        if self.outcome_names is None:
            self.default_names = True
        self.default_values = False
        if self.outcome_values is None:
            self.default_values = True
    
    def data_space_info(self, data):
        shape = data.shape

        if self.trajectory_mode is False:
            assert np.size(shape) <= 3, 'unrecognized state format, state needs to be of dimension [N, N_d, N_x], [N, N_d], [N]'
            boundary_idx = 1

        if self.trajectory_mode:
            assert np.size(shape) in (2,3,4), 'unrecognized state format, state needs to be of dimension [N, N_t, N_d, N_x] or [N, N_t, N_d], [N,N_t]'
            boundary_idx = 2

        trials_shape = shape[:boundary_idx]
        coord_space_shape = shape[boundary_idx:]


        return [trials_shape, coord_space_shape]


    def apply(self, data):
        data = data.squeeze()
        measured_data = self.transform(data)

        target_shape, post_measure_coord_space = self.data_space_info(measured_data)
        try:
            d = post_measure_coord_space[0]
        except:
            d = 1

        if self.default_names:
            self.outcome_names = get_default_names(d)
            self.default_names = False
        if self.default_values:
            self.outcome_values = get_default_values(d)
            self.default_values = False

        which_ax = post_measure_coord_space
        if which_ax is not ():
            which_ax = -1

        state_booleans = self.is_data_in_outcome(measured_data, target_shape, which_ax)

        return measured_data, state_booleans
    
    def get_lookup(self, attribute):
        assert len(self.outcome_names) == len(attribute), 'number of elements must match number of state names' 
        return dict(zip(self.outcome_names, attribute))

    def is_data_in_outcome(self, data, target_shape, which_ax, outcome_vals_list=None):
        if outcome_vals_list is None:
            outcome_vals_list = self.outcome_values
    
        outcome_vals = [ item if type(item) is list else [item] for item in outcome_vals_list]

        bools = []

        for outcomes in outcome_vals :
            temp_bools = [np.all(data == outcome, axis=which_ax) for outcome in outcomes]

            if temp_bools[0].shape != target_shape:
                assert len(temp_bools[0].shape) > len(target_shape), 'dimension mismatch between outcome_values and the measured data'
                temp_bools = [np.all(item, axis=-1) for item in temp_bools]

            if len(outcomes) > 1:
                temp_bools = [np.asarray(temp_bools).any(axis=0)]

            bools.append(temp_bools[0])

        return bools



class Measurement:
    def __init__(self, dataset, MeasurementDevice=MeasurementDevice()):
        self.device = MeasurementDevice
        self.data = dataset
        self.trajectories = None
        self.measured_data, self.booleans = self.device.apply(self.data)

    def measure_new_dataset(self, new_data, confirmation=False):
        if confirmation is False:
            print('will delete all current measurement and trajectory related attributes, set keyword confirmation=True and re-call the method to continue')
        if confirmation:
            print('setting new data, make sure they are compatible with current device outcome_names, outcome_values, and transform attributes')
            self.data = new_data
            self.measured_data = None
            self.booleans = None
            self.trajectories = None
            self.outcome_numbers = None

            self.measured_data, self.booleans = self.device.apply()


    def calculate_trajectory_statistics(self, trajectories=None):
        if trajectories is None:
            if self.trajectories is None:
                self.trajectories = self.trajectories_by_number()
            trajectories = self.trajectories

        assert len(trajectories.shape) == 2, 'unrecognized trajectory format, must be ndarray of dimension [N, N_t]'
    
        N_trajectories = len(trajectories)

        unique_trajectories, trajectory_counts = np.unique(trajectories, axis=0, return_counts=True)

        print( 'found {} unique trajectories in {} samples'.format(len(unique_trajectories), N_trajectories))

        sorting_idx = trajectory_counts.argsort()[::-1]
        sorted_trajectories = unique_trajectories[sorting_idx]
        trajectory_probs = (np.sort(trajectory_counts)[::-1])/np.sum(trajectory_counts)
        
        return self.make_trajectory_ensemble(sorted_trajectories, trajectory_probs)

    def trajectories_by_number(self, boolean_list = None):
        '''
        takes the state booleans and turns them into an array or trajectories where the value at each times step is related to the state
        '''
        if boolean_list is None:
            bools = self.booleans 

        n_outcomes = len(bools)
        d_type = 'uint8'
        for item in [8, 16, 32]:
            if np.log2(n_outcomes) > item:
                d_type = 'uint' + '{}'.format(2*item)

        full_array = np.array(bools)
        numbers = np.array(range(n_outcomes))+1
        numbered_array = np.transpose( full_array.transpose()* numbers)
        trajectories = numbered_array.sum(axis=0, dtype = d_type)

        self.outcome_numbers = numbers

        return trajectories
    
    def make_trajectory_ensemble(self, unique_traj, probabilities, outcome_names=None):
        if outcome_names is None:
            outcome_names = self.device.outcome_names
        return TrajectoryEnsemble(unique_traj, probabilities, outcome_names)


class TrajectoryEnsemble:
    def __init__(self, unique_trajectories, probabilities, outcome_names):
        self.unique_traj = unique_trajectories
        self.traj_probs = probabilities
        self.outcome_names = outcome_names
        self.outcome_numbers = np.array(range(len(self.outcome_names))).astype(np.uint8)+1
        self.trajectory_statistics = [self.unique_traj, self.traj_probs]

    def show_trajectories(self):
        return replace_int_keys_with_string(self.outcome_numbers, self.outcome_names, self.unique_traj)

    def calculate_dynamics(self, trajectory_statistics=None, N_states=None):

        if trajectory_statistics is None:
            trajectory_statistics = self.trajectory_statistics

        trajectories, probs = trajectory_statistics

        N_t = len(trajectories[0])

        if N_states is None:
            N_states = len(self.outcome_numbers)

        transition_matrices = np.zeros((N_t-1, N_states, N_states))
        for i in range(N_t-1):
            transition_matrices[i, ...] = self.get_local_transition_matrix(trajectories, probs, i)

        self.dynamics = transition_matrices
        return transition_matrices

    def markovity_test(self, transition_matrices, tol_kwargs=None):
        if tol_kwargs is None:
            abs_tol = .05
            rel_tol = .1
            tol_kwargs = {'atol':abs_tol, 'rtol':rel_tol}

        mean_matrix = np.mean(transition_matrices, axis=0)

        is_close = np.isclose(mean_matrix, transition_matrices, **tol_kwargs)

        fraction_within = np.sum(is_close, axis=0)/len(is_close)
        threshhold_ratio = .8
        fraction_within = fraction_within > threshhold_ratio

        print('{} of {} transition matrix elements stay within {} of their means {}% of the time'.format(fraction_within.sum(), fraction_within.size, abs_tol, 100*threshhold_ratio ))

        return(is_close)

    def get_local_transition_matrix(self, trajectories, probs, i, outcome_numbers=None):
        local_state = trajectories[:, i:i+2]
        if outcome_numbers is None:
            outcome_numbers = self.outcome_numbers
        N_states = len(outcome_numbers)
        transition_matrix = np.zeros((N_states, N_states))
        for x in range(N_states):
            if sum(local_state[:,0] == outcome_numbers[x]) != 0:
                for y in range(N_states):
                    transition_bool = np.all( local_state == [outcome_numbers[x], outcome_numbers[y]], axis=1)
                    scaled_transitions =np.sum(transition_bool * probs)
                    transition_matrix[x, y] = scaled_transitions
            else:
                transition_matrix[x, :] = np.nan

        transition_matrix = ((transition_matrix.transpose()/transition_matrix.sum(axis=1)).transpose())
        return transition_matrix

    def reverse_probability(self, i, trajectory_statistics=None, reverse_transformation=None, atol=None, rtol=None):
        if trajectory_statistics is None:
            trajectory_statistics = self.trajectory_statistics

        trajectories, probs = trajectory_statistics

        current_traj = trajectories[i]

        if reverse_transformation is None:
            reverse_traj = current_traj[::-1]
        else:
            reverse_traj = reverse_transformation(current_traj)

        find_reverse = np.all(trajectories == reverse_traj, axis=1)

        if sum(find_reverse) > 1:
            print('found more than one reverse trajectory for a single trajectory, might want to double check trajectory_statistics')

        return sum(probs[find_reverse])


    def reversibility_test(self, trajectory_statistics = None, tol_kwargs=None):
        if tol_kwargs is None:
            abs_tol = .05
            rel_tol = .1
            tol_kwargs = {'atol':abs_tol, 'rtol':rel_tol}
        if trajectory_statistics is None:
            trajectory_statistics = self.trajectory_statistics

        reverse_probs=[]
        for i in range(len(trajectory_statistics[0])):
            print( '{:.1f}% done'.format(100*(i/len(trajectory_statistics[0]))), end='\r')
            rev_prob = self.reverse_probability(i, trajectory_statistics=trajectory_statistics)
            reverse_probs.append(rev_prob)

        test = np.isclose( trajectory_statistics[1], reverse_probs, **tol_kwargs)
        sum_of_test = np.sum(test)
        print('{} out of {} trajectories were reversible'.format(sum_of_test, len(trajectory_statistics[1])))

        return reverse_probs





def get_default_values(d):
    assert d in (
        1, 2, 3), 'default inf_state names and coords provided for only 1,2, and 3 dimensions'

    if d == 1:
        informational_outcome_values = [0, 1]
    if d == 2:
        informational_outcome_values = [(0, 0), (0, 1), (1, 0), (1, 1)]
    if d == 3:
        informational_outcome_values = [
            (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]

    return informational_outcome_values


def get_default_names(d):
    assert d in (
        1, 2, 3), 'default inf_state names and coords provided for only 1,2, and 3 dimensions'

    if d == 1:
        informational_outcome_names = ['0', '1']
    if d == 2:
        informational_outcome_names = ['00', '01', '10', '11']
    if d == 3:
        informational_outcome_names = ['000', '001',
                                     '010', '011', '100', '101', '110', '111']

    return informational_outcome_names

def replace_int_keys_with_string(keys, values, input_array):
    keys = np.array(keys)
    values = np.array(values)

    mapping_array = np.zeros(keys.max()+1, dtype=values.dtype)
    mapping_array[keys] = values

    return mapping_array[input_array]

def make_sample_data(N=10000, N_t=8, N_d=2, markov=True):
    data = np.zeros((N, N_t, N_d))
    if markov is False:
        means = np.linspace(-10, 10, N_t )
        for i in range(N_t):
            data[:, i, :] = np.random.normal(means[i], 2, (N, N_d) )
        return data
    else:
        means = np.random.uniform(-5, 5, (N, N_d))
        for i in range(N_t):
            data[:, i, :] = np.random.normal(means, 1, (N,N_d))
            means = data[:, i, :] 
        return data

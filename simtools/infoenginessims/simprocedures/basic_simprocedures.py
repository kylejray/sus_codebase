from numpy import empty, s_, histogramdd, mean, shape, array, average
from scipy.stats import sem


class SimProcedure:
    """Base class for simulation procedures.

    _Methods_
    do_initial_task: Called during the simulation's run method after all of its
        initialization steps and right before the main state evolution loop.
    do_intermediate_task: Called at the end of each step of the simulation's
        main state evolution loop.
    do_final_task: Called right before the end of the simulation's run method.
        Should return the output appropriate the procedure.  If the output of
        the simulation should have no contribution from this procedure, the
        output of this method should be None.
    """

    def do_initial_task(self, simulation):
        pass

    def do_intermediate_task(self):
        pass

    def do_final_task(self):
        return None


# --------- State Measurements --------- #

class ReturnFinalState(SimProcedure):
    """Measurement that returns the supposedly existent final next_states."""

    def do_initial_task(self, simulation, output_name='final_state'):

        self.simulation = simulation
        self.output_name = output_name

    def do_final_task(self):
        return self.simulation.next_state

class ReturnInitialState(SimProcedure):
    """Measurement that returns the supposedly existent final next_states."""

    def do_initial_task(self, simulation, output_name='initial_state'):

        self.simulation = simulation
        self.output_name = output_name

    def do_final_task(self):
        return self.simulation.initial_state


class MeasureAllState(SimProcedure):
    """Measurement that returns for a subset of trials.

    The trial_indices argument can take lists of indices (integer array
    indexing), slices, and numpy index expressions.
    """

    def __init__(self, step_request=s_[:], trial_request=s_[:],
                 output_name='all_state'):

        self.step_request = step_request
        self.trial_request = trial_request
        self.output_name = output_name

    def do_initial_task(self, simulation):

        self.simulation = simulation

        initial_state = simulation.initial_state

        state_shape = initial_state.shape[1:]
        # nstate_dims = initial_state.shape[1]

        trial_indices = range(self.simulation.ntrials)[self.trial_request]
        step_indices = range(self.simulation.nsteps + 1)[self.step_request]

        all_states_shape = [len(trial_indices), len(step_indices)]
        all_states_shape.extend(state_shape)

        states = empty(all_states_shape)

        self.all_state = {'step_indices': step_indices,
                          'trial_indices': trial_indices,
                          'states': states}

        try:

            step_index = step_indices.index(0)
            initial_state = self.simulation.initial_state

            states[:, step_index, ...] = initial_state[trial_indices, ...]

        except ValueError:
            pass

    def do_intermediate_task(self):

        next_step = self.simulation.current_step + 1

        try:

            step_indices = self.all_state['step_indices']
            step_index = step_indices.index(next_step)

            next_state = self.simulation.next_state
            trial_indices = self.all_state['trial_indices']
            states = self.all_state['states']

            states[:, step_index, ...] = next_state[trial_indices, ...]

        except ValueError:
            pass

    def do_final_task(self):

        return self.all_state

class MeasureStepValue(SimProcedure):
    """Measurement that returns a value for a subset of trials.

    The trial_indices argument can take lists of indices (integer array
    indexing), slices, and numpy index expressions.
    """

    def __init__(self, get_value, output_name='all_value', step_request=s_[:], trial_request=s_[:]):
        self.get_val = get_value
        self.output_name = output_name
        self.step_request = step_request
        self.trial_request = trial_request
        


    def do_initial_task(self, simulation):

        self.simulation = simulation

        initial_val = self.get_val(self.simulation, self.trial_request)

        val_shape = shape(initial_val)

        step_indices = range(self.simulation.nsteps + 1)[self.step_request]


        all_val_shape = [len(step_indices), *val_shape]

        vals = empty(all_val_shape)

        vals[0, ...] = initial_val

        self.all_value = {'step_indices': step_indices, 'trial_indices': self.trial_request, 'values': vals}


    def do_intermediate_task(self):

        next_step = self.simulation.current_step + 1

        try:

            step_indices = self.all_value['step_indices']
            step_index = step_indices.index(next_step)

            try:
                next_value = self.get_val(self.simulation, self.trial_request)

                vals = self.all_value['values']

                vals[step_index, ...] = next_value

            except ValueError:
                print('shape fail')

        
        except ValueError:
            pass
        
    def do_final_task(self):

        return self.all_value

class MeasureMeanValue(MeasureStepValue):

    """Measurement that returns a value for a subset of trials.

    The trial_indices argument can take lists of indices (integer array
    indexing), slices, and numpy index expressions.
    """
    def __init__(self, get_value, output_name='all_value', step_request=s_[:], trial_request=s_[:], weights=None):
        self.get_val = lambda x,y: [average(get_value(x,y), axis=0, weights=weights), sem(get_value(x,y))] 
        self.output_name = output_name
        self.step_request = step_request
        self.trial_request = trial_request
    
    def do_final_task(self):
        self.all_value['std_error'] = self.all_value['values'][:,1,...]
        self.all_value['values'] = self.all_value['values'][:,0,...]
        
        return self.all_value


class MeasureAllStateDists(SimProcedure):
    """Records a running set of state histograms."""

    def __init__(self, bins, step_request=s_[:],
                 output_name='all_state_dists'):

        self.bins = bins
        self.step_request = step_request
        self.output_name = output_name

    def do_initial_task(self, simulation):

        self.simulation = simulation

        if self.bins is None:
            self.bins = simulation.initial_dist.bins

        step_indices = range(self.simulation.nsteps + 1)[self.step_request]
        hists = []

        self.all_dists = {'step_indices': step_indices, 'hists': hists}

        if 0 in step_indices:

            initial_state = simulation.initial_state
            bins = self.bins

            dist = histogramdd(initial_state, bins=bins)
            hists.append(dist)

    def do_intermediate_task(self):

        next_step = self.simulation.current_step + 1

        if next_step in self.all_dists['step_indices']:

            next_state = self.simulation.next_state
            bins = self.bins
            hists = self.all_dists['hists']

            dist = histogramdd(next_state, bins=bins)
            hists.append(dist)

    def do_final_task(self):

        return self.all_dists

# Created 2019-09-30 from OLD4_simulation.py
#
# Part of September 2019 refactor.

from math import sqrt
from numpy import array, empty
from random import Random, getrandbits

from infoenginessims.gregtools import Bunch


class Simulation:
    """The main simulation object.  Initializing runs a sim by default."""

    def __init__(self, update_state, procedures, nsteps, dt,
                 initial_state=None,
                 initial_dist=None, ntrials=None, sample_seed=None,
                 # dynamic=dynamic, system=None, protocol=None, infospace=None,
                 to_auto_run=False, nrandbits=32):
        """Initializes a simulation instance.

        Should provide at least one of initial_state and initial_dist.
        Providing both leaves initial_dist without any particular function.
        If only initial_dist is provided, should also provide ntrials.
        """

        if initial_state is not None:

            ntrials = initial_state.shape[0]

        else:  # Assume initial_dist and ntrials is given

            if sample_seed is None:
                sample_seed = getrandbits(nrandbits)

            initial_state = initial_dist.sample(ntrials, sample_seed)

            self.sample_seed = sample_seed

        if initial_dist is not None:
            self.initial_dist = initial_dist

        self.update_state = update_state
        self.procedures = procedures
        self.nsteps = nsteps
        self.dt = dt
        self.ntrials = ntrials
        self.initial_state = initial_state

        # if dynamic is not None:
        #     self.dynamic = dynamic
        # if system is not None:
        #     self.system = system
        # if protocol is not None:
        #     self.protocol = protocol
        # if infospace is not None:
        #     self.infospace = infospace

        if to_auto_run:
            self.output = self.run()

    def run(self, verbose=False):

        initial_state = self.initial_state
        update_state = self.update_state
        procedures = self.procedures
        nsteps = self.nsteps
        dt = self.dt
        self.current_time = 0

        self.current_state = array(initial_state)
        self.next_state = empty(initial_state.shape)

        current_state = self.current_state
        next_state = self.next_state

        # Do initial tasks
        for procedure in procedures:

            procedure.do_initial_task(self)

        # Main evolution loop
        for step in range(nsteps):

            time = step * dt

            self.current_step = step
            self.current_time = time

            next_state[...] = update_state(current_state, time, dt)

            # Do intermediate tasks
            for procedure in procedures:
                procedure.do_intermediate_task()

            current_state[...] = next_state[...]

            if verbose:
                print("\r step {} of {} complete".format(step, nsteps),end="")
                
                

        # Do final tasks and get outputs
        outputs = Bunch()
        for procedure in procedures:

            output = procedure.do_final_task()

            if hasattr(procedure, 'output_name'):
                setattr(outputs, procedure.output_name, output)

        return outputs


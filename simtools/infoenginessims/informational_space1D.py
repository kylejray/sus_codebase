import numpy as np
from numpy import full

from infoenginessims.gregtools import get_rough_minima_alt as get_minima


class InformationalSpace:
    """Has potential extrema and informational state space tools.

    Finds the informational states of a potential over position and time by
    finding the local minima and maxima and their bifurcations.

    Assumes that the informational states are regions of local minima
    in a 1D potential function of position and that there will always
    only be bifurcations of the local extrema due to pitchforks or
    saddle nodes exclusively and one at a time.

    Should possess:
        + Lists(?) of maxima and minima
        (Nixed) Lists(?) of informational states and their duration
        - method for getting informational state from a position and time
        - method for determining transition between two successive states

    Data:
        - key_times: list of continuous extrema block time transitions,
                     including initial and final times of the process
        - key_steps: list of continuous extrema block step transitions,
                     including initial and final times of the process
        - cont_extrema_blocks: list of continuous extrema.  Each is a
                               dictionary of minima_threads and maxima_threads,
                               each of those having a list for each thread,
                               made up of the extrema position and value as
                               tuples at all steps of the corresponding
                               continuous extrema block.

    Methods:
        - get_info_state: returns info state for a position and time
        - is_transition: returns truth value of a transition occurring between
                         two times and positions
    """

    def __init__(self, potential, substage_times, nsteps, xlim, nxsteps=1000,
                 initial_time=None, final_time=None):
        """Finds the key times, key steps, and continuous extrema blocks."""

        if initial_time is None:
            initial_time = substage_times[0]

        if final_time is None:
            final_time = substage_times[-1]

        duration = final_time - initial_time
        times = [duration * (step / nsteps) for step in range(nsteps + 1)]
        dt = duration / nsteps

        key_times = []
        key_steps = []
        cont_extrema_blocks = []

        n_minima_threads = []
        n_maxima_threads = []

        initial_minima = get_minima(lambda x: potential(x, initial_time),
                                    xlim, nxsteps)
        initial_maxima = get_minima(lambda x: -potential(x, initial_time),
                                    xlim, nxsteps)

        n_minima_threads.append(len(initial_minima))
        n_maxima_threads.append(len(initial_maxima))

        initial_minima_threads = []
        for minima in initial_minima:
            thread = []
            thread.append((minima, potential(minima, initial_time)))
            initial_minima_threads.append(thread)

        initial_maxima_threads = []
        for maxima in initial_maxima:
            thread = []
            thread.append((maxima, potential(maxima, initial_time)))
            initial_maxima_threads.append(thread)

        initial_cont_extrema = {'minima_threads': initial_minima_threads,
                                'maxima_threads': initial_maxima_threads}

        key_times.append(initial_time)
        key_steps.append(0)
        cont_extrema_blocks.append(initial_cont_extrema)

        current_minima_threads = initial_minima_threads
        current_maxima_threads = initial_maxima_threads

        for step, time in enumerate(times[1:]):

            current_minima = get_minima(lambda pos: potential(pos, time),
                                        xlim, nxsteps)
            current_maxima = get_minima(lambda pos: -potential(pos, time),
                                        xlim, nxsteps)

            if len(current_minima) != n_minima_threads[-1] \
                    or len(current_maxima) != n_maxima_threads[-1]:

                current_minima_threads = [[] for _ in \
                                            range(len(current_minima))]
                current_maxima_threads = [[] for _ in \
                                            range(len(current_maxima))]

                n_minima_threads.append(len(current_minima_threads))
                n_maxima_threads.append(len(current_maxima_threads))

                current_cont_extrema \
                    = {'minima_threads': current_minima_threads,
                       'maxima_threads': current_maxima_threads}

                key_times.append(time)
                key_steps.append(step + 1)
                cont_extrema_blocks.append(current_cont_extrema)

            for index, minima in enumerate(current_minima):
                thread = current_minima_threads[index]
                thread.append((minima, potential(minima, time)))

            for index, maxima in enumerate(current_maxima):
                thread = current_maxima_threads[index]
                thread.append((maxima, potential(maxima, time)))

        key_times.append(final_time)
        key_steps.append(nsteps)

        self.nsteps = nsteps
        self.times = times
        self.dt = dt

        self.xlim = xlim
        self.nxsteps = nxsteps

        self.initial_time = initial_time
        self.final_time = final_time

        self.key_times = key_times
        self.key_steps = key_steps
        self.cont_extrema_blocks = cont_extrema_blocks

        nss_info_states \
            = [n_minima_threads[self.get_block_index_from_time(time)] \
               for time in substage_times[:-1]]
        nss_info_states.append(n_minima_threads[-1])

        self.nss_info_states = nss_info_states


    def get_info_state(self, position, step=None,
                       block_index=None, block_step=None):
        """Returns the informational state for the position and step boundary.

        Associates an informational state for each local minima in the
        potential at each point in time.

        Labels the informational states at a time as 0 through n-1 where n is
        the number of informational states and ordered from lowest
        minima value through highest.
        """

        info_state = np.zeros(position.shape, int)

        if block_index is None:
            block_index = self.get_block_index_from_step(step)

        cont_extrema = self.cont_extrema_blocks[block_index]

        if len(cont_extrema['maxima_threads']) == 0:

            return info_state

        else:

            if block_step is None:
                block_step = self._get_block_step_from_step(step, block_index)

            maxima = cont_extrema['maxima_threads'][0][block_step][0]

            info_state[position > maxima] = 1

            return info_state

    def get_info_state_from_time(self, position, time, block_index=None):
        """Returns the informational state for the position and time.

        Associates an informational state for each local minima in the
        potential at each point in time.

        Labels the informational states at a time as 0 through n-1 where n is
        the number of informational states and ordered from lowest
        minima value through highest.
        """

        info_state = np.zeros(position.shape, int)

        if block_index is None:
            block_index = self.get_block_index_from_time(time)

        cont_extrema = self.cont_extrema_blocks[block_index]

        if len(cont_extrema['maxima_threads']) == 0:

            return info_state

        else:

            block_step = self._get_block_step_from_time(time, block_index)
            maxima = cont_extrema['maxima_threads'][0][block_step][0]
            # try:
            #     maxima = cont_extrema['maxima_threads'][0][block_step][0]
            # except IndexError:
            # nmax_threads = len(cont_extrema['maxima_threads'][0])
            # print(f'bs:\t{block_step}\t',
            #       f'time:\t{time:.4g}\t',
            #       f'bindx:\t{block_index}\t',
            #       f'nmaxt:\t{nmax_threads}\t',
            #       f'endbs:\t{self.key_steps[block_index + 1]}\t')
                # raise
            # return info_state

            info_state[position > maxima] = 1

            return info_state

    def is_transition(self, position_0, position_1, time_0, time_1):
        """Estimates whether a transition over a maxima occured.

        Returns false whenever the times are for different continous
        extrema blocks.
        """

        block_index_0 = self.get_block_index_from_time(time_0)
        block_index_1 = self.get_block_index_from_time(time_1)

        if block_index_0 != block_index_1:
            return False
        else:
            info_state_0 = self.get_info_state_from_time(position_0, time_0)
            info_state_1 = self.get_info_state_from_time(position_1, time_1)
            return info_state_0 != info_state_1

    def is_transition__improved(self, info_state_0, info_state_1,
                                block_index_0, block_index_1):
        """Estimates whether a transition over a maxima occured.

        Uses better arguments than is_transition."""

        if block_index_0 != block_index_1:
            # return False
            return full(info_state_0.shape, False)
        else:
            return info_state_0 != info_state_1

    def get_block_index_from_step(self, step):

        for index in range(len(self.key_steps)):
            if step < self.key_steps[index]:
                block_index = index - 1
                break
        else:
            block_index = len(self.key_steps) - 2

        return block_index

    def get_block_index_from_time(self, time: object) -> object:
        """Estimates the continuous extrema block of the time or step."""

        for index in range(len(self.key_times)):
            if time < self.key_times[index]:
                block_index = index - 1
                break
        else:
            block_index = len(self.key_times) - 2

        return block_index

    def get_ninfostates_at_time(self, time):

        block_index = self.get_block_index_from_time(time)

        ninfostates = len(self.cont_extrema_blocks[block_index]\
                            ['minima_threads'])

        return ninfostates

    def get_minimum(self, minimum_index, time=None,
                    block_index=None, block_step=None):

        if block_index is None:
            block_index = self.get_block_index_from_time(time)

        if block_step is None:
            block_step = self._get_block_step_from_time(time, block_index)

        pos_minimum = self.cont_extrema_blocks[block_index]['minima_threads']\
                            [minimum_index][block_step]

        return pos_minimum

    def get_maximum(self, maximum_index, time=None,
                    block_index=None, block_step=None):

        if block_index is None:
            block_index = self.get_block_index_from_time(time)

        if block_step is None:
            block_step = self._get_block_step_from_time(time, block_index)

        pos_maximum = self.cont_extrema_blocks[block_index]['maxima_threads']\
                            [maximum_index][block_step]

        return pos_maximum

    def _get_block_step_from_time(self, time, block_index=None):
        """Estimates the step in the block closest to the given time."""

        if block_index is None:
            block_index = self.get_block_index_from_time(time)

        block_start_step = self.key_steps[block_index]
        block_end_step = self.key_steps[block_index + 1]

        for block_step0, time0 \
                in enumerate(self.times[block_start_step: block_end_step]):
            if time < time0:
                block_step = block_step0 - 1
                break
        else:
            block_step = block_end_step - block_start_step - 1

        return block_step

        # step = round(self.nsteps * (time - self.initial_time)
        #                 / (self.final_time - self.initial_time))
        #
        # return step - self.key_steps[block_index]

    def _get_block_step_from_step(self, step, block_index=None):
        pass

    def _get_step_from_block_step(self, block_index, block_step):

        return self.key_steps[block_index] + block_step


import sys
import os

from numpy import empty, zeros, multiply

# from infoenginessims.simprocedures.basic_simprocedures import SimProcedure
from .basic_simprocedures import SimProcedure

is_path = os.path.expanduser('~/source/informational_states/')
sys.path.insert(0, is_path)

from measure import MeasurementDevice


class MeasureTrajectory(SimProcedure):
    """Keeps the current step's next value."""

    def __init__(self, output_name='trajectories', **kwargs):
        self.output_name = output_name

        kwargs['trajectory_mode'] = False
        self.measure = MeasurementDevice(**kwargs)

    def get_dict(self, state):
        _, bools = self.measure.apply(state)
        return self.measure.get_lookup(bools)
        

    def do_initial_task(self, simulation):

        self.simulation = simulation

        self.trajectory_dict = self.get_dict(simulation.initial_state)

    def do_intermediate_task(self):

        new_dict = self.get_dict(self.simulation.current_state)

        for key in self.trajectory_dict:
            self.trajectory_dict[key] = self.trajectory_dict[key] * new_dict[key]

    def do_final_task(self):
        return self.trajectory_dict


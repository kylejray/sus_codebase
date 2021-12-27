
import sys
import os

from numpy import empty, zeros, multiply, s_

sim_path = os.path.dirname(os.path.realpath(__file__))+"/../../../informational_states"
sys.path.append(sim_path)
from measure import MeasurementDevice

# from infoenginessims.simprocedures.basic_simprocedures import SimProcedure
from .basic_simprocedures import SimProcedure, MeasureMeanValue

'''
class MeasureTrajectories(SimProcedure):
    def __init__(self, SimProcedure, **traj_kwargs):
        traj_kwargs['trajectory_mode'] = False
        self.measure = MeasurementDevice(**traj_kwargs)

    def get_dict(self, state):
        _, bools = self.measure.apply(state)
        return self.measure.get_lookup(bools)
    
    def do_initial_task(self, simulation):
        self.trajectory_dict = {}
        for key in self.measure.outcome_names:
            self.trajectory_dict = 
'''

class AlwaysIn(SimProcedure):
    """Checks if always in the trajectory classes defined in **kwargs."""

    def __init__(self, output_name='trajectories', state_slice=s_[:], step_request=None, **kwargs):
        self.output_name = output_name
        self.state_slice = state_slice
        self.step_request=step_request

        kwargs['trajectory_mode'] = False
        self.measure = MeasurementDevice(**kwargs)

    def get_dict(self, state):
        _, bools = self.measure.apply(state[self.state_slice])
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

class CountJumps(AlwaysIn):
    """Returns how many jumps in and out of the trajectory classes defined in **kwargs."""
        

    def do_initial_task(self, simulation):

        self.simulation = simulation
        self.trajectory_dict = self.get_dict(simulation.initial_state)
        self.current_in_class = self.get_dict(simulation.initial_state)
        self.initial_class = self.get_dict(self.simulation.initial_state)

        self.step_indices=range(0)
        if self.step_request is not None:
            self.step_indices = range(self.simulation.nsteps + 1)[self.step_request]
            self.step_list=[]

    def do_intermediate_task(self):

        new_in_class = self.get_dict(self.simulation.current_state)

        for key in self.trajectory_dict:
            jump = (new_in_class[key] ^ self.current_in_class[key]).astype('int')
            self.trajectory_dict[key] = self.trajectory_dict[key] + jump
        self.current_in_class = new_in_class
        
        if self.simulation.current_step in self.step_indices:
            temp_dict={}
            for key in self.trajectory_dict:
                temp_dict[key] = self.trajectory_dict[key]
                temp_dict[key][~self.initial_class[key]]=False
            self.step_list.append(temp_dict)

    def do_final_task(self):
        for key in self.trajectory_dict:
            # self.trajectory_dict[key] -= 1
            self.trajectory_dict[key][~self.initial_class[key]] = False
        
        if self.step_request is not None:
            self.step_list.append(self.trajectory_dict)
            del(self.trajectory_dict)
            self.trajectory_dict={}
            self.trajectory_dict['step_list'] = self.step_list
            self.trajectory_dict['step_request'] = self.step_indices
        return self.trajectory_dict


        



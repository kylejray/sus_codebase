import os
import sys

from math import sqrt

sim_path = os.path.dirname(os.path.realpath(__file__))+"/simtools/infoenginessims/"
sys.path.append(sim_path)


from integrators import rkdeterm_eulerstoch
from dynamics import langevin_underdamped, langevin_overdamped
from simprocedures import basic_simprocedures as sp
from simprocedures import running_measurements as rp
from simulation import Simulation


def setup_sim(system, init_state, procedures=None, sim_params=None, dt=1/200, damping=1, temp=1, extra_time=1):
    '''
    returns a quick and dirty langevin Simulation object, overdamped if system.has_velocity is false, underdamped otherwise

    Parameters
    ----------
    system: instantiation of the System class
        this is where the simulation will pull the force and potential from
    init_state: ndarray of shape [N_trials, shape(state)]
        array representing the initial state of the trials
    procedures: list of SimProcs
        these are the procedures that will be done during the sim
    sim_params: list of length 3 for underdamped and 2 for overdamped
        gives the dimensionless simulation parameters, defaults to 1 for all parameter
    dt: float
        the sim dt value for the integrator
    damping: float
        meta parameter that changes just the amount of damping, in standard language it is often called gamma
    temp : float
        meta parameter that changes just the temperature
    extra time: float >= 1
        when neq 1, the sim runs beyond protocol end time. if set to 1.3, for example, it will run another 30% longer than the protocol duration
    Returns
    -------
    sim: instantiation of Simuation class
        bundles all the inputs into a usable simulation object

        


    '''


    if system.has_velocity:
        if sim_params is None:
            sim_params=[1.,1.,1.]

        
        gamma = sim_params[0] * damping
        theta = sim_params[1]
        eta = sim_params[2] * sqrt(damping) * sqrt(temp)

        dynamic = langevin_underdamped.LangevinUnderdamped(theta, gamma, eta,
                                                           system.get_external_force)

    else:
        if sim_params is None:
            sim_params=[1.,1.]

        omega = sim_params[0]
        xi = sim_params[1] * sqrt(temp) * sqrt(damping)
        dynamic = langevin_overdamped.LangevinOverdamped(omega, xi,
                                                         system.get_external_force)

    dynamic.mass = system.mass
    integrator = rkdeterm_eulerstoch.RKDetermEulerStoch(dynamic)

    if procedures is None:
        procedures = [
                    sp.ReturnFinalState(),
                    sp.MeasureAllState(trial_request=slice(0, 1000)),
                    rp.MeasureAllValue(rp.get_dW, 'all_W'), 
                    rp.MeasureFinalValue(rp.get_dW, 'final_W')]

    total_time = extra_time * (system.protocol.t_f-system.protocol.t_i)

    nsteps = int(total_time / dt)

    sim = Simulation(integrator.update_state, procedures, nsteps, dt,
                                initial_state=init_state)

    sim.system = system

    return sim

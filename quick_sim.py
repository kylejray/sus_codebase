import os
import sys

from math import sqrt

infoenginessims_path = os.path.expanduser("~/source/") + "infoenginessims/"
sys.path.insert(0, infoenginessims_path)

from infoenginessims.api import *
from infoenginessims.integrators import rkdeterm_eulerstoch
from infoenginessims.dynamics import langevin_underdamped, langevin_overdamped
from infoenginessims.state_distributions import sd_tools, state_distribution
from infoenginessims import simulation
from infoenginessims.simprocedures import basic_simprocedures as sp
from infoenginessims.simprocedures import running_measurements as rp
from infoenginessims import analysis
import infoenginessims.analysis.running_quantities


def setup_sim(system, initial_state, procedures=None, nsteps=1000, damping=1, temp=1, extra_time=1):

    if system.has_velocity:

        theta = 1.
        gamma = 1. * damping
        eta = 1. * sqrt(damping) * sqrt(temp)

        dynamic = langevin_underdamped.LangevinUnderdamped(theta, gamma, eta,
                                                           system.get_external_force)

    else:

        omega = 1.
        xi = 1.
        dynamic = langevin_overdamped.LangevinOverdamped(omega, xi,
                                                         system.get_external_force)

    integrator = rkdeterm_eulerstoch.RKDetermEulerStoch(dynamic)

    if procedures is None:
        procedures = [
                    sp.ReturnFinalState(),
                    sp.MeasureAllState(trial_request=slice(0, 1000)),
                    rp.MeasureAllValue(rp.get_dW, 'all_W'), 
                    rp.MeasureFinalValue(rp.get_dW, 'final_W')]

    total_time = extra_time * (system.protocol.t_f-system.protocol.t_i)

    dt = total_time / nsteps

    sim = simulation.Simulation(integrator.update_state, procedures, nsteps, dt,
                                initial_state)

    sim.system = system

    return sim

'''
def run_sim(sim, procedures):
    sim.output = sim.run()


t_finals=[]
mean=[]
std=[]
acc=[]
fail=[]
    
print(system.protocol.t_f, end ="\r")
has_velocity = True




final_W = sim.output.final_W
final_state=sim.output.final_state
    
a,f = szilard_accuracy(initial_state, final_state)
m=final_W.mean()
s=final_W.std()
    
t_finals.append(system.protocol.t_f)
mean.append(m)
std.append(s)
acc.append(a)
fail.append(f)
    
system.protocol.time_stretch(2)


mean=np.asarray(mean)
std=np.asarray(std)
t_finals=np.asarray(t_finals)
acc=np.asarray(acc)
fail=np.asarray(fail)
print("done")
'''
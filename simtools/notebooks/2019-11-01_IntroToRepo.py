# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Summary

# %% [markdown]
# Starting a new repo for collaboration with Kyle.  First I'm just adding the core code for doing basic simulations, including the big 1D protocols, 1D duffing system, and ND Langevin Dynamics.  Also including basic 2D harmonic oscillator system and simple protocols, previously just in another notebook.
#
# Will add more to this repo as further functionality is updated and cleaned up.

# %% [markdown]
# ---
# ## Imports for whole notebook

# %%

# %%
import sys
infoenginessims_path = ".."
sys.path.insert(0, infoenginessims_path)

from infoenginessims.api import *

from infoenginessims.integrators import rkdeterm_eulerstoch
from infoenginessims.dynamics import langevin_underdamped, langevin_overdamped
from infoenginessims.state_distributions import sd_tools, state_distribution

from infoenginessims import simulation
from infoenginessims.simprocedures import basic_simprocedures as sp
from infoenginessims.simprocedures import running_measurements as rp

from infoenginessims import analysis
import infoenginessims.analysis.state_dists_1D
import infoenginessims.analysis.running_quantities
import infoenginessims.analysis.hists_1D
import infoenginessims.analysis.protocols
import infoenginessims.analysis.animations


# %matplotlib inline

# %%
#kyles tools and systems#

protocol_designer_path = "../../protocol_designer/"
sys.path.insert(0, protocol_designer_path)

import bit_flip_protocols as bfp
quick_flip = bfp.bf_1D


import protocol_designer as prd
from protocol_designer import system as syst

# %%

# %% [markdown]
# ---
# ## 1D Example

# %%
import infoenginessims.analysis.infospace_1D

# %%
system = quick_flip
total_t = 100.

protocol = erasure_efficient.ErasureEfficient(total_t=total_t)

# %%
system.show_potential(0)

# %%
theta = 1.0
gamma = 1.0
eta = 1.0

state_shape = (2,)

mass = 1 / theta
kappa = eta**2 / (gamma * theta)

dynamic = langevin_underdamped.LangevinUnderdamped(
                theta, gamma, eta, system.get_external_force)

# %%
integrator = rkdeterm_eulerstoch.RKDetermEulerStoch(dynamic)

# %%
get_initial_potential = lambda x: system.potential.potential(x, system.protocol.get_params(0))


# %%

# %%
# %%time
initial_dist = sd_tools.make_1DUD_metastable_dist(
                    get_initial_potential, mass, kappa)

# %%
analysis.state_dists_1D.full_plot(initial_dist)

# %%
# alpha = 7.
# zeta = 3.

# system = duffing.Duffing(protocol, alpha, zeta, has_velocity=False)

# get_initial_potential = lambda x: system.get_potential(x, 0)

# omega = 1.0
# xi = 1.0

# state_shape = ()

# kappa = omega / (xi**2)

# dynamic = langevin_overdamped.LangevinOverdamped(omega, xi, system.get_external_force)

# initial_dist = sd_tools.make_1DOD_metastable_dist(get_initial_potential, kappa)

# plt.plot(initial_dist.bins[0][:-1], initial_dist.probs)

# %%
# %%time
ntrials = 50_000

initial_state = initial_dist.get_sample(ntrials)

# %%
np.shape(initial_state)

# %%
plt.hist(initial_state[25000:,:], bins=100);

# %%
procedures = [
              sp.ReturnFinalState(),
              sp.MeasureAllState(trial_request=slice(0, None)),  
              rp.MeasureAllValue(rp.get_dW, 'all_W'),
              rp.MeasureFinalValue(rp.get_dW, 'final_W'),
             ]

# %%
nsteps = 5_000

total_t = system.protocol.total_t

dt = total_t / nsteps

sim = simulation.Simulation(integrator.update_state, procedures, nsteps, dt,
                            initial_state, initial_dist, ntrials)

sim.system = system

# %%
# %%time

sim.output = sim.run(verbose=True)

# %%
all_state = sim.output.all_state['states']
all_W = sim.output.all_W
final_W = sim.output.final_W

# %%
all_state.shape

# %%
analysis.running_quantities.plot_running_quantity(all_state[:100,:,0])

# %%
analysis.running_quantities.plot_running_quantity(all_W[:100])

# %%
np.mean(final_W)

# %%
final_W_hist = np.histogram(final_W, bins=50)
analysis.hists_1D.plot_hist(final_W_hist, log=True)

# %%
infoenginessims_path = "../.."
sys.path.insert(0, infoenginessims_path)

from system_experiments import kyle_tools as kt

# %%
w, lr = kt.crooks_analysis_tsp(final_W, nbins=100)

# %%
np.mean(np.exp(-final_W))

# %%
lr

# %% [markdown]
# ---
# ### Example 2D System

# %%
from infoenginessims.protocols.simple_linear_protocol import SimpleLinearProtocol
from infoenginessims.systems.harmonic_2D import Harmonic2D

# %%
param_initial_vals = [1., 1., 0., 0., 0.]
param_final_vals = [1., 1., 1., 1., 1.]

total_t = 1000.

protocol = SimpleLinearProtocol(param_initial_vals, param_final_vals, total_t)

# %%
spring_const0_scale = 1.
spring_const1_scale = 1.
center0_scale = 1.
center1_scale = 1.
min_height_scale = 1.

has_velocity = True

system = Harmonic2D(protocol,
                    spring_const0_scale, spring_const1_scale,
                    center0_scale, center1_scale, min_height_scale,
                    has_velocity=has_velocity)

# %%
t = total_t * 1

nxbins = 100
X = np.linspace(-1.5, 1.5, nxbins)

U = np.empty(shape=(nxbins, nxbins))
F0 = np.empty(shape=(nxbins, nxbins))
F1 = np.empty(shape=(nxbins, nxbins))
for (i, x), (j, y) in product(enumerate(X), repeat=2):
    
    state = np.array([x, y])
    potential = system.get_potential(state, t, False)
    force = system.get_external_force(state, t, False)
#     print(force)
    
    U[i, j] = potential
    F0[i, j] = force[..., 0]
    F1[i, j] = force[..., 1]
    
extent = (X[0], X[1], X[0], X[1])

fig, ax = plt.subplots()
im = ax.imshow(U.transpose()[::-1,:], extent=extent)
colorbar = plt.colorbar(im)

fig, ax = plt.subplots()
im = ax.imshow(F0.transpose()[::-1,:], extent=extent)
colorbar = plt.colorbar(im)

fig, ax = plt.subplots()
im = ax.imshow(F1.transpose()[::-1,:], extent=extent)
colorbar = plt.colorbar(im)

# %%
if has_velocity:

    theta = 1.
    gamma = 1.
    eta = 1.

    dynamic = langevin_underdamped.LangevinUnderdamped(theta, gamma, eta,
                                                       system.get_external_force)
    
else:
    
    omega = 1.
    xi = 1.
    
    dynamic = langevin_overdamped.LangevinOverdamped(omega, xi,
                                                     system.get_external_force)

# %%
integrator = rkdeterm_eulerstoch.RKDetermEulerStoch(dynamic)

# %%
ntrials = 100
posvel_dim = 2 if has_velocity else 1

initial_state = np.zeros((ntrials, 2, posvel_dim))

# %%
procedures = [
              sp.ReturnFinalState(),
              sp.MeasureAllState(trial_request=slice(0, 100)),  
              rp.MeasureAllValue(rp.get_dW, 'all_W'),
              rp.MeasureFinalValue(rp.get_dW, 'final_W'),
             ]

# %%
nsteps = 10_000

total_time = system.protocol.total_time

dt = total_time / nsteps

sim = simulation.Simulation(integrator.update_state, procedures, nsteps, dt,
                            initial_state)

sim.system = system

# %%
# %%time

sim.output = sim.run()

# %%
all_state = sim.output.all_state['states']
all_W = sim.output.all_W
final_W = sim.output.final_W

# %%
all_state.shape

# %%
end_plot_time = total_time #* 1 / 100
trial_indices = np.s_[:100]

analysis.running_quantities.plot_running_quantity(all_state[trial_indices, :, 0, 0],
                                                  final_time=total_t,
                                                  end_plot_time=end_plot_time)

analysis.running_quantities.plot_running_quantity(all_state[trial_indices, :, 0, 1],
                                                  final_time=total_t,
                                                  end_plot_time=end_plot_time)

analysis.running_quantities.plot_running_quantity(all_state[trial_indices, :, 1, 0],
                                                  final_time=total_t,
                                                  end_plot_time=end_plot_time)

analysis.running_quantities.plot_running_quantity(all_state[trial_indices, :, 1, 1],
                                                  final_time=total_t,
                                                  end_plot_time=end_plot_time)

# %%
analysis.running_quantities.plot_running_quantity(all_W[trial_indices],
                                                  final_time=total_t,
                                                  end_plot_time=end_plot_time)

# %%
final_W_hist = np.histogram(final_W, bins=50)
analysis.hists_1D.plot_hist(final_W_hist, log=True)

# %%
analysis.animations

# %%

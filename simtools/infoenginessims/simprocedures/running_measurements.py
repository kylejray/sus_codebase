from numpy import empty, zeros, multiply, s_

# from infoenginessims.simprocedures.basic_simprocedures import SimProcedure
from .basic_simprocedures import SimProcedure


# -------- First, functions to get the appropriate values at each time

def get_current_state(simulation, trial_request=s_[:]):
    """just returns a subset of current state, based on request."""
    
    return simulation.current_state[trial_request]

def get_dW(simulation):
    """Gets step change in inclusive work."""

    time = simulation.current_time
    dt = simulation.dt
    get_potential = simulation.system.get_potential
    state = simulation.current_state

    dpotential = get_potential(state, time + dt) - get_potential(state, time)

    return dpotential

def get_kinetic(simulation, trial_request=s_[:]):
    """Gets step change in inclusive work."""

    get_KE = simulation.system.get_kinetic_energy

    state = simulation.current_state[trial_request]

    KE = get_KE(state)

    return KE

def get_potential(simulation, trial_request=s_[:]):
    """Gets step change in inclusive work."""

    t = simulation.current_time
    get_PE = simulation.system.get_potential
    state = simulation.current_state[trial_request]

    PE = get_PE(state, t)

    return PE

def get_EPT(simulation, trial_request=s_[:]):
    """Gets step change in inclusive work."""

    t = simulation.current_time
    get_force = simulation.system.get_external_force
    state = simulation.current_state[trial_request]
    F = get_force(state, t)

    if simulation.system.has_velocity:
        state = state[...,0]

    

    return multiply(state, -F)


def get_dW0(simulation):
    """Gets step change in a exclusive work wrt potential."""

    t = simulation.current_time
    get_potential = simulation.system.get_potential
    current_state = simulation.current_state
    next_state = simulation.next_state
    # current_position = simulation.current_state[:, 0]
    # next_position = simulation.next_state[:, 0]

    dpotential = get_potential(next_state, t) \
                 - get_potential(current_state, t)

    # dpotential = get_potential(next_position, t) \
    #               - get_potential(current_position, t)

    return -dpotential


def get_dW01(simulation):
    """Exclusive work for change in potential from initial."""

    t = simulation.current_time
    current_state = simulation.current_state
    next_state = simulation.next_state
    # current_position = simulation.current_state[:, 0]
    # next_position = simulation.next_state[:, 0]
    get_potential = simulation.system.get_potential

    val_1 = get_potential(next_state, t) - get_potential(next_state, 0)
    val_0 = get_potential(current_state, t) \
            - get_potential(current_state, 0)

    # val_1 = get_potential(next_position, t) - get_potential(next_position, 0)
    # val_0 = get_potential(current_position, t) \
    #          - get_potential(current_position, 0)

    dpotential = val_1 - val_0

    return -dpotential


def get_dWdrag(simulation):

    gamma = simulation.system.dynamic.gamma
    theta = simulation.system.dynamic.theta
    v_curr = simulation.current_state[:, 1]
    v_next = simulation.next_state[:, 1]

    avg_v2 = (v_curr**2 + v_next**2) / 2

    return -gamma / theta * avg_v2


def get_dQ(simulation):
    """Gets step change in heat absorbed from reservoir."""

    mass = simulation.system.dynamic.mass
    current_state = simulation.current_state
    next_state = simulation.next_state
    # current_position = simulation.current_state[:, 0]
    # next_position = simulation.next_state[:, 0]
    # current_velocity = simulation.current_state[:, 1]
    # next_velocity = simulation.next_state[:, 1]
    potential = simulation.system.potential
    current_time = simulation.current_time
    dt = simulation.dt

    next_time = current_time + dt

    # Poor estimate!
    # K_diff = 1 / theta * (next_velocity - current_velocity) \
    #           * current_velocity

    if simulation.system.has_velocity:
        K0 = 1/2 * mass * current_state[..., 1] ** 2
        K1 = 1/2 * mass * next_state[..., 1] ** 2
        K_diff = K1 - K0
    else:
        K_diff = 0

    V0 = potential(current_state, next_time)
    V1 = potential(next_state, next_time)
    V_diff = V1 - V0

    dQ = K_diff + V_diff
    return dQ


class KeepNextValue(SimProcedure):
    """Keeps the current step's next value."""

    def __init__(self, get_dvalue, output_name):

        self.get_dvalue = get_dvalue
        self.output_name = output_name

    def do_initial_task(self, simulation):

        self.simulation = simulation

        self.next_value = zeros(simulation.ntrials)

    def do_intermediate_task(self):

        dvalue = self.get_dvalue(self.simulation)

        self.next_value = self.next_value + dvalue

class MeasureAllValue(KeepNextValue):

    """Returns values for each step."""

    def do_initial_task(self, simulation):

        KeepNextValue.do_initial_task(self, simulation)

        self.all_value = empty((simulation.ntrials, simulation.nsteps + 1))

        self.all_value[:, 0] = self.next_value

    def do_intermediate_task(self):

        super().do_intermediate_task()

        step = self.simulation.current_step

        self.all_value[:, step + 1] = self.next_value
    
    def do_final_task(self):
        return self.all_value

class MeasureFinalValue(KeepNextValue):
    """Returns the final value."""

    def do_final_task(self):

        return self.next_value

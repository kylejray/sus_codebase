# Created 2019-09-23


from math import sqrt
from numpy import empty


class LangevinUnderdamped:
    """Nondimensional ND underdamped Langevin dynamics.

    Takes generic nondim langevin parameters and a force function of
    position and time.

    The equations of motion (with nondimensional variables and parameters)
    are the following:

    dx = v dt
    dv = (- theta partial_x g - gamma v) dt + eta sqrt(2) r() sqrt(dt)

    where g is the nondimensional potential, a function of position and time.

    These are derivable from the standard 1D underdamped Langevin dynamics
    (with dimensional variables and parameters):

    dx = v dt
    m dv = (- partial_x U - lambda v) dt + sqrt(2 kT lambda) r() sqrt(dt)

    when we assume that g = U / kBT.
    """

    def __init__(self, theta, gamma, eta, get_external_force):
        """Initialize a 1D nondim underdamped Langevin dynamics."""

        self.theta = theta
        self.gamma = gamma
        self.eta = eta
        self.get_external_force = get_external_force

    def get_determ_dsdt(self, state, time):

        theta = self.theta
        gamma = self.gamma
        get_external_force = self.get_external_force

        # position = state[..., 0]
        velocity = state[..., 1]

        determ_dxdt = velocity
        determ_dvdt = theta * get_external_force(state, time) - gamma * velocity

        determ_dsdt = empty(state.shape)
        determ_dsdt[..., 0] = determ_dxdt
        determ_dsdt[..., 1] = determ_dvdt

        return determ_dsdt

    def get_stoch_dsdt(self, state, time, rng):

        eta = self.eta

        # want random numbers for only the velocity values
        vel_rng_shape = state.shape[:-1]

        stoch_dxdt = 0
        stoch_dvdt = eta * sqrt(2) * rng.normal(size=vel_rng_shape)

        stoch_dsdt = empty(state.shape)
        stoch_dsdt[..., 0] = stoch_dxdt
        stoch_dsdt[..., 1] = stoch_dvdt

        return stoch_dsdt
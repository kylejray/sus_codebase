# Created 2019-10-02 as part of Sept refactor

from math import sqrt

class LangevinOverdamped:
    """Nondimensional 1D overdamped Langevin dynamics.

    Takes generic nondim overdamped langevin parameters and a force function
    of position and time.

    The equation of motion (with nondimensional variables and parameters)
    is the following:

    dx = - omega partial_x V dt + xi sqrt(2) r() sqrt(dt)

    where V is the nondimensional potential, a function of position and time.

    This is derivable from the standard 1D overdamped Langevin dynamics
    (with dimensional variables and parameters):

    gamma dx = - partial_x U dt + sqrt(2 kT lambda) r() sqrt(dt)

    when we assume that V = U / kBT.
    """

    def __init__(self, omega, xi, get_external_force):
        """Initialize a 1D nondim overdamped Langevin dynamics."""

        self.omega = omega
        self.xi = xi
        self.get_external_force = get_external_force

    def get_determ_dsdt(self, state, time):

        omega = self.omega
        get_external_force = self.get_external_force

        determ_dsdt = omega * get_external_force(state, time)
        # print('determ_dsdt', determ_dsdt)

        return determ_dsdt

    def get_stoch_dsdt(self, state, time, rng):

        xi = self.xi

        rng_shape = state.shape

        # print('state', state)

        stoch_dsdt = xi * sqrt(2) * rng.normal(size=rng_shape)

        # print('stoch_dsdt', stoch_dsdt)

        return stoch_dsdt
# Created 2019-09-23


from math import sqrt
from random import getrandbits
from numpy.random import RandomState


class RKDetermEulerStoch:
    """Runge Kutta Deterministic, Euler Stochastic Integrator."""

    # def __init__(self, rng=None, seed=None, nrandbits=32):
    #
    #     self.rng = rng
    #     self.seed = seed
    #     self.nrandbits = nrandbits
    #
    # def initial_task(self, dynamic):
    #
    #     rng = self.rng
    #     seed = self.seed
    #     nrandbits = self.nrandbits
    #
    #     if rng is None:
    #         if seed is None:
    #             seed = getrandbits(nrandbits)
    #         rng = Random(seed)
    #
    #     self.rng = rng
    #     self.seed = seed
    #
    #     self.get_determ_dsdt = dynamic.get_determ_dsdt
    #     self.get_stoch_dsdt = dynamic.get_stoch_dsdt

    def __init__(self, dynamic, rng=None, seed=None, nrandbits=32):

        if rng is None:
            if seed is None:
                seed = getrandbits(nrandbits)
            rng = RandomState(seed)

        self.seed = seed
        self.rng = rng
        self.get_determ_dsdt = dynamic.get_determ_dsdt
        self.get_stoch_dsdt = dynamic.get_stoch_dsdt

    def update_state(self, state, time, dt):

        get_determ_dsdt = self.get_determ_dsdt
        get_stoch_dsdt = self.get_stoch_dsdt
        rng = self.rng

        k1 = get_determ_dsdt(state,               time)
        k2 = get_determ_dsdt(state + k1 * dt / 2, time + dt / 2)
        k3 = get_determ_dsdt(state + k2 * dt / 2, time + dt / 2)
        k4 = get_determ_dsdt(state + k3 * dt,     time + dt)

        ds_determ = dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        ds_stoch = get_stoch_dsdt(state, time, rng) * sqrt(dt)

        next_state = state + ds_determ + ds_stoch

        return next_state


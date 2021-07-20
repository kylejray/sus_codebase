class ErasureEfficient:
    """Bechhoefer-like erasure protocol for a system, core version.

    This version only has the four non-constant substages, excluding initial
    and final constant substages."""

    def __init__(self, barrier_0=1., tilt_0=0., barrier_1=0., tilt_1=1.,
                 total_t=1., substage_times=None):
        """Bechhoefer-like erasure protocol for a system."""

        self.barrier_0 = barrier_0  # default barrier
        self.tilt_0 = tilt_0  # default slant
        self.barrier_1 = barrier_1  # maximum augmented barrier
        self.tilt_1 = tilt_1  # maximum augmented slant

        self.total_t = total_t


        if substage_times is None:
            substage_times = [(i / 4) * self.total_t for i in range(5)]
        self.substage_times = substage_times

        self.params = {'barrier': self.barrier, 'tilt': self.tilt}

    def tilt(self, t):
        """Control for the tilt (roughly anti-linear).

        The default slant is given by phix_min, which is 0 for true
        Bechhoefer erasure.  The maximum slant magnitude is given by
        phix_max.
        """

        y0 = self.tilt_0
        ydelta = self.tilt_1 - y0
        times = self.substage_times

        if t < times[1]:
            tilt_val = y0

        elif t < times[2]:
            tilt_val = y0 + ydelta * (t - times[1]) / (times[2] - times[1])

        elif t < times[3]:
            tilt_val = y0 + ydelta

        elif t < times[4]:
            tilt_val = y0 + ydelta * (times[4] - t) / (times[4] - times[3])

        else:
            tilt_val = y0

        return tilt_val

    def barrier(self, t):
        """Control for the barrier.

        The default barrier is given by phixdc_min.  The wells are maximally
        raised at phixdc_max, effectively lowering the barrier.  The barrier
        would then be zero when phixdc_max is 2 acos(1/BetaL)."""

        y0 = self.barrier_0
        ydelta = self.barrier_1 - y0
        times = self.substage_times

        if t < times[1]:
            barrier_val = y0 + ydelta * (t - times[0]) / (times[1] - times[0])

        elif t < times[2]:
            barrier_val = y0 + ydelta

        elif t < times[3]:
            barrier_val = y0 + ydelta * (times[3] - t) / (times[3] - times[2])

        elif t < times[4]:
            barrier_val = y0

        else:
            barrier_val = y0

        return barrier_val
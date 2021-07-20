class ErasureLandauer:
    """Bechhoefer-like erasure protocol for a system."""

    def __init__(self, barrier_val=1., tilt_0=0., tilt_1=1.,
                 total_t=1., substage_times=None):
        """Simple tilt then untilt erasure protocol for a system."""

        self.barrier_val = barrier_val  # default, consant barrier
        self.tilt_0 = tilt_0  # default slant
        self.tilt_1 = tilt_1  # maximum augmented slant

        self.total_t = total_t

        if substage_times is None:
            substage_times = [0.0]
            substage_times.extend([(1/8 + i / 4) * self.total_t \
                                   for i in range(4)])
            substage_times.append(total_t)
        self.substage_times = substage_times

        self.params = {'tilt': self.tilt}

    def tilt(self, t):
        """Control for the tilt."""

        y0 = self.tilt_0
        ydelta = self.tilt_1 - y0
        times = self.substage_times

        if t < times[1]:
            return y0

        if t < times[2]:
            return y0 + ydelta * (t - times[1]) / (times[2] - times[1])

        if t < times[3]:
            return y0 + ydelta

        if t < times[4]:
            return y0 + ydelta * (times[4] - t) / (times[4] - times[3])

        return y0

    def barrier(self, t):
        """Control for the barrier."""

        return self.barrier_val
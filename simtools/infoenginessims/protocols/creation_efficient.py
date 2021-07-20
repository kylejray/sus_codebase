class CreationEfficient:

    def __init__(self, barrier_0=1., tilt_0=0., barrier_1=0., tilt_1=1.,
                 total_t=1., substage_times=None):

        self.barrier_0 = barrier_0  # default barrier
        self.tilt_0 = tilt_0  # default slant
        self.barrier_1 = barrier_1  # maximum augmented barrier
        self.tilt_1 = tilt_1  # maximum augmented slant

        self.total_t = total_t

        if substage_times is None:
            substage_times = [(ssb / 4) * total_t for ssb in range(5)]
        self.substage_times = substage_times

        self.params = {'barrier': self.barrier, 'tilt': self.tilt}

    def barrier(self, t):
        """Control for the barrier."""

        y0 = self.barrier_0
        ydelta = self.barrier_1 - y0
        times = self.substage_times

        if t < times[1]:
            yval = y0

        elif t < times[2]:
            yval = y0 + ydelta * (t - times[1]) / (times[2] - times[1])

        elif t < times[3]:
            yval = y0 + ydelta

        elif t < times[4]:
            yval = y0 + ydelta * (times[4] - t) / (times[4] - times[3])

        else:
            yval = y0

        return yval

    def tilt(self, t):
        """Control for the tilt."""

        y0 = self.tilt_0
        ydelta = self.tilt_1 - y0
        times = self.substage_times

        # if t < times[0]:
        #     yval = y0

        if t < times[1]:
            yval = y0 + ydelta * (t - times[0]) / (times[1] - times[0])

        elif t < times[2]:
            yval = y0 + ydelta

        elif t < times[3]:
            yval = y0 + ydelta * (times[3] - t) / (times[3] - times[2])

        else:
            yval = y0

        return yval
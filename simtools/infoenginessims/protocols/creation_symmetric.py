class CreationSymmetric:

    def __init__(self, barrier_0=1., barrier_1=0., tilt_val=0., total_t=1.,
                 substage_times=None):

        self.barrier_0 = barrier_0
        self.barrier_1 = barrier_1
        self.tilt_val = tilt_val

        self.total_t = total_t

        if substage_times is None:
            substage_times = [0.0]
            substage_times.extend([(1/8 + i / 4) * total_t
                                   for i in range(4)])
            substage_times.append(total_t)
        self.substage_times = substage_times

        self.params = {'barrier': self.barrier}

    def barrier(self, t):

        y0 = self.barrier_0
        ydelta = self.barrier_1 - y0
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

    def tilt(self, t):

        return self.tilt_val


from numpy import empty, ndarray


class HarmonicWellPotential:

    def __init__(self, k):

        self.k = k

    def __call__(self, x, x_left, x_right):

        x_center = (x_left + x_right) / 2
        return self.k * ((x - x_center) ** 2 - (x_left - x_center) ** 2)


class HarmonicWellForce:

    def __init__(self, k):

        self.k = k

    def __call__(self, x, x_left, x_right):

        x_center = (x_left + x_right) / 2
        return - self.k * (x - x_center)


class FixedWells:

    def __init__(self, protocol, barrier_max, tilt_max,
                 well_width=0.95, barrier_width=0.1, wall_slope=50,
                 well_potential_bottom=HarmonicWellPotential(1),
                 well_force_bottom=HarmonicWellForce(1)):

        self.protocol = protocol

        self.barrier_max = barrier_max
        self.tilt_max = tilt_max

        self.well_width = well_width
        self.barrier_width = barrier_width
        self.wall_slope = wall_slope

        self.well_potential_bottom = well_potential_bottom
        self.well_force_bottom = well_force_bottom

    def get_potential(self, x, t):

        if type(x) is ndarray:
            return self.get_potential_array(x, t)

        if x < 0:
            return self.get_left_potential(x, t)
        else:
            return self.get_right_potential(x, t)

    def get_external_force(self, x, t):

        if type(x) is ndarray:
            return self.get_force_array(x, t)

        if x < 0:
            return self.get_left_force(x, t)
        else:
            return self.get_right_force(x, t)

    def get_potential_array(self, x, t):

        V = empty(x.size)

        V[x < 0] = self.get_left_potential_array(x[x < 0], t)
        V[x >= 0] = self.get_right_potential_array(x[x >= 0], t)

        return V

    def get_force_array(self, x, t):

        F = empty(x.size)

        F[x < 0] = self.get_left_force_array(x[x < 0], t)
        F[x >= 0] = self.get_right_force_array(x[x >= 0], t)

        return F

    def get_left_potential_array(self, x, t):

        V = empty(x.size)

        tilt_val = self.tilt_max * self.protocol.tilt(t)
        V_min = -tilt_val
        tilt_mag = abs(tilt_val)

        x_left = -self.barrier_width / 2 - self.well_width

        cond = (x < x_left)
        x_0 = x[cond]
        V[cond] = -self.wall_slope * (x_0 - x_left) + V_min

        x_right = -self.barrier_width / 2

        cond = ((x < x_right) & (x >= x_left))
        x_1 = x[cond]
        V[cond] = V_min + self.well_potential_bottom(x_1, x_left, x_right)

        barrier_val = self.barrier_max * self.protocol.barrier(t) + tilt_mag
        barrier_slope = (barrier_val - V_min) / (-x_right)

        cond = (x >= x_right)
        x_2 = x[cond]
        V[cond] = (x_2 - x_right) * barrier_slope + V_min

        return V

    def get_left_force_array(self, x, t):

        F = empty(x.size)

        x_left = -self.barrier_width / 2 - self.well_width

        cond = (x < x_left)
        F[cond] = self.wall_slope

        x_right = -self.barrier_width / 2

        cond = ((x < x_right) & (x >= x_left))
        x_1 = x[cond]
        F[cond] = self.well_force_bottom(x_1, x_left, x_right)

        tilt_val = self.tilt_max * self.protocol.tilt(t)
        V_min = -tilt_val
        tilt_mag = abs(tilt_val)
        barrier_val = self.barrier_max * self.protocol.barrier(t) + tilt_mag
        barrier_slope = (barrier_val - V_min) / (-x_right)

        cond = ((x >= x_right))
        F[cond] = -barrier_slope

        return F

    def get_right_potential_array(self, x, t):

        V = empty(x.size)

        tilt_val = self.tilt_max * self.protocol.tilt(t)
        V_min = tilt_val
        tilt_mag = abs(tilt_val)

        x_right = self.barrier_width / 2 + self.well_width

        cond = (x > x_right)
        x_0 = x[cond]
        V[cond] = self.wall_slope * (x_0 - x_right) + V_min

        x_left = self.barrier_width / 2

        cond = ((x > x_left) & (x <= x_right))
        x_1 = x[cond]
        V[cond] = V_min + self.well_potential_bottom(x_1, x_left, x_right)

        barrier_val = self.barrier_max * self.protocol.barrier(t) + tilt_mag
        barrier_slope = -(barrier_val - V_min) / (x_left)

        cond = (x <= x_left)
        x_2 = x[cond]
        V[cond] = (x_2 - x_left) * barrier_slope + V_min

        return V

    def get_right_force_array(self, x, t):

        F = empty(x.size)

        x_right = self.barrier_width / 2 + self.well_width

        cond = (x > x_right)
        F[cond] = -self.wall_slope

        x_left = self.barrier_width / 2

        cond = ((x > x_left) & (x <= x_right))
        x_1 = x[cond]
        F[cond] = self.well_force_bottom(x_1, x_left, x_right)

        tilt_val = self.tilt_max * self.protocol.tilt(t)
        V_min = tilt_val
        tilt_mag = abs(tilt_val)
        barrier_val = self.barrier_max * self.protocol.barrier(t) + tilt_mag
        barrier_slope = -(barrier_val - V_min) / (x_left)

        cond = (x <= x_left)
        F[cond] = -barrier_slope

        return F


    def get_left_potential(self, x, t):

        tilt_val = self.tilt_max * self.protocol.tilt(t)
        V_min = -tilt_val
        tilt_mag = abs(tilt_val)

        x_left = -self.barrier_width / 2 - self.well_width
        if x < x_left:
            return -self.wall_slope*(x - x_left) + V_min

        x_right = -self.barrier_width / 2
        if x < x_right:
            return V_min + self.well_potential_bottom(x, x_left, x_right)

        barrier_val = self.barrier_max * self.protocol.barrier(t) + tilt_mag
        barrier_slope = (barrier_val - V_min) / (-x_right)

        return (x - x_right) * barrier_slope + V_min

    def get_left_force(self, x, t):

        x_left = -self.barrier_width / 2 - self.well_width
        if x < x_left:
            return self.wall_slope

        x_right = -self.barrier_width / 2
        if x < x_right:
            return self.well_potential_bottom(x, x_left, x_right)

        tilt_val = self.tilt_max * self.protocol.tilt(t)
        V_min = -tilt_val
        tilt_mag = abs(tilt_val)
        barrier_val = self.barrier_max * self.protocol.barrier(t) + tilt_mag
        barrier_slope = (barrier_val - V_min) / (-x_right)

        return -barrier_slope

    def get_right_potential(self, x, t):

        tilt_val = self.tilt_max * self.protocol.tilt(t)
        V_min = tilt_val
        tilt_mag = abs(tilt_val)

        x_right = self.barrier_width / 2 + self.well_width
        if x > x_right:
            return self.wall_slope*(x - x_right) + V_min

        x_left = self.barrier_width / 2
        if x > x_left:
            return V_min + self.well_potential_bottom(x, x_left, x_right)

        barrier_val = self.barrier_max * self.protocol.barrier(t) + tilt_mag
        barrier_slope = -(barrier_val - V_min) / x_left

        return (x - x_left) * barrier_slope + V_min

    def get_right_force(self, x, t):

        x_right = self.barrier_width / 2 + self.well_width
        if x > x_right:
            return self.wall_slope

        x_left = self.barrier_width / 2
        if x > x_left:
            return self.well_potential_bottom(x, x_left, x_right)

        tilt_val = self.tilt_max * self.protocol.tilt(t)
        V_min = tilt_val
        tilt_mag = abs(tilt_val)
        barrier_val = self.barrier_max * self.protocol.barrier(t) + tilt_mag
        barrier_slope = -(barrier_val - V_min) / (x_left)

        return -barrier_slope
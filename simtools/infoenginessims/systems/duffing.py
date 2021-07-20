from math import sqrt


def get_widthheight(alpha, beta):
    '''Returns tuple of width and height given alpha and beta of double-well.

    width = sqrt(-Beta / (2 Alpha))
    height = Beta**2 / (4 Alpha)
    '''
    return sqrt(-beta / (2 * alpha)), beta ** 2 / (4 * alpha)


def get_alphabeta(width, height):
    '''Returns tuple of alpha and beta given width and height of double-well.

    alpha = h / w**4
    beta = -2 h / w**2
    '''
    return height / width**4, -2 * height / width**2


class Duffing:
    """Standard Duffing type nondimensional system.

    Assumes the following nondimensional potential:

    U_nd(x, t) = alpha (x^4 - 2 b_f(t) x^2) + zeta c_f(t) x ,

    where all variables and params are dimensionless.
    It is derivable from the following dimensional potential:

    U(x, t) = a x^4 + b_0 b_f(t) x^2 + c_0 c_f(t) x ,

    where all variables and parameters are dimensional.
    To do so, we use the following definitions:

    w = sqrt(-b_0 / 2a)
    h = b_0^2 / 4a
    x_c = w
    alpha = aw^4 / kBT
    zeta = c_0 / aw^3
    U_nd = U / kBT
    """

    def __init__(self, protocol, alpha, zeta, has_velocity=True):

        self.protocol = protocol

        self.alpha = alpha
        self.zeta = zeta

        self.has_velocity = has_velocity

    def get_potential(self, state, time, has_velocity=None):

        b_f = self.protocol.barrier(time)
        c_f = self.protocol.tilt(time)

        if has_velocity is None:
            has_velocity = self.has_velocity

        x = state[..., 0] if has_velocity else state

        potential = ((x * x - 2 * b_f) * x * self.alpha + self.zeta * c_f) * x

        return potential

    def get_external_force(self, state, time, has_velocity=None):

        b_f = self.protocol.barrier(time)
        c_f = self.protocol.tilt(time)

        if has_velocity is None:
            has_velocity = self.has_velocity

        x = state[..., 0] if has_velocity else state

        potential_force = - ((4 * x * x - 4 * b_f) * x * self.alpha \
                             + self.zeta * c_f)

        return potential_force

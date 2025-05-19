from ..protocol_designer import Potential
import numpy as np

def ubg_potential(x, y, params):
    kx, ky, u = params
    return .5 * (kx * x**2 + ky * y**2) + u*x*y


def ubg_force(x, y, params):
    kx, ky, u = params
    dx = kx*x + u*y
    dy = ky*y + u*x
    return (-dx, -dy)

ubg_defaults = [2,2,1]

gyrator_potential = Potential(ubg_potential, ubg_force, 3, 2, default_params = ubg_defaults )

import scipy
from scipy.stats import multivariate_normal


def gyrator_ness(k: float, u: float, m: float, gamma: float, T1: float, T2: float) -> scipy.stats.multivariate_normal:
    '''
    gives the distribution as a function of time assuming gamma=kBT=mass=1
    form PRE "Inertial effects on Brownan gyrator" Bae 2021
    '''
    c1 = 2*k**2*gamma**2
    c2 = k*gamma**2 + u**2*m
    c3 = T1+T2
    c4 = T2-T1
    c5 = k**2-u**2

    C11 = (c1*T1 + u**2 * (k*m*c3 + gamma**2*c4)) / (c5*c2)
    C22 = (c1*T2 + u**2 * (k*m*c3 - gamma**2*c4)) / (c2*c5)
    C33 = ( (c1/k)*T1 + u**2*m*c3) / (m*c2)
    C44 = ( (c1/k)*T2 + u**2*m*c3) / (m*c2)
    C12 = C21 = -u*c3 / c5
    C14 = C41 =  u*gamma*c5 / c2
    C23 = C32 = -u*gamma*c4 / c2
    
    cov = np.array( [ [C11, C12, 0, C14], [C21, C22, C23, 0],[0, C32, C33, 0], [C41, 0, 0, C44] ] ) / 2

    mean = np.zeros(4)

    return  multivariate_normal(mean, cov, allow_singular=True)
from ..protocol_designer import Potential, Protocol, Compound_Protocol
import numpy as np

from scipy.interpolate import UnivariateSpline



def ew_1D(x, depth, x_0, localization):
    return -depth * np.exp( -localization * (x-x_0)**2 )
def ew_1D_deriv(x, depth, x_0, localization):
    return 2 * depth * (x-x_0) * localization * np.exp( -localization * (x-x_0)**2 )
def symm_quartic_1D(x, depth, x_0):
    a, b = depth/x_0**4, 2*depth/x_0**2
    return a*x**4 - b*x**2
def symm_quartic_deriv(x, depth, x_0):
    a, b = depth/x_0**4, 2*depth/x_0**2
    return 4*a*x**3 - 2 * b * x
'''
 WIP
def quart_gauss_potential(x, params):
    params = depth, x_0, quart_weight
    local = 20

    quartic = symm_quart(x, depth, x_0)

    gauss = ew_1D(x, depth, x_0, local) + ew_1D(ew_1D(x, depth, -x_0, local)
'''

'''    
def odw_gaussian_pot(x, params):
    x_0, x_1, d_0, d_1, local_0, local_1 = params

    z_gauss = ew_1D(x, d_0, x_0, local_0)
    o_gauss = ew_1D(x, d_1, x_1, local_1)

    x_m, x_p = x_0-1/np.sqrt(2*local_0), x_1+1/np.sqrt(2*local_1)
    slope_0, slope_1 = -d_0 * np.sqrt(2*local_0/ np.e), d_1 * np.sqrt(2*local_1/ np.e)
    y_0, y_1 = -d_0/np.sqrt(np.e), -d_1/np.sqrt(np.e)

    s_values = [[slope_0, x_m, y_0], [slope_1, x_p, y_1]]
    
    linear_stab = [ item[0]*(x-item[1]) + item[2] for item in s_values]

    stability = np.heaviside(-x+x_m, 0)*linear_stab[0] + np.heaviside(x-x_p, 0)*linear_stab[1]


    U = np.heaviside(x-x_m, 0) * np.heaviside(-x+x_p, 0)*(z_gauss+o_gauss) + stability

    return U

def odw_gaussian_force(x, params):
    x_0, x_1, d_0, d_1, local_0, local_1 = params

    z_gauss = ew_1D_deriv(x, d_0, x_0, local_0)
    o_gauss = ew_1D_deriv(x, d_1, x_1, local_1)

    x_m, x_p = x_0-1/np.sqrt(2*local_0), x_1+1/np.sqrt(2*local_1)
    slope_0, slope_1 = -d_0 * np.sqrt(2*local_0/ np.e), d_1 * np.sqrt(2*local_1/ np.e)
    y_0, y_1 = -d_0/np.sqrt(np.e), -d_1/np.sqrt(np.e)

    s_values = [[slope_0, x_m, y_0], [slope_1, x_p, y_1]]
    
    linear_stab_deriv = [ item[0] for item in s_values]

    stability = np.heaviside(-x+x_m, 0)*linear_stab_deriv[0] + np.heaviside(x-x_p, 0)*linear_stab_deriv[1]

    d_U = np.heaviside(x-x_m, 0) * np.heaviside(-x+x_p, 0)*(z_gauss+o_gauss) + stability

    return -d_U

odwg_params = (-1., 1., 1., 1., 5., 5.)
odwg_domain = [[-5.],[5.]]

odw_gaussian = Potential(odw_gaussian_pot, odw_gaussian_force, 6, 1, default_params=odwg_params, relevant_domain=odwg_domain)

'''

def linear_stability(x, params, deriv=False):
    x_0, L, d = params
    xc = x_0+np.sign(x_0)/np.sqrt(2*L)
    slope= d * np.sqrt(2*L/ np.e) * np.sign(x_0)
    y_0 = -d/np.sqrt(np.e)
    if deriv:
        return slope
    else:
        return slope*(x-xc) + y_0


def quartic_stability(x, params, deriv=False):
    x_0, L, d = params
    a = L**2/np.sqrt(np.e)
    c = 5/(4*np.sqrt(np.e))
    if deriv:
        return 4*d*a*(x-x_0)**3
    else:
        return d*(a*(x-x_0)**4 - c)

def odw_gaussian_pot(x, params):

    x_0, x_1, d_0, d_1, local_0, local_1 = params

    z_gauss = ew_1D(x, d_0, x_0, local_0)
    o_gauss = ew_1D(x, d_1, x_1, local_1)

    x_m, x_p = x_0-1/np.sqrt(2*local_0), x_1+1/np.sqrt(2*local_1)
    s_values = [[x_0, local_0, d_0], [x_1, local_1, d_1]]
    
    stab = [ quartic_stability(x, item)  for item in s_values]

    stability = np.heaviside(-x+x_m, 0)*stab[0] + np.heaviside(x-x_p, 0)*stab[1]

    U = np.heaviside(x-x_m, 0) * np.heaviside(-x+x_p, 0)*(z_gauss+o_gauss) + stability

    return U

def odw_gaussian_force(x, params):
    x_0, x_1, d_0, d_1, local_0, local_1 = params

    z_gauss = ew_1D_deriv(x, d_0, x_0, local_0)
    o_gauss = ew_1D_deriv(x, d_1, x_1, local_1)

    x_m, x_p = x_0-1/np.sqrt(2*local_0), x_1+1/np.sqrt(2*local_1)
    s_values = [[x_0, local_0, d_0], [x_1, local_1, d_1]]
    
    stab = [ quartic_stability(x, item, deriv=True)  for item in s_values]

    stability = np.heaviside(-x+x_m, 0)*stab[0] + np.heaviside(x-x_p, 0)*stab[1]

    d_U = np.heaviside(x-x_m, 0) * np.heaviside(-x+x_p, 0)*(z_gauss+o_gauss) + stability

    return -d_U

odwg_params = (-1., 1., 1., 1., 5., 5.)
odwg_domain = [[-5.],[5.]]

odw_gaussian = Potential(odw_gaussian_pot, odw_gaussian_force, 6, 1, default_params=odwg_params, relevant_domain=odwg_domain)

def lintilt_gaussian_pot(x, params):

    x_0, x_1, d_0, d_1, local_0, local_1, tilt = params

    slope = tilt / (x_1-x_0)
  

    z_gauss = ew_1D(x, d_0, x_0, local_0)
    o_gauss = ew_1D(x, d_1, x_1, local_1)

    x_m, x_p = x_0-1/np.sqrt(2*local_0), x_1+1/np.sqrt(2*local_1)
    s_values = [[x_0, local_0, d_0], [x_1, local_1, d_1]]
    
    stab = [ quartic_stability(x, item)  for item in s_values]

    stability = np.heaviside(-x+x_m, 0)*stab[0] + np.heaviside(x-x_p, 0)*stab[1]

    U = -slope*x + np.heaviside(x-x_m, 0) * np.heaviside(-x+x_p, 0)*(z_gauss+o_gauss) + stability

    return U

def lintilt_gaussian_force(x, params):
    x_0, x_1, d_0, d_1, local_0, local_1, tilt = params

    slope = tilt / (x_1-x_0)
    d_0 = np.mean([d_0,d_1])
    d_1 = d_0

    z_gauss = ew_1D_deriv(x, d_0, x_0, local_0)
    o_gauss = ew_1D_deriv(x, d_1, x_1, local_1)

    x_m, x_p = x_0-1/np.sqrt(2*local_0), x_1+1/np.sqrt(2*local_1)
    s_values = [[x_0, local_0, d_0], [x_1, local_1, d_1]]
    
    stab = [ quartic_stability(x, item, deriv=True)  for item in s_values]

    stability = np.heaviside(-x+x_m, 0)*stab[0] + np.heaviside(x-x_p, 0)*stab[1]

    d_U = -slope + np.heaviside(x-x_m, 0) * np.heaviside(-x+x_p, 0)*(z_gauss+o_gauss) + stability

    return -d_U

ltg_params = (-1., 1., 1., 1., 5., 5., 0)
ltg_domain = [[-3.],[3.]]

lintilt_gaussian = Potential(lintilt_gaussian_pot, lintilt_gaussian_force, 7, 1, default_params=ltg_params, relevant_domain=ltg_domain)


def symm_quart_exp_potential(x, params):
    '''
    potential that interpolates between quartic and exponential 1d confinement
    '''
    local = 20
    a, b, quart_weight= params

    
    depth = b**2 / (4*a)
    location = np.sqrt(b/(2*a))

    quart = a*x**4 - b*x**2

    stability = a*x**4
    s_depth= a*location**4
    exponential = ew_1D(x, depth+.05*s_depth, location, local) + ew_1D(x, depth+.05*s_depth, -location, local)+ .05*stability
    U = quart_weight * quart + (1-quart_weight) * exponential + depth
    return U

def symm_quart_exp_force(x,params):
    '''
    potential that interpolates between quartic and exponential 1d confinement
    '''
    a, b, quart_weight= params
    local=20

    d_quart = 4*a*x**3 - 2*b*x
    depth = b**2 / (4*a)
    location = np.sqrt(b/(2*a))
    d_stability = 4*a*x**3
    s_depth= a*location**4
    d_exponential = ew_1D_deriv(x, depth+.05*s_depth, location, local) + ew_1D_deriv(x, depth+.05*s_depth, -location, local) + .05*d_stability
    d = quart_weight * d_quart + (1-quart_weight) * d_exponential
    return -d

default = (2., 4., 1.)
dom = [[-3.],[3.]]

symm_quart_exp_pot = Potential(symm_quart_exp_potential, symm_quart_exp_force, 3, 1, default_params=default, relevant_domain=dom)


prm = np.zeros((3,2))
prm[:, 0] = symm_quart_exp_pot.default_params
prm[:, 1] = symm_quart_exp_pot.default_params
prm[2,1] = 0


t=(0,1)

q_to_g_prot = Protocol(t, prm)




def pwl_double_well(params):
    k = 100
    dx=.05
    def x_points(params):
        x0, x1, d0, d1, w0, w1, _ = params
        left_well = [x0-w0/2-dx, x0-w0/2, x0+w0/2, (x0+w0/2)*.975]
        right_well = [(x1-w1/2)*.975, x1-w1/2, x1+w1/2, x1+w1/2+dx]
        return left_well + right_well

    def y_points(params):
        x0, x1, d0, d1, w0, w1, tilt = params
        slope=tilt/(x1-x0)
        if slope==0:
            left_well = [d0+k*dx, -d0, -d0, 0 ]
            right_well = [d1+k*dx, -d1, -d1, 0 ][::-1]
        else:
            left_well = [d0+k*dx, slope*w0/2, -slope*w0/2, -slope*(w0/2+dx) ]
            right_well = [-(x1-x0-w1/2-dx)*slope, -(x1-x0-w1/2-dx)*slope-d1, -(x1-x0-w1/2-dx)*slope-d1, d1+k*dx  ]
        return left_well + right_well

    return UnivariateSpline(x_points(params), y_points(params), k=1, s=0)

def pwl_dw_potential(x, params):
    U = pwl_double_well(params)
    return U(x)

def pwl_dw_force(x, params):
    U = pwl_double_well(params)
    dU = U.derivative()
    return -dU(x)


pwl_default = [-1, 1, 1, 1, .5, .5, 0]
pwl_dom = [[-3.],[3.]]

pwl_double_pot = Potential(pwl_dw_potential, pwl_dw_force, 7, 1, default_params=pwl_default, relevant_domain=pwl_dom)





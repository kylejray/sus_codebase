from ..protocol_designer import Potential, Protocol, Compound_Protocol
import numpy as np

#first we defined the potential

def three_well_periodic_potential(x, params):
    depth,tilt,asym, *_ = params

    U = (-depth*np.cos(2*np.pi*x)+ depth - tilt*x) - asym*np.cos(2*np.pi*x/3)
    return U

def three_well_periodic_force(x, params):
    depth, tilt, asym, *_ = params
    d_U1 = 2*np.pi*depth*np.sin(2*np.pi*x) - tilt
    d_U2 = asym * (np.pi/3) * np.sin(2*np.pi*x/3)

    F = -d_U1 - d_U2 
    return F

def three_well_periodic_repulsive_force(x, params):
    depth, tilt, asym, str = params
    d_U1 = 2*np.pi*depth*np.sin(2*np.pi*x) - tilt
    d_U2 = asym * (np.pi/3) * np.sin(2*np.pi*x/3)

    F_MB = get_interaction(x, bins=[-1.5, -.5, .5, 1.5], strength=str*np.ones(3))
    F = -d_U1 - d_U2
    F += F_MB
    return F

def get_interaction(x, bins, strength):
    assert len(strength) == len(bins)-1
    min, max = np.min(bins), np.max(bins)
    periodic_x = (x-min)%(max-min)+min
    counts = np.histogram(periodic_x, bins=bins)[0]
    prob = counts/sum(counts)
    f_diff = np.diff(np.append(prob,prob[0]))
    r_diff = np.diff(np.append(prob[::-1],prob[-1]))[::-1]
    bias = (r_diff+f_diff)/2
    repulsion = np.zeros(np.shape(periodic_x))
    for i in range(len(bins[:-1])):
        repulsion += - strength[i] * rect(periodic_x, bins[i], bins[i+1]) * bias[i]

    return repulsion



#domain
defaults = [1.,1.,0.]
mb_defaults = [1.,1.,0.,1.]

dom = [[-1.5],[1.5]]


three_well_periodic = Potential(three_well_periodic_potential, three_well_periodic_force, 3, 1, relevant_domain=dom, default_params=defaults)

three_well_periodic_mb = Potential(three_well_periodic_potential, three_well_periodic_repulsive_force, 4, 1, relevant_domain=dom, default_params=mb_defaults)

#bimodal attempt
def quartic(var, barrier, location, mode='potential'):
    a = barrier / location**4
    b = 2*barrier / location**2 
    if mode is 'potential':
        return a*var**4 - b*var**2
    if mode is 'force':
        return -4*a*var**3 + 2*b*var 


def bimodal_potential(x, y, params):
    depth, loc, _, _ = params
    return quartic(y, depth, loc)

def bimodal_force(x, y, params):
    depth, loc, z_tilt, o_tilt = params
    x_force = np.heaviside(y,0)*z_tilt + np.heaviside(-y,0)*o_tilt
    y_force = quartic(y, depth, loc, mode='force')
    return x_force, y_force

b_dom = [[-1.,-3.],[1.,3.]]
b_defaults = [1., 1., 1., 1.]
bimodal = Potential(bimodal_potential, bimodal_force, 4, 2, relevant_domain=b_dom, default_params=b_defaults)

#PWL
def piecewise_linear(x, params):
    k0, k1, *lims = params
    min = np.min(lims)
    max = np.max(lims)
    periodic_x = (x-min)%(max-min)+min
    U = -periodic_x * k0*rect(periodic_x, lims[0], lims[1])
    U += -periodic_x * k1*rect(periodic_x, lims[1], lims[2])
    return U

def piecewise_linear_force(x, params):
    k0, k1, *lims = params
    min = np.min(lims)
    max = np.max(lims)
    periodic_x = (x-min)%(max-min)+min
    F = k0*rect(periodic_x, lims[0], lims[1])
    F += k1*rect(periodic_x, lims[1], lims[2])
    return F

def rect(x, min, max):
    return np.heaviside(x-min,0)*np.heaviside(max-x,0)

pwl_dom = [[-1.5],[1.5]]
pwl_defaults = [1,.5,-1.5,0,1.5]

pwl = Potential(piecewise_linear, piecewise_linear_force, 5, 1, relevant_domain=pwl_dom, default_params=pwl_defaults)

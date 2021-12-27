from ..protocol_designer import Potential, Protocol, Compound_Protocol
import numpy as np

###### SINGLE FLUX QUBIT######

#first we defined the potential

def three_well_periodic_potential(x, params):
    depth,tilt = params
    U = -depth*np.cos(2*np.pi*x)+ depth - tilt*x
    return U

def three_well_periodic_force(x, params):
    depth, tilt = params
    F = -2*np.pi*depth*np.sin(2*np.pi*x) + tilt
    return F


#domain
defaults = [1,1]
dom = [[-1.5],[1.5]]


three_well_periodic = Potential(three_well_periodic_potential, three_well_periodic_force, 2, 1, relevant_domain=dom, default_params=defaults)
#once the potential is done, we make some simple one-time-step protocols, that will serves as buulding blocks

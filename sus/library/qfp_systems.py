from ..protocol_designer import Potential, Protocol, Compound_Protocol
import numpy as np

###### SINGLE QFP######

#first we defined the potential, from takeuchi 2013

# in units of Ej = I_c phi_0 / 2*pi

# EJ = IC * alpha
# betaQ = Lq * Ic / alpha
# betaL = L * Ic / alpha
# U_0 = alpha**2 / Lq

# Ej/betaQ = alpha**2 / Lq
# Ej/(betaL+2betaQ) = alpha^2 / Lq( L/Lq+2 )
# 2 * Ej = IC * alpha
# U -> U * E_j/U_o

def qfp_pot(phi_p,phi_m, params):
    phi_in, phi_x, betaQ, betaL = params
    U = (1/(betaL+2*betaQ))*(phi_p-phi_in)**2 + (1/betaL)*(phi_m-phi_x)**2 - 2*np.cos(phi_p)*np.cos(phi_m)
   
    return U

def qfp_force(phi_p,phi_m, params):
    phi_in, phi_x, betaQ, betaL = params
    dpp = (2/(betaL+2*betaQ))*(phi_p-phi_in) + 2*np.sin(phi_p)*np.cos(phi_m)
    dpm = (2/betaL)*(phi_m-phi_x) +  2*np.cos(phi_p)*np.sin(phi_m)
    return (-dpp, -dpm)

#realistic:
default_real = (0, 0, .2, 1.6)
#domain
dom = ((-12,-12),(12,12))

qfp_pot = Potential(qfp_pot, qfp_force, 4, 2, default_params=default_real, relevant_domain=dom)

from functools import reduce
from operator import iconcat
from math import exp, inf, sqrt
from numpy import multiply, array


from .state_distribution import StateDistribution #, UDStateDistribution

from ..gregtools import get_minima


def get_hist_from_marginals(marginal_hists):
    """Gets histogram in histogramdd form."""

    state_bins = [hist[1] for hist in marginal_hists]

    marginal_probs = [hist[0] for hist in marginal_hists]
    state_probs = reduce(multiply.outer, marginal_probs)

    state_hist = state_probs, state_bins

    return state_hist


def get_hist_from_pos_vel_marginals(pos_hists, vel_hists):

    nconfig_dim = len(pos_hists)

    assert nconfig_dim == len(vel_hists)

    marginal_hist_pairs = [pair for pair in zip(pos_hists, vel_hists)]

    marginal_hists = reduce(iconcat, marginal_hist_pairs, [])

    hist = get_hist_from_marginals(marginal_hists)

    return hist


# def make_equilibrium_dist(hamiltonian, mass, kappa, pos_lim=None)


def make_1DOD_metastable_dist(potential, kappa,
                              is_weights=None, infostates=None,
                              pos_lim=None, nbins=200,
                              pZ_threshold=1e-4,
                              min_start_points=[-1, 1],
                              max_start_points=[0]):

    hist = make_1D_pos_metastable_hist(potential, kappa, is_weights,
                                       infostates,
                                       pos_lim, nbins, pZ_threshold,
                                       min_start_points, max_start_points)

    hist = hist[0], (hist[1],)  # Put into histogramdd form
    state_shape = ()

    state_dist = StateDistribution(hist, state_shape)

    return state_dist


def make_1DUD_metastable_dist(get_potential, mass, kappa,
                              is_weights=None, infostates=None,
                              pos_lim=None, npos_bins=200,
                              pos_pZ_threshold=1e-4,
                              min_start_points=[-1, 1], max_start_points=[0],
                              vel_lim=None, nvel_bins=100,
                              max_vel_sig_dist=3.5):

    pos_hist = make_1D_pos_metastable_hist(get_potential, kappa, is_weights,
                                           infostates,
                                           pos_lim, npos_bins, pos_pZ_threshold,
                                           min_start_points, max_start_points)

    vel_hist = make_1D_vel_equilibrium_hist(mass, kappa, vel_lim, nvel_bins,
                                            max_vel_sig_dist)

    state_hist = get_hist_from_pos_vel_marginals([pos_hist], [vel_hist])

    state_shape = (2,)

    state_dist = StateDistribution(state_hist, state_shape)

    return state_dist


def make_1D_pos_metastable_hist(get_potential, kappa, weights=None, wells=None,
                                pos_lim=None, nbins=201, pZ_threshold=1e-4,
                                min_start_points=[-1, 1], max_start_points=[0]):

    """Makes a metastable distribution of the potential over the wells.

    Wells don't have to encompass the entire position state space.  And each
    well can be given different weights.  So arbitrary metastable distributions
    can be generated.

    Keywords:
    potential  -- (Polynomial like) A numpy Polynomial or other object that
    returns values of the potential and has appropriate deriv() and roots()
    methods.
    wells -- (None or list) If a list, each element is a tuple giving the
    minimum and left and right bounds of a well, in that order.  If None, the
    entire set of wells for the potential will be calculated and used.
    weights -- (None or list) If None, the metastable distribution will have
    assigned weights to each well in accordance with the canonical distribution.
    Otherwise, should be a list of weights, one for each well.
    nbins -- (integer) The number of bins to use.  Will bin up the used space
    evenly in x and without holes.
    """

    if wells is None:
        wells = make_wells(get_potential, min_start_points, max_start_points)

    minimum = min([well[0] for well in wells])
    def get_pZ(x):
        return exp((1/kappa) * (get_potential(minimum) - get_potential(x)))

    if pos_lim is not None:
        pos_min, pos_max = pos_lim
    else:
        pos_min = _find_pos_min(wells[0], get_pZ, nbins, pZ_threshold)
        pos_max = _find_pos_max(wells[-1], get_pZ, nbins, pZ_threshold)

    # Make the x_values and bin size
    dx = (pos_max - pos_min) / (nbins - 1)
    pos_values = [pos_min + i * dx for i in range(nbins)]

    # Find the pZs for the wells and pos_values
    well_pZs = [0]*len(wells)
    pos_pZs = [0]*nbins
    for i, well in enumerate(wells):
        for j, x in enumerate(pos_values):
            if well[1] < x and x < well[2]:
                pos_pZs[j] = get_pZ(x)
                well_pZs[i] += pos_pZs[j]

    # Properly weigh each pos_value
    if weights is None:
        Z = sum(pos_pZs)
        pos_probs = [pZ / Z for pZ in pos_pZs]
    else:
        pos_probs = [0]*nbins
        for i, well in enumerate(wells):
            well_prop = weights[i] / sum(weights)
            for j, x in enumerate(pos_values):
                if well[1] < x and x < well[2]:
                    pos_probs[j] = pos_pZs[j] / well_pZs[i] * well_prop

    pos_bins = [x - dx / 2 for x in pos_values]
    pos_bins.append(pos_values[-1] + dx / 2)

    return (array(pos_probs), array(pos_bins))


def _find_pos_min(well, get_pZ, nbins, pZ_threshold):

    if well[1] > -inf:
        pos_min = well[1]

    else:

        # Get the rough lower boundary search radius
        search_step = 1
        contracting = True
        while(contracting):
            try:
                if get_pZ(well[0] - search_step * nbins) < pZ_threshold:
                    search_step /= 10
                else:
                    contracting = False
            except OverflowError:
                search_step /= 10
        while get_pZ(well[0] - search_step * nbins) > pZ_threshold:
            search_step *= 10

        # Find the precise lower boundary for this well
        pos_min = well[0]
        pZmin = get_pZ(pos_min)
        while pZmin > pZ_threshold:
            pos_min -= search_step
            pZmin = get_pZ(pos_min)

    return pos_min

def _find_pos_max(well, get_pZ, nbins, pZ_threshold):

    if well[2] < inf:
        pos_max = well[2]

    else:

        # Get the rough lower boundary search radius
        search_step = 1
        contracting = True
        while(contracting):
            try:
                if get_pZ(well[0] + search_step * nbins) < pZ_threshold:
                    search_step /= 10
                else:
                    contracting = False
            except OverflowError:
                search_step /= 10
        while get_pZ(well[0] + search_step * nbins) > pZ_threshold:
            search_step *= 10

        # Find the precise lower boundary for this well
        pos_max = well[0]
        pZmin = get_pZ(pos_max)
        while pZmin > pZ_threshold:
            pos_max += search_step
            pZmin = get_pZ(pos_max)

    return pos_max


def make_wells(potential, min_start_points=[-1, 1], max_start_points=[0],
               get_minima=get_minima):
    """Makes the wells associated with the potential."""

    minima = get_minima(potential, min_start_points)
    maxima = get_minima(lambda x: -potential(x), max_start_points)

    wells = []

    for minimum in minima:

        lower_bound = -inf
        for maximum in maxima:
            if maximum < minimum:
                lower_bound = maximum
            else:
                break

        upper_bound = inf
        for maximum in reversed(maxima):
            if maximum > minimum:
                upper_bound = maximum
            else:
                break

        wells.append((minimum, lower_bound, upper_bound))

    return wells


def make_1D_vel_equilibrium_hist(mass, kappa, vel_lim=None, nvel_bins=101,
                                 max_vel_sig_dist=3.5):
    """Gets a histogram for an equilibrium distribution of velocities.

    Returns (vel_values, vel_probs) tuple."""

    def pZ(v):
        return exp(-(1/kappa) * (1/2) * mass * v ** 2)

    if vel_lim is not None:
        vel_min, vel_max = vel_lim
        dv = (vel_max - vel_min) / (nvel_bins - 1)
    else:
        vel_min = -max_vel_sig_dist * sqrt(kappa / mass)
        dv = 2 * abs(vel_min) / (nvel_bins - 1)

    vel_values = [vel_min + i * dv for i in range(nvel_bins)]

    pZs = [pZ(v) for v in vel_values]
    Z = sum(pZs)
    vel_probs = [pZ / Z for pZ in pZs]

    vel_bins = [v - dv / 2 for v in vel_values]
    vel_bins.append(vel_values[-1] + dv / 2)

    return (array(vel_probs), array(vel_bins))
# Created 2019-09-30 from disttools1D.py during Sept 2019 refactor

from numpy import unravel_index, array
from numpy.random import RandomState
from operator import itemgetter, mul
from functools import reduce


class StateDistribution:

    def __init__(self, hist, state_shape):
        """Creates instance from a numpy histogramdd.

        A numpy histogramdd will have as its first entry a D dimensional
        array, where D is the dimensionality of a state, whose values
        are probabilities of the corresponding states.  The second entry
        is a D length list of arrays, each giving the bins in their
        corresponding dimension.  These bin lists specifically give
        the boundaries between each bin, including the lower bound and
        upper bound on the entire range.

        state_shape should be the shape of an array giving a single trial's
        state at some time.
        The convention Kyle and I decided to use is that the shape should
        have the following form:
        (nparticle_spatial_dims, nposvel_subdims)
        where nposvel_subdims is 1 if there's just positions and 2 if
        there are positions and velocities.
        However, if any slot has a one, I'm currently imposing that
        it should be removed.
        E.g. a single particle in 1D with position and velocity would have
        shape (nposvel_subdims,) = (2,).
        A state with four particles in 3 spatial dimensions but only positions
        would have state shape (4, 3).
        whereas if they also had velocities the state shape would be (4, 3, 2).
        A state for a single particle in one spatial dimension with only
        position would have shape ().
        """

        bins = hist[1]

        hist_range = [(dim_bins[0], dim_bins[1]) for dim_bins in bins]

        norm = hist[0].sum()
        probs = hist[0] / norm

        probs_hist = probs, bins

        ndim = len(bins)
        dim_nbins = [len(bins[i]) - 1 for i in range(ndim)]

        assert ndim == reduce(mul, state_shape, 1)

        self.state_shape = state_shape
        self.bins = bins
        self.hist_range = hist_range
        self.probs = probs
        self.probs_hist = probs_hist
        self.ndim = ndim
        self.dim_nbins = dim_nbins

    def __getitem__(self, item):

        return self.probs_hist[item]

    def get_sample(self, ntrials, seed=None):
        """Samples ntrials from the distribution, perhaps with a given seed."""

        state_shape = self.state_shape
        bins = self.bins
        probs = self.probs
        dim_nbins = self.dim_nbins
        # ndim = self.ndim

        rng = RandomState(seed) if seed else RandomState()

        flat_indices = rng.choice(probs.size, p=probs.flatten(), size=ntrials)

        coordinate_indices = unravel_index(flat_indices, dim_nbins)

        dim_vals = []
        for dim, dim_bins in enumerate(bins):

            dim_indices = coordinate_indices[dim]
            lower_bounds = dim_bins[dim_indices]
            upper_bounds = dim_bins[dim_indices + 1]

            dim_val = rng.uniform(lower_bounds, upper_bounds)
            dim_vals.append(dim_val)

        array_shape = [-1]
        array_shape.extend(state_shape)

        state_vals = array(dim_vals).reshape(array_shape)

        return state_vals

    def get_marginal_hist(self, dimensions):
        """Gets marginalized histogram for given dimensions.

        dimensions should be a list like.
        """

        bins = self.bins
        probs = self.probs
        ndim = self.ndim

        axes_to_sum = [dim for dim in range(ndim) if dim not in dimensions]

        marginal_bins = [bins[dim] for dim in dimensions]
        marginal_vals = probs.sum(axis=tuple(axes_to_sum))
        marginal_hist = marginal_vals, marginal_bins

        return marginal_hist


# class UDStateDistribution(StateDistribution):
#     """State Distribution for an underdamped type state.
#
#     Underdamped here means that a state is characterized by both positions
#     and velocities.  Overdamped would mean the state is characterized by
#     only posions.  This is important for reconstructing an array of states
#     to unravel the state components into indices for both the number of
#     configurational dimensions and one specifying either position or velocity.
#
#     Note that sampled states will have shape of (ntrials, nconfig_dim, 2)
#     if nconfig_dim is > 1, else the shape will be (ntrials, 2).
#     """
#
#     def __init__(self, hist):
#
#         super().__init__(hist)
#
#         ndim = self.ndim
#         assert ndim % 2 == 0
#
#         nconfig_dim = int(ndim / 2)
#
#         self.nconfig_dim = nconfig_dim
#
#     def get_sample(self, ntrials, seed=None):
#
#         nconfig_dim = self.nconfig_dim
#
#         unshaped_state_array = super().get_sample(ntrials, seed)
#
#         shape = (-1, nconfig_dim, 2) if nconfig_dim > 1 else (-1, 2)
#
#         state_array = unshaped_state_array.reshape(shape)
#
#         return state_array


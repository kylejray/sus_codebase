# gregtools.py
# Started April, 2017 by Greg
#
# A set of tools I couldn't find.

import numpy as np
from math import floor
from numpy import take, linspace
from scipy.optimize import fmin


def get_rough_minima_from_array(Y):
    """Returns all local minima of the array, ordered."""

    X = linspace(0, len(Y)-1, len(Y), dtype=int)

    return X[ np.r_[Y[:-1]<Y[1:], False] & np.r_[False, Y[1:]<Y[:-1]]]


def get_rough_minima_from_array_alt(Y):
    """Returns all local minima of the array, ordered."""

    X = linspace(0, len(Y)-1, len(Y), dtype=int)

    return X[ np.r_[Y[:-1]<Y[1:], False] & np.r_[False, Y[1:]<=Y[:-1]]]


def get_rough_minima(func, xlim, xsteps=10_000):
    """Returns all local minima of the function within the range, ordered."""

    X = linspace(*xlim, xsteps + 1)
    try:
        Y = func(X)
    except TypeError:
        Y = np.array(func(x) for x in X)

    return X[ np.r_[Y[:-1]<Y[1:], False] & np.r_[False, Y[1:]<Y[:-1]]]


def get_rough_minima_alt(func, xlim, xsteps=10_000):
    """Returns all local minima of the function within the range, ordered."""

    X = linspace(*xlim, xsteps + 1)
    try:
        Y = func(X)
    except TypeError:
        Y = np.array(func(x) for x in X)

    return X[ np.r_[Y[:-1]<Y[1:], False] & np.r_[False, Y[1:]<=Y[:-1]]]


def get_minima(func, start_points, **kwargs):
    """Gets a sorted set of minima for a function.

    kwargs gets fed to fmin from scipy.optimize.
    So one should add an 'xtol' arg to set the tolerance manually.
    The value of xtol is used to check if minima are the same with np.isclose.
    """

    # 0.0001 is the default value for xtol in scipy.optimize.fmin
    if 'xtol' not in kwargs:
        kwargs['xtol'] = 0.0001

    temp_minima = sorted({fmin(func, x0, **kwargs)[0] for x0 in start_points})

    if len(temp_minima) == 0:
        return []

    last_minimum = temp_minima[0]
    minima = [last_minimum]
    for minimum in temp_minima[1:]:
        if not np.isclose(minimum, last_minimum, atol=4*kwargs['xtol']):
            minima.append(minimum)
            last_minimum = minimum

    return minima


def get_poly_minima(polynomial, max_iteration=100):
    '''Finds the local minima of a numpy Polynomial.

    Could also be anything that has appropriate deriv() and roots() methods.'''

    derivative = polynomial.deriv()
    roots = derivative.roots()
    extrema = [root.real for root in roots if root.imag == 0]

    derivative2 = derivative.deriv()
    def is_minima(x, p, iteration=1):

        if iteration > max_iteration:
            print("Error: reached max iteration count.")
            return

        y = p(x)
        if y > 0:
            return True
        if y < 0:
            return False
        if y == 0:
            return is_minima(x, p.deriv(), iteration + 1)

        print("Error: found a complex root of some derivative.")
        return

    minima = []
    for extremum in extrema:
        if is_minima(extremum, derivative2):
            minima.append(extremum)

    return sorted(minima)


def multitake(a, ax_indices_pairs):
    '''Expands numpy's take function to taking indices from multiple axes.

    ax_indicies_pairs should be a list of pairs whose first elements are
    axes to collapse and whose second elements are lists of indices to take from
    the corresponding axis.'''

    ax_indices_pairs = sorted(ax_indices_pairs, key=lambda pair: pair[0],
                              reverse=True)

    for ax, indices in ax_indices_pairs:
        a = take(a, indices, ax)

    return a


class Bunch:
    """For making generic objects with arbitrary attribute assignments."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def get_hist_mean(hist):

    bins = hist[1]
    vals = hist[0]

    midpoints = bins[:-1] + (bins[1] - bins[0]) / 2

    mean = (vals * midpoints).sum() / vals.sum()

    return mean


def attempt(func, *args, **kwargs):
    """Attempts to run func with args and kwargs.

    Will escape if the exception is an AttributeError."""

    try:
        func(*args, **kwargs)

    except AttributeError:
        pass


#------------------------------------------------
#---- Representing numbers in scientific notation

# Here, an example of scientific notation means an expression given by
# a real number in decimal notation, called the significand, followed by
# 10^n where n is an integer called the exponent.  In normalized notation,
# the significand must be at least 1 and less than 10.

def get_normalized_exponent(a):
    '''Finds the normalized scientific notation exponent of a number.

    Specifically, the significand would be a number between 1 and 10, left
    inclusive, right exclusive.  If the number is zero, returns None.
    '''

    if a == 0:
        return None

    b = abs(a)

    if floor(b) > 0:
        n = 0
        while floor(b / 10 ** n) > 10:
            n += 1
    else:
        n = -1
        while floor(b / 10 ** n) < 1:
            n -= 1
    return n

def get_min_normalized_exponent(A):
    '''Finds the minimum of the normalized notation exponents of numbers.

    Considers only non-zero elements of the numbers given.  Returns None if all
    numbers are 0.'''

    min_exp = None
    for a in A:
        exp = get_normalized_exponent(a)
        if exp is not None:
            min_exp = exp if min_exp is None else min(exp, min_exp)

    return min_exp

def get_significand(a, exp=None):
    '''Returns the sci-notation significand of a number given an exponent.

    If exp is None, the exponent is the normalized notation exponent.
    '''

    if exp is None:
        exp = get_normalized_exponent(a)

    return '{:.2f}'.format(a / 10 ** exp)

def get_common_exp_significands(A, exp=None):
    '''Returns the sci-notation significands of numbers given an exponent.

    If exp is None, the exponent is the minimum of the normalized notation
    exponents of the numbers.
    '''

    if exp is None:
        exp = get_min_normalized_exponent(A)

    return [get_significand(a, exp) for a in A]

def get_scinotation_rep(a, exp=None):
    '''Returns the number in sci-notation given an exponent.

    If exp is None, the exponent is the normalized notation exponent.
    '''

    if exp is None:
        exp = get_normalized_exponent(a)

    return '{:.2f}e{}'.format(a / 10 ** exp, exp)

def get_common_exp_scinotation_reps(A, exp=None):
    '''Returns the numbers in sci-notation given an exponent.

    If exp is None, the exponent is the minimum of the normalized notation
    exponents of the numbers.
    '''

    if exp is None:
        exp = get_min_normalized_exponent(A)

    return [get_scinotation_rep(a, exp) for a in A]

from math import log
from numpy import histogram, array, sum


def get_avg_quantity(quantity):

    return quantity.sum(axis=0) / quantity.shape[0]


def get_bit_entropy(avg_bit_value):

    def plogp(p):
        return p * log(p, 2) if p > 0 else 0

    return -plogp(avg_bit_value) - plogp(1 - avg_bit_value)


# def F(avg_bit_value):
#     return log(2) * bit_entropy(avg_bit_value)


def get_hist_avg(hist):

    vals = hist[0]
    bins = hist[1]

    xvals = (bins[1:] + bins[:-1]) / 2

    return sum(xvals * vals)


def get_hist_var(hist):

    vals = hist[0]
    bins = hist[1]

    xvals = (bins[1:] + bins[:-1]) / 2
    avg = sum(xvals * vals)

    return sum((xvals - avg) ** 2 * vals)


def make_pr_mass_hist(quantity, total=None, **kwargs):

    count_hist = histogram(quantity, **kwargs)

    if total is None:
        total = count_hist[0].sum()

    bins = count_hist[1]

    pr_mass_vals = count_hist[0] / total if total > 0 else 0
    pr_mass_hist = pr_mass_vals, bins

    return pr_mass_hist


def make_pr_density_hist(quantity, total=None, **kwargs):

    count_hist = histogram(quantity, **kwargs)

    if total is None:
        total = count_hist[0].sum()

    bins = count_hist[1]
    dbin = bins[1] - bins[0]

    pr_density_vals = count_hist[0] / (total * dbin) if total > 0 else 0
    pr_density_hist = pr_density_vals, bins

    return pr_density_hist


def convert_count_to_pr_density_hist(count_hist, total=None):

    if total is None:
        total = count_hist[0].sum()

    bins = count_hist[1]
    dbin = bins[1] - bins[0]

    pr_density_vals = count_hist[0] / (total * dbin) if total > 0 else 0
    pr_density_hist = pr_density_vals, bins

    return pr_density_hist


def convert_count_to_pr_mass_hist(count_hist, total=None):

    if total is None:
        total = count_hist[0].sum()

    bins = count_hist[1]

    pr_mass_vals = count_hist[0] / (total) if total > 0 else 0
    pr_mass_hist = pr_mass_vals, bins

    return pr_mass_hist


def convert_pr_mass_to_pr_density_hist(pr_mass_hist):

    pr_mass_vals = pr_mass_hist[0]

    bins = pr_mass_hist[1]
    dbin = bins[1] - bins[0]

    pr_density_vals = pr_mass_vals / dbin
    pr_density_hist = pr_density_vals, bins

    return pr_density_hist


def sum_hists(hists, weights=None):

    if weights is None:
        weights = [1.] * len(hists)

    bins = hists[0][1]

    summed_vals = 0

    for hist, weight in zip(hists, weights):

        summed_vals += hist[0] * weight

    summed_hist = summed_vals, bins

    return summed_hist


def make_key_steps(key_times, dt, t0=0):

    key_steps = []
    for key_time in key_times:
        step = round((key_time - t0) / dt)
        key_steps.append(step)

    return key_steps

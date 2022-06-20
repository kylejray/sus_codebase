
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem


def w_TUR(initial_state, final_state, mean_work, current_function=None):
    if current_function is None:
        displacement = np.array(final_state)[...,0] - np.array(initial_state)[...,0]
        current = displacement
    else:
        current = current_function(initial_state, final_state)
    scaled_var = np.var(current, axis=0) / np.mean(current, axis=0)**2
    bound = 2/(np.exp(mean_work)-1)

    return bound, scaled_var

def ft_moment(sigma, moment, condition=None):
    '''
    returns moentis of a distribution, if the distribution obeys the ft p(-x)=e^(-x)p(x) then the conditions sigma>0 or sigma<0 can be included in order to give the moments using just positive or just negative pieces of the distribution and inferring the other half to be consistant with the FT.
    '''
    N = len(sigma)
    mean = 0

    if condition is None:
        if moment > 1:
            mean = np.mean(sigma)
        f_sigma = (sigma-mean)**moment

    if condition is not None:
        c_sigma = sigma[condition]
        Nc = len(c_sigma)
        if moment > 1:
            mean = ft_moment(sigma, 1, condition=condition)[0]
        f_sigma = (Nc/N) * ( (c_sigma-mean)**moment + (-c_sigma-mean)**moment * np.exp(-c_sigma))

    return [ np.mean(f_sigma), sem(f_sigma) ]

    
def crooks_analysis_tsp(work, nbins=25, beta=1, low_stats=True):
    '''
    function to do crooks analysis for a list of works that come frmo a time symmeteic protocol. does some plots, returns some info

    Arguments
    --------
    work: ndarray of dimension [N_samples]
        the works, this is for time symmetric protocols, so no reverse process is needed
    nbins: int
        desired number of work bins
    beta: float
        1/(kB*T)
    low_stats: boolean
        if set to True (default), the function will atempt to look only in the subspace of work where we have both +W and -W realizations

    Returns
    -------
    works: ndarray of dimension [nbins,]
        array of the works asspcoated with the...
    counts: ndarray of dimensions [2, nbins]
        array of the log of the counts associate with the works above counts[0]/counts[1] is the counts for negative/positive works
    '''

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].hist(work, bins=2*nbins+1, log=True)
    ax[0].set_xlabel('$W$')
    ax[0].set_title('bare work histogram')

    naive_jar = np.mean(np.exp(-work))
    total_count = len(work)
    zero_work_count = sum(work == 0)
    print('naive jarzynski: average of exp -Beta W is {}'.format(naive_jar))
    work = work[work != 0]

    w_min, w_max = np.min(work), np.max(work)

    if low_stats:
        limit = min(abs(w_min), abs(w_max))
    else:
        limit = max(abs(w_min), abs(w_max))

    bins = np.linspace(-limit, limit, 2*nbins+1)

    counts = np.histogram(work, bins=bins)[0]

    neg_counts = counts[:nbins]
    neg_counts = neg_counts[::-1]
    pos_counts = counts[nbins:]

    step_size = limit/nbins
    works = np.linspace(step_size/2, limit-step_size/2, nbins)

    trunc_exp_work = (zero_work_count + sum(neg_counts*np.exp(beta*works)) + sum(pos_counts*np.exp(-beta*works)))/(sum(counts)+zero_work_count)

    print('binned jarzynski: binned average of exp -Beta W using only values of work where we have +W and -W realizations: {}'.format(trunc_exp_work))
    ignored_ratio = (total_count-zero_work_count-sum(counts))/total_count
    print('this means ignoring {:.1f} percent of trials'.format(100*ignored_ratio))

    ax[1].hist(work, bins, log=True)
    ax[1].set_xlabel('$W$')
    ax[1].set_title('histogram with truncated data')

    log_ratio = np.log(np.divide(pos_counts, neg_counts))

    ax[2].scatter(beta*works, log_ratio)
    ax[2].plot(beta*works, beta*works, '--')
    ax[2].set_xlabel('$\\beta W$')
    ax[2].set_ylabel('$\\ln \\frac{{P(W)}}{{P(-W)}}$')
    ax[2].set_title('Crooks for truncated data')
    plt.show()

    return works, [neg_counts, pos_counts]

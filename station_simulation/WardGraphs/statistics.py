import scipy.stats as st
from fit_nbinom import fit_nbinom
from fit_poisson import fit_poisson
import matplotlib.pyplot as plt
import numpy as np
from math import ceil, floor
import sys

def bin_pmf(pmf, bins, n=1):
    hist_sum = n
    f_exp = []
    prev_bin_max = bins[0]
    for bin_max in bins[1:]:
        bin_sum = sum([hist_sum * pmf(k) for k in range(ceil(prev_bin_max), ceil(bin_max))]) 
        f_exp += [bin_sum]
        prev_bin_max = bin_max
    return f_exp


def hist_bins(data, bins):
    ret = []
    prev_bin_max = bins[0]
    for bin_max in bins[1:]:
        bin_sum = sum([1 if (prev_bin_max <= x) and (x < bin_max) else 0 for x in data])
        ret += [bin_sum]
        prev_bin_max = bin_max
    return ret

def get_best_distribution_discrete(data, binsize=1):
    dist_names = ["nbinom", "poisson"]
    #dist_names = ["poisson"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        if dist_name == 'nbinom':
            param = fit_nbinom(data)
        elif dist_name == 'poisson':
            param = fit_poisson(data)
        else:
            param = dist.fit(data)

        bins = range(0,int(max(data)+1), binsize)
        f_obs = hist_bins(data, bins)

        hist_sum = sum(f_obs)
        f_exp = bin_pmf(lambda k: dist.pmf(k, **param), bins, n=hist_sum)

        params[dist_name] = param
        # Applying the chi squared test
        # print(f'testing {dist_name}...', file=sys.stderr)
        D, p = st.chisquare(f_obs, f_exp)
        # print("p value for "+dist_name+" = "+str(p), file=sys.stderr)
        dist_results.append((dist_name, dist, p))

    # select the best fitted distribution
    best_dist_name, best_dist, best_p = (max(dist_results, key=lambda item: item[2]))
    # store the name of the best fit and its p value

    # print("Best fitting distribution: "+str(best_dist_name), file=sys.stderr)
    # print("Best p value: "+ str(best_p), file=sys.stderr)
    # print("Parameters for the best fit: "+ str(params[best_dist_name]), file=sys.stderr)

    return best_dist, best_p, params[best_dist_name]


def get_best_distribution(data):
    dist_names = ["gamma", "beta", "expon", "pearson3", "triang", "lognorm", "powerlaw", "uniform", "norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
    #dist_names = ["expon"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        print(f'testing {dist_name}...')
        D, p = st.kstest(data, dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, dist, p))

    # select the best fitted distribution
    best_dist_name, best_dist, best_p = (max(dist_results, key=lambda item: item[2]))
    # store the name of the best fit and its p value

    print(params)

    print("Best fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist_name]))

    return best_dist, best_p, params[best_dist_name]
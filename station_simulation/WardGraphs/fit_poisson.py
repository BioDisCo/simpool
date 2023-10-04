import numpy as np
from scipy.special import gammaln
from scipy.special import psi
from scipy.special import factorial
from scipy.optimize import fmin_l_bfgs_b as optim
from scipy.optimize import minimize
from scipy import stats


def fit_poisson(X, initial_params=None):

    def poisson(k, lamb):
        """poisson pdf, parameter lamb is the fit parameter"""
        return (lamb**k/factorial(k)) * np.exp(-lamb)


    def negative_log_likelihood(params, data):
        """
        The negative log-Likelihood-Function
        """
        lnl = - np.sum(np.log(poisson(data, params[0])))
        return lnl

    def negative_log_likelihood(params, data):
        ''' better alternative using scipy '''
        return -stats.poisson.logpmf(data, params[0]).sum()


    # get poisson deviated random numbers
    data = np.random.poisson(2, 1000)

    # minimize the negative log-Likelihood
    result = minimize(negative_log_likelihood,  # function to minimize
                     x0=np.ones(1),            # start value
                     args=(data,),             # additional arguments for function
                     method='Powell',          # minimization method, see docs
                     )
    return {'mu': float(result.x)}
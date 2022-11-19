import numpy as np
import cupy as cp
import cupyx
import scipy

def np_poisson_logpmf(k, r):
    # should return the same as scipy.stats.logpmf(k, r), but is  ~4x? faster
    #return k  * np.log(r) - r - np.log(scipy.special.factorial(k))
    return k  * np.log(r) - r - scipy.special.gammaln(k+1)

def cp_poisson_logpmf(k, r):
    return k * cp.log(r) - r - cupyx.scipy.special.gammaln(k+1)

def cuda_poisson_logpmf(k, r):
    return k * cp.log(r) - r - cupyx.scipy.special.gammaln(k+1)

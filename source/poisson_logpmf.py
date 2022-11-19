import numpy as np
import cupy as cp
import cupyx
import scipy

from source_C import poisson_logpmf

def np_poisson_logpmf(k, r):
    # should return the same as scipy.stats.logpmf(k, r), but is  ~4x? faster
    #return k  * np.log(r) - r - np.log(scipy.special.factorial(k))
    return k  * np.log(r) - r - scipy.special.gammaln(k+1)

def cp_poisson_logpmf(k, r):
    if isinstance(k, np.ndarray) and isinstance(r, np.ndarray):
        k_ = cp.asarray(k)
        r_ = cp.asarray(r)
        return k_ * cp.log(r_) - r_ - cupyx.scipy.special.gammaln(k_+1)
    if isinstance(k, cp.ndarray) and isinstance(r, np.ndarray):
        r_ = cp.asarray(r)
        return k * cp.log(r_) - r_ - cupyx.scipy.special.gammaln(k+1)
    if isinstance(k, np.ndarray) and isinstance(r, cp.ndarray):
        k_ = cp.asarray(k)
        return k_ * cp.log(r) - r - cupyx.scipy.special.gammaln(k_+1)
    if isinstance(k, cp.ndarray) and isinstance(r, cp.ndarray):
        return k * cp.log(r) - r - cupyx.scipy.special.gammaln(k+1)
    return NotImplemented

def cuda_poisson_logpmf(k, r):
    if isinstance(k, np.ndarray) and isinstance(r, np.ndarray):
        return poisson_logpmf(k, r)
    if isinstance(k, cp.ndarray) and isinstance(r, np.ndarray):
        out = cp.zeros_like(r)
        poisson_logpmf(k.data.ptr, r, out.data.ptr)
        return out
    if isinstance(k, np.ndarray) and isinstance(r, cp.ndarray):
        return NotImplemented
    if isinstance(k, cp.ndarray) and isinstance(r, cp.ndarray):
        out = cp.zeros_like(r)
        poisson_logpmf(k.data.ptr, r.data.ptr, out.data.ptr, k.size)
        return out
    return NotImplemented

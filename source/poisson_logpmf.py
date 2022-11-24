import numpy as np
import cupy as cp
import cupyx
import scipy

from source_C import poisson_logpmf

def np1_poisson_logpmf(k, r):
    # should return the same as scipy.stats.logpmf(k, r), but is  ~4x? faster
    return k  * np.log(r) - r - np.log(scipy.special.factorial(k))

def np2_poisson_logpmf(k, r):
    # should return the same as scipy.stats.logpmf(k, r), but is  ~4x? faster
    #return k  * np.log(r) - r - np.log(scipy.special.factorial(k))
    return k  * np.log(r) - r - scipy.special.gammaln(k+1)

def cp_poisson_logpmf(k, r):
    if isinstance(k, np.ndarray) and isinstance(r, np.ndarray):
        k_ = cp.asarray(k)
        r_ = cp.asarray(r)

        temp = cp.log(r_)
        temp = k_ * temp
        temp = temp - r_
        temp2 = k_ + 1
        temp2 = cupyx.scipy.special.gammaln(temp2)
        ret = temp - temp2

        #ret = k_ * cp.log(r_) - r_ - cupyx.scipy.special.gammaln(k_+1)
        return cp.asnumpy(ret)
    if isinstance(k, cp.ndarray) and isinstance(r, np.ndarray):
        r_ = cp.asarray(r)

        temp = cp.log(r_)
        temp = k * temp
        temp = temp - r_
        temp2 = k + 1
        temp2 = cupyx.scipy.special.gammaln(temp2)
        ret = temp - temp2

        #return k * cp.log(r_) - r_ - cupyx.scipy.special.gammaln(k+1)
        return ret
    if isinstance(k, np.ndarray) and isinstance(r, cp.ndarray):
        k_ = cp.asarray(k)

        temp = cp.log(r)
        temp = k_ * temp
        temp = temp - r
        temp2 = k_ + 1
        temp2 = cupyx.scipy.special.gammaln(temp2)
        ret = temp - temp2

        #return k_ * cp.log(r) - r - cupyx.scipy.special.gammaln(k_+1)
        return ret
    if isinstance(k, cp.ndarray) and isinstance(r, cp.ndarray):

        temp = cp.log(r)
        temp = k * temp
        temp = temp - r
        temp2 = k + 1
        temp2 = cupyx.scipy.special.gammaln(temp2)
        ret = temp - temp2

        #return k * cp.log(r) - r - cupyx.scipy.special.gammaln(k+1)
        return ret
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

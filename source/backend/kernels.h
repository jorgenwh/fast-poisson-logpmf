#ifndef KERNELS_H_
#define KERNELS_H_

#include <cuda_runtime.h>

#include "common.h"

void poisson_logpmf_np_and_np_to_np(const int *k, const double *r, double *out, const int size);
void poisson_logpmf_cp_and_np_to_cp(const int *k, const double *r, double *out, const int size);
void poisson_logpmf_cp_and_cp_to_cp(const int *k, const double *r, double *out, const int size);

#endif // KERNELS_H_


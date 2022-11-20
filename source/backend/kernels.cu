#include <cuda_runtime.h>

#include "common.h"
#include "kernels.h"

__global__ static void poisson_logpmf_kernel(
    const int *k, const double *r, double *out, const int size)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size)
  {
    return;
  }

  out[i] = k[i] * logf(r[i]) - r[i] - lgammaf(k[i]+1);
}

// k is data in host RAM
// r is data in host RAM
// out is data in host RAM
void poisson_logpmf_np_and_np_to_np(
    const int *k, const double *r, double *out, const int size)
{
  //int stream_size = size / NUM_STREAMS;

  //cudaStream_t streams[NUM_STREAMS];
  //for (int i = 0; i < NUM_STREAMS; i++)
  //{
    //cuda_errchk(cudaStreamCreate(&streams[i]));
  //}

  int *k_d;
  double *r_d, *out_d;
  cudaMalloc(&k_d, sizeof(int)*size);
  cudaMalloc(&r_d, sizeof(double)*size);
  cudaMalloc(&out_d, sizeof(double)*size);

  //for (int i = 0; i < NUM_STREAMS; i++)
  //{
    //int offset = i * stream_size;
    //int chunk_size = (i < NUM_STREAMS-1) ? stream_size : size-offset;
  int num_blocks = size / THREAD_BLOCK_SIZE + (size % THREAD_BLOCK_SIZE > 0);

  cuda_errchk(cudaMemcpy(k_d, k, 
        sizeof(int)*size, cudaMemcpyHostToDevice));
  cuda_errchk(cudaMemcpy(r_d, r, 
        sizeof(double)*size, cudaMemcpyHostToDevice));

  poisson_logpmf_kernel<<<num_blocks, THREAD_BLOCK_SIZE>>>(
      k_d, r_d, out_d, size);

  cuda_errchk(cudaMemcpy(out, out_d, 
        sizeof(double)*size, cudaMemcpyDeviceToHost));
  //}

  cuda_errchk(cudaDeviceSynchronize());
  cuda_errchk(cudaFree(k_d));
  cuda_errchk(cudaFree(r_d));
  cuda_errchk(cudaFree(out_d));
}

// k is data in GPU global memory
// r is data in host RAM
// out is data in GPU global memory
void poisson_logpmf_cp_and_np_to_cp(
    const int *k, const double *r, double *out, const int size)
{
  //int stream_size = size / NUM_STREAMS;

  //cudaStream_t streams[NUM_STREAMS];
  //for (int i = 0; i < NUM_STREAMS; i++)
  //{
    //cuda_errchk(cudaStreamCreate(&streams[i]));
  //}

  double *r_d;
  cudaMalloc(&r_d, sizeof(double)*size);

  //for (int i = 0; i < NUM_STREAMS; i++)
  //{
    //int offset = i * stream_size;
    //int chunk_size = (i < NUM_STREAMS-1) ? stream_size : size-offset;
  int num_blocks = size / THREAD_BLOCK_SIZE + (size % THREAD_BLOCK_SIZE > 0);

  cuda_errchk(cudaMemcpy(r_d, r, 
        sizeof(double)*size, cudaMemcpyHostToDevice));

  poisson_logpmf_kernel<<<num_blocks, THREAD_BLOCK_SIZE>>>(
      k, r_d, out, size);
  //}

  cuda_errchk(cudaDeviceSynchronize());
  cuda_errchk(cudaFree(r_d));
}

// k is data in GPU global memory
// r is data in GPU global memory
// out is data in GPU global memory
void poisson_logpmf_cp_and_cp_to_cp(
    const int *k, const double *r, double *out, const int size)
{
  int num_blocks = size / THREAD_BLOCK_SIZE + (size % THREAD_BLOCK_SIZE > 0);
  poisson_logpmf_kernel<<<num_blocks, THREAD_BLOCK_SIZE>>>(
      k, r, out, size);
  cuda_errchk(cudaDeviceSynchronize());
}

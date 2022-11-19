#ifndef COMMON_H_
#define COMMON_H_

#include <stdio.h>
#include <cuda_runtime.h>

#define THREAD_BLOCK_SIZE 128
#define NUM_STREAMS 10

#define cuda_errchk(err) { cuda_errcheck(err, __FILE__, __LINE__); }

inline void cuda_errcheck(cudaError_t code, const char *file, int line, bool abort=true) 
{
#ifdef _CUDA_ERROR_CHECK
  if (code != cudaSuccess) 
  {
    switch (code) 
    {
      case 2:
        fprintf(stderr, "CUDA out of memory error in %s at line %d\n", file, line);
        break;
      default:
        fprintf(stderr, "CUDA assert: '%s', in %s, at line %d\n", cudaGetErrorString(code), file, line);
    }
    exit(code);
  }
#endif // _CUDA_ERROR_CHECK
}

#endif // COMMON_H_

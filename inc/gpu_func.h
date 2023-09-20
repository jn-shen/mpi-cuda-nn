#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <iostream>

#include "../utils/types.h"

#define CUDA_CHECK(error)                                          \
  {                                                                \
    if (error != cudaSuccess)                                      \
      std::cerr << __FILE__ << ":" << __LINE__ << " CUDA Error: '" \
                << cudaGetErrorString(error) << "'\n";             \
  }

#define CEIL(a, b) ((a + b - 1) / b)

struct event_pair
{
  cudaEvent_t start;
  cudaEvent_t end;
};

inline void check_launch(const char *kernel_name)
{
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess)
  {
    std::cerr << "error in " << kernel_name << " kernel" << std::endl;
    std::cerr << "error was: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}

inline void start_timer(event_pair *p)
{
  cudaEventCreate(&p->start);
  cudaEventCreate(&p->end);
  cudaEventRecord(p->start, 0);
}

inline double stop_timer(event_pair *p)
{
  cudaEventRecord(p->end, 0);
  cudaEventSynchronize(p->end);

  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, p->start, p->end);
  cudaEventDestroy(p->start);
  cudaEventDestroy(p->end);
  return elapsed_time;
}

struct GPU_NN
{
  real *d_W0;
  real *d_W1;
  real *d_b0;
  real *d_b1;
  int n_rows_0;
  int n_cols_0;
  int n_rows_1;
  int n_cols_1;
};

struct GPU_cache
{
  real *d_z1;
  real *d_z2;
  real *d_a1;
  real *d_yc;
};

struct GPU_grads
{
  real *d_dW0;
  real *d_dW1;
  real *d_db0;
  real *d_db1;
};

struct BP_temp
{
  real *d_diff;
  real *d_da1;
  real *d_dz1;
  real *d_dz1_t1;
  real *d_dz1_t2;
};

struct Matrix_t
{
  real *d_a1_t;
  real *d_W1_t;
  real *d_X_batch_t;
};

int myGEMM(real *A, real *B, real *C, real *alpha, real *beta, int M, int N, int K);

void GPUrepmat(const real *d_src, real *d_dst, int M, int N);

void GPUsigmoid(const real *d_input, real *d_output, int M, int N);

void GPUsoftmax(const real *d_input, real *d_output, int M, int N);

void GPUtranspose(const real *d_input, real *d_output, int M, int N);

void GPUadd(const real *d_A, const real *d_B, real *d_C, real alpha, real beta, int M, int N);

void GPUcolumnSum(const real *d_input, real *d_output, int M, int N);

void GPUhadamardProduct(const real *d_a, const real *d_b, real *d_output, int M, int N);
// TODO
// Add additional function declarations

#endif

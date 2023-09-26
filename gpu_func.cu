#include "gpu_func.h"
#include <cuda_runtime.h>

#include <iostream>
#include "cublas_v2.h"

#define TILE_WIDTH 16

__global__ void basicGEMM(const real *A, const real *B, real *C, real alpha, real beta, int M, int N, int K)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N)
  {
    real sum = 0;
    for (int k = 0; k < K; k++)
    {
      sum += A[k * M + row] * B[col * K + k];
    }
    C[col * M + row] = alpha * sum + beta * C[col * M + row];
  }
}

__global__ void sharedGEMM(const real *A, const real *B, real *C, real alpha, real beta, int M, int N, int K)
{
  __shared__ real Asub[TILE_WIDTH][TILE_WIDTH];
  __shared__ real Bsub[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = bx * TILE_WIDTH + tx;
  int col = by * TILE_WIDTH + ty;

  real sum = 0;

  for (int m = 0; m < CEIL(K, TILE_WIDTH); ++m)
  {
    if (row < M && m * TILE_WIDTH + ty < K)
      Asub[ty][tx] = A[(m * TILE_WIDTH + ty) * M + row];
    else
      Asub[ty][tx] = 0.0;
    
    if (m * TILE_WIDTH + tx < K && col < N)
      Bsub[ty][tx] = B[m * TILE_WIDTH + tx + K * col];
    else
      Bsub[ty][tx] = 0.0;

    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; k++)
    {
      sum += Asub[k][tx] * Bsub[ty][k];
    }

    __syncthreads();
  }

  if (row < M && col < N)
  {
    C[col * M + row] = alpha * sum + beta * C[col * M + row];
  }

}

/*
Routine to perform an in-place GEMM operation, i.e., C := alpha*A*B + beta*C
*/
int myGEMM(real *__restrict__ A, real *__restrict__ B, real *__restrict__ C, real *alpha, real *beta, int M, int N, int K)
{
  // dim3 threadsPerBlock(16, 16);
  // dim3 numBlocks(CEIL(M, threadsPerBlock.x), CEIL(N, threadsPerBlock.y));
  // basicGEMM<<<numBlocks, threadsPerBlock>>>(A, B, C, *alpha, *beta, M, N, K);

  dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 numBlocks(CEIL(M, TILE_WIDTH), CEIL(N, TILE_WIDTH));
  sharedGEMM<<<numBlocks, threadsPerBlock>>>(A, B, C, *alpha, *beta, M, N, K);
  return 0;
}

__global__ void repmatKernel(const real *d_src, real *d_dst, int M, int N)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N)
  {
    d_dst[col * M + row] = d_src[row];
  }
}

void GPUrepmat(const real *d_src, real *d_dst, int M, int N)
{
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(CEIL(M, threadsPerBlock.x), CEIL(N, threadsPerBlock.y));

  repmatKernel<<<numBlocks, threadsPerBlock>>>(d_src, d_dst, M, N);
}

__global__ void sigmoidKernel(const real *d_input, real *d_output, int M, int N)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N)
  {
    d_output[col * M + row] = 1.0 / (1.0 + exp(-d_input[col * M + row]));
  }
}

void GPUsigmoid(const real *d_input, real *d_output, int M, int N)
{
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(CEIL(M, threadsPerBlock.x), CEIL(N, threadsPerBlock.y));

  sigmoidKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, M, N);
}

__global__ void softmaxKernel(const real *d_input, real *d_output, int M, int N)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col < N)
  {
    // Find max for each column to ensure numerical stability
    real max_val = -INFINITY;
    for (int row = 0; row < M; row++)
    {
      max_val = max(max_val, d_input[col * M + row]);
    }

    // Exponentiate and compute sum
    real sum = 0.0;
    for (int row = 0; row < M; row++)
    {
      d_output[col * M + row] = exp(d_input[col * M + row] - max_val); // subtract max_val for numerical stability
      sum += d_output[col * M + row];
    }

    // Normalize to ensure the sum is 1
    for (int row = 0; row < M; row++)
    {
      d_output[col * M + row] /= sum;
    }
  }
}

void GPUsoftmax(const real *d_input, real *d_output, int M, int N)
{
  dim3 threadsPerBlock(256);
  dim3 numBlocks(CEIL(N, threadsPerBlock.x));

  softmaxKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, M, N);
}

__global__ void transposeKernel(const real *d_input, real *d_output, int M, int N)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N)
  {
    d_output[row * N + col] = d_input[col * M + row];
  }
}

void GPUtranspose(const real *d_input, real *d_output, int M, int N)
{
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(CEIL(M, threadsPerBlock.x), CEIL(N, threadsPerBlock.y));

  transposeKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, M, N);
}

__global__ void elementwiseAddKernel(const real *d_A, const real *d_B, real *d_C, real alpha, real beta, int M, int N)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N)
  {
    d_C[col * M + row] = alpha * d_A[col * M + row] + beta * d_B[col * M + row];
  }
}

void GPUadd(const real *d_A, const real *d_B, real *d_C, real alpha, real beta, int M, int N)
{
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(CEIL(M, threadsPerBlock.x), CEIL(N, threadsPerBlock.y));
  elementwiseAddKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, alpha, beta, M, N);
}

__global__ void columnSumKernel(const real *d_input, real *d_output, int M, int N)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M)
  {
    real sum = 0;
    for (int col = 0; col < N; col++)
    {
      sum += d_input[row + col * M];
    }
    d_output[row] = sum;
  }
}

void GPUcolumnSum(const real *d_input, real *d_output, int M, int N)
{
  dim3 threadsPerBlock(256);
  dim3 numBlocks(CEIL(N, threadsPerBlock.x));

  columnSumKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, M, N);
}

__global__ void hadamardProductKernel(const real *d_a, const real *d_b, real *d_output, int M, int N)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N)
  {
    int index = col * M + row;
    d_output[index] = d_a[index] * d_b[index];
  }
}

void GPUhadamardProduct(const real *d_a, const real *d_b, real *d_output, int M, int N)
{
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(CEIL(M, threadsPerBlock.x), CEIL(N, threadsPerBlock.y));

  hadamardProductKernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_output, M, N);
}
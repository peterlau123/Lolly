#include "fmt/color.h"
#include "lolly/parallel/basic.h"

#include <cmath>
#include <cuda_runtime.h>

// please refer to https://zhuanlan.zhihu.com/p/654027980 for more details

using namespace Lolly::parallel;

__global__ void reduce_sum(float *input, float *output, int offset) {
  //int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //atomicAdd(output, input[idx] + input[idx + offset]);
  int tid = threadIdx.x;
  float* dv = input + blockDim.x*blockIdx.x;
  for(int offset = blockDim>>1;0<offset;offset>>1)
  {
    if(tid<offset)
    {
      dv[tid] + = dv[tid+offset];
    }
    __syncthreads();
  }
  if(0==tid)
  {
    *output+=dv[0];
  }
}

__global__ void reduce_max(float *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    atomicMax(output, input[idx]);
  }
}

__global__ void reduce_min(float *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    atomicMin(output, input[idx]);
  }
}

void Lolly::reduce(float *input, float **out, int size, ReduceType::Type type) {
  if (nullptr == input || out == nullptr || size <= 0) {
    return;
  }
  if (nullptr == *out) {
    *out = new float[size];
    fmt::print(fg(fmt::color::yellow) | fmt::emphasis::bold,
               "out is not allocated outside, allocate it inside reduce");
  }

  cudaError_t err = cudaSuccess;

  // Allocate device memory
  float *d_input = nullptr;
  float *d_output = nullptr;
  err = cudaMalloc((void **)&d_input, sizeof(float));
  if (err != cudaSuccess) {
    fprintf(
        stderr,
        "Failed to allocate device memory for input array (error code %s)!\n",
        cudaGetErrorString(err));
    return;
  }

  err = cudaMalloc((void **)&d_output, sizeof(float));
  if (err != cudaSuccess) {
    fprintf(
        stderr,
        "Failed to allocate device memory for output array (error code %s)!\n",
        cudaGetErrorString(err));
    cudaFree(d_input);
    return;
  }
  cudaMemset(d_output, 0, 1);

  cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);

  int MAX_BLOCK_THREADS = 256;

  //int half_len = std::ceil(size * 1.0 / 2); // TODO:变为2的倍数
  int NUM_OF_BLOCKS = (size + MAX_BLOCK_THREADS-1) / MAX_BLOCK_THREADS;

  dim3 blockDim(MAX_BLOCK_THREADS, 1, 1);
  dim3 gridDim(NUm_OF_BLOCKS, 1, 1);
  // Launch the kernel
  switch (type) {
  case ReduceType::SUM: {
    reduce_sum<<<gridDim, blockDim>>>(d_input, d_output);
    break;
  }
  case ReduceType::MAX:
    reduce_max(d_input, d_output, size);
    break;
  case ReduceType::MIN:
    reduce_min(d_input, d_output, size);
    break;
  default:
    fprintf(stderr, "Invalid reduce type!\n");
    return;
  }
  cudaFree(d_input);
  cudaFree(d_output);
}
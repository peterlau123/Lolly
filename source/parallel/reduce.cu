#include "lolly/parallel/basic.h"
#include "fmt/color.h"

#include <cmath>
#include <cuda_runtime.h>

// please refer to https://zhuanlan.zhihu.com/p/654027980 for more details

using namespace Lolly::parallel;


__global__ void  reduce_sum(float *input, float *output, int offset) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  atomicAdd(output, input[idx]+input[idx+offset]);
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
  cudaMemset(d_output,0 ,1 );


  err = cudaMalloc((void **)&d_output, size * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(
        stderr,
        "Failed to allocate device memory for output array (error code %s)!\n",
        cudaGetErrorString(err));
    cudaFree(d_input);
    return;
  }

  cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);

  int MAX_BLOCK_X_DIMENSION = 1024;
  int MAX_BLOCK_THREADS = 1024;

  int half_len = std::ceil(size*1.0/2);//TODO:变为2的倍数
  int num_of_blocks = std:ceil(half_len*1.0/MAX_BLOCK_THREADS);

  // Launch the kernel
  switch (type) {
    case ReduceType::SUM:{
      int loop_cnt = std::ceil(num_of_blocks/MAX_BLOCK_X_DIMENSION);
      int offset = half_len;
      for(int i=0;i<loop_cnt;i++){
          int block_size = std::min(MAX_BLOCK_X_DIMENSION, num_of_blocks - i * MAX_BLOCK_X_DIMENSION);
          reduce_sum<<<block_size,MAX_BLOCK_THREADS>>>(d_input, d_output, offset);//range size
          offset>>1;
      }
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
      cudaFree(d_input);
      cudaFree(d_output);
      return;
  }
}
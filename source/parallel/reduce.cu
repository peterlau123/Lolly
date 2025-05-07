#include "fmt/color.h"
#include "lolly/parallel/basic.h"

#include <cmath>
#include <cuda_runtime.h>

// please refer to https://zhuanlan.zhihu.com/p/654027980 for more details

using namespace Lolly::parallel;

template <typename Op, typename Type>
__device__ Type operator()(Type a, Type b, Op op) {
  return op(a, b);
}

template <typename Type> struct Sum {
  __device__ Type operator()(Type a, Type b) { return a + b; }
};

template <typename Type> struct Max {
  __device__ Type operator()(Type a, Type b) { return max(a, b); }
};

template <typename Type> struct Min {
  __device__ Type operator()(Type a, Type b) { return min(a, b); }
};

#if 0
//use global memory
template <typename Op>
__global__ void reduce(float *input, int len, float *output, Op op) {
  int tid = threadIdx.x;
  float *dv = input + blockDim.x * blockIdx.x;
  for (int offset = blockDim >> 1; 0 < offset; offset >> 1) {
    if (tid < offset) {
      op(dv[tid], dv[tid + offset]);
    }
    __syncthreads();
  }
  if (0 == tid) {
    //*output+=dv[0];//may have problem when different thread blocks
    atomicAdd(output, dv[0]);
  }
}
#else
// use shared memory
extern __shared__ float sdata[];
template <typename Op>
__global__ void reduce(float *input, int len, float *output, Op op) {
  int tid = threadIdx.x;
  float *dv = input + blockDim.x * blockIdx.x;
  sdata[tid] = tid < len ? dv[tid] : 0.0f;
  __syncthreads();

  for (int offset = blockDim >> 1; 0 < offset; offset >> 1) {
    if (tid < offset) {
      op(sdata[tid], sdata[tid + offset]);
    }
    __syncthreads();
  }

  if (0 == tid) {
    atomicAdd(output, sdata[0]);
  }
}
#endif

void Lolly::reduce(float *input, float **out, int len, ReduceType::Type type) {
  if (nullptr == input || out == nullptr || len <= 0) {
    return;
  }
  if (nullptr == *out) {
    *out = new float[len];
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

  cudaMemcpy(d_input, input, len * sizeof(float), cudaMemcpyHostToDevice);

  int MAX_BLOCK_THREADS = 256;

  // int half_len = std::ceil(size * 1.0 / 2); // TODO:变为2的倍数
  int NUM_OF_BLOCKS = (len + MAX_BLOCK_THREADS - 1) / MAX_BLOCK_THREADS;

  dim3 blockDim(MAX_BLOCK_THREADS, 1, 1);
  dim3 gridDim(NUm_OF_BLOCKS, 1, 1);
  // Launch the kernel
  switch (type) {
  case ReduceType::SUM: {
    reduce<<<gridDim, blockDim>>>(d_input, len, d_output, Sum());
    break;
  }
  case ReduceType::MAX: {
    reduce<<<gridDim, blockDim>>>(d_input, len, d_output, Max());
    break;
  }
  case ReduceType::MIN: {
    reduce<<<gridDim, blockDim>>>(d_input, len, d_output, Min());
    break;
  }
  default:
    fprintf(stderr, "Invalid reduce type!\n");
    return;
  }
  cudaFree(&d_input);
  cudaFree(d_output);
}
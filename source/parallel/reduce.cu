#include "lolly/parallel/basic.h"

#include <cuda_runtime.h>

//please refer to https://zhuanlan.zhihu.com/p/654027980 for more details

using namespace Lolly::parallel;

void Lolly::reduce(float *left, float **out, int size, ReduceType::Type type) {
  if (nullptr == left || out == nullptr || size <= 0) {
    return;
  }
  if (nullptr == *out) {
    *out = new float[size];
  }

  cudaError_t err = cudaSuccess;

  // Allocate device memory
  float *d_input = nullptr;
  float *d_output = nullptr;
  err = cudaMalloc((void **)&d_input, size * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(
        stderr,
        "Failed to allocate device memory for input array (error code %s)!\n",
        cudaGetErrorString(err));
    return;
  }
  err = cudaMalloc((void **)&d_output, size * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(
        stderr,
        "Failed to allocate device memory for output array (error code %s)!\n",
        cudaGetErrorString(err));
    cudaFree(d_input);
    return;
  }

  switch (type) {}
}
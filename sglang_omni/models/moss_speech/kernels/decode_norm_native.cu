// SPDX-License-Identifier: Apache-2.0
// Isolated decode-only pointwise arithmetic. Native torch.mean remains
// external.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void square_kernel(const __nv_bfloat16 *x, float *square, int groups,
                              int width, int xs, int ss) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= groups * width)
    return;
  int r = i / width, c = i % width;
  float value = __bfloat162float(x[r * xs + c]);
  square[r * ss + c] = __fmul_rn(value, value);
}

template <bool debug>
__global__ void
finish_kernel(const __nv_bfloat16 *x, const float *mean,
              const __nv_bfloat16 *weight, __nv_bfloat16 *output, int groups,
              int width, int xs, int os, float eps, float *added,
              float *inverse, float *scaled, __nv_bfloat16 *rounded) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= groups * width)
    return;
  int r = i / width, c = i % width;
  float value = __bfloat162float(x[r * xs + c]);
  float sum = __fadd_rn(mean[r], eps);
  float inv = ::rsqrtf(sum);
  float scale = __fmul_rn(value, inv);
  __nv_bfloat16 round = __float2bfloat16_rn(scale);
  float product =
      __fmul_rn(__bfloat162float(weight[c]), __bfloat162float(round));
  output[r * os + c] = __float2bfloat16_rn(product);
  if constexpr (debug) {
    if (c == 0) {
      added[r] = sum;
      inverse[r] = inv;
    }
    scaled[i] = scale;
    rounded[i] = round;
  }
}

static bool geometry(int groups, int width, int xs, int ss) {
  return ((groups == 1 && width == 4096) ||
          ((groups == 8 || groups == 32) && width == 128)) &&
         xs >= width && ss >= width;
}
extern "C" int norm_abi_version() { return 1; }
extern "C" int norm_square(const void *x, void *square, int groups, int width,
                           int xs, int ss, void *stream) {
  if (!x || !square || !geometry(groups, width, xs, ss))
    return -1;
  square_kernel<<<(groups * width + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
      (const __nv_bfloat16 *)x, (float *)square, groups, width, xs, ss);
  return (int)cudaGetLastError();
}
extern "C" int norm_finish(const void *x, const void *mean, const void *weight,
                           void *output, int groups, int width, int xs, int os,
                           float eps, void *stream, void *added, void *inverse,
                           void *scaled, void *rounded) {
  if (!x || !mean || !weight || !output || !geometry(groups, width, xs, os) ||
      !(eps > 0.0f))
    return -1;
  bool debug = added && inverse && scaled && rounded;
  if ((added || inverse || scaled || rounded) && !debug)
    return -2;
  if (debug)
    finish_kernel<true>
        <<<(groups * width + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
            (const __nv_bfloat16 *)x, (const float *)mean,
            (const __nv_bfloat16 *)weight, (__nv_bfloat16 *)output, groups,
            width, xs, os, eps, (float *)added, (float *)inverse,
            (float *)scaled, (__nv_bfloat16 *)rounded);
  else
    finish_kernel<false>
        <<<(groups * width + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
            (const __nv_bfloat16 *)x, (const float *)mean,
            (const __nv_bfloat16 *)weight, (__nv_bfloat16 *)output, groups,
            width, xs, os, eps, nullptr, nullptr, nullptr, nullptr);
  return (int)cudaGetLastError();
}

extern "C" int norm_cuda_version() { return CUDART_VERSION; }

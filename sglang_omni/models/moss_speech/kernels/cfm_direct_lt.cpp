// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <new>
#include <vector>

namespace moss_cfm_lt {

constexpr size_t WORKSPACE_BYTES = 1 << 20;
constexpr int QUALIFIED_LT_VERSION = 130101;
struct Projection {
  cublasLtMatmulDesc_t descriptor = nullptr;
  cublasLtMatrixLayout_t weight_layout = nullptr, input_layout = nullptr,
                         output_layout = nullptr;
  cublasLtMatmulAlgo_t algorithm{};
  const float *weight = nullptr;
};
struct Context {
  cublasLtHandle_t handle = nullptr;
  void *workspace = nullptr;
  cudaStream_t stream = nullptr;
  int device = 0;
  std::vector<Projection *> projections;
};
static int release_projection(Projection *projection) {
  int status = 0;
  for (auto layout : {projection->weight_layout, projection->input_layout,
                      projection->output_layout}) {
    if (layout) {
      int current = cublasLtMatrixLayoutDestroy(layout);
      if (!status)
        status = current;
    }
  }
  if (projection->descriptor) {
    int current = cublasLtMatmulDescDestroy(projection->descriptor);
    if (!status)
      status = current;
  }
  delete projection;
  return status;
}
extern "C" int lt_context_destroy(Context *context) {
  if (!context)
    return 0;
  int previous = 0;
  cudaGetDevice(&previous);
  cudaSetDevice(context->device);
  int status = static_cast<int>(cudaDeviceSynchronize());
  for (auto projection : context->projections) {
    int current = release_projection(projection);
    if (!status)
      status = current;
  }
  if (context->workspace) {
    int current = static_cast<int>(cudaFree(context->workspace));
    if (!status)
      status = current;
  }
  if (context->handle) {
    int current = cublasLtDestroy(context->handle);
    if (!status)
      status = current;
  }
  cudaSetDevice(previous);
  delete context;
  return status;
}
extern "C" int lt_abi_version() { return 1; }
extern "C" int lt_header_version() { return CUBLAS_VERSION; }
extern "C" int lt_loaded_version() { return cublasLtGetVersion(); }
extern "C" int lt_context_create(Context **output, void *stream) {
  if (!output || CUBLAS_VERSION != QUALIFIED_LT_VERSION ||
      cublasLtGetVersion() != QUALIFIED_LT_VERSION)
    return -1;
  *output = nullptr;
  auto context = new (std::nothrow) Context;
  if (!context)
    return -2;
  cudaGetDevice(&context->device);
  context->stream = static_cast<cudaStream_t>(stream);
  cudaDeviceProp properties{};
  int status =
      static_cast<int>(cudaGetDeviceProperties(&properties, context->device));
  if (!status && (properties.major != 8 || properties.minor != 0))
    status = -3;
  if (!status)
    status = cublasLtCreate(&context->handle);
  if (!status)
    status = static_cast<int>(cudaMalloc(&context->workspace, WORKSPACE_BYTES));
  if (status) {
    lt_context_destroy(context);
    return status;
  }
  *output = context;
  return 0;
}
extern "C" int lt_projection_create(Context *context, const void *weight,
                                    const void *bias, int rows, int width,
                                    int features, Projection **output) {
  if (!context || !weight || !output ||
      !(rows == 2784 || rows == 2800 || rows == 2848 || rows == 3000))
    return -1;
  const bool affine_output = features == 256 && (width == 512 || width == 1024);
  const bool affine_first = features == 1024 && width == 256;
  const bool qkv = features == 512 && width == 256;
  if (!(affine_output || affine_first || qkv) || ((bias != nullptr) == qkv))
    return -2;
  *output = nullptr;
  auto projection = new (std::nothrow) Projection;
  if (!projection)
    return -3;
  projection->weight = static_cast<const float *>(weight);
  int status = cublasLtMatmulDescCreate(&projection->descriptor,
                                        CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasOperation_t transpose = CUBLAS_OP_T;
  if (!status)
    status = cublasLtMatmulDescSetAttribute(projection->descriptor,
                                            CUBLASLT_MATMUL_DESC_TRANSA,
                                            &transpose, sizeof(transpose));
  if (bias && !status) {
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    status = cublasLtMatmulDescSetAttribute(projection->descriptor,
                                            CUBLASLT_MATMUL_DESC_EPILOGUE,
                                            &epilogue, sizeof(epilogue));
    if (!status)
      status = cublasLtMatmulDescSetAttribute(projection->descriptor,
                                              CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                              &bias, sizeof(bias));
  }
  if (!status)
    status = cublasLtMatrixLayoutCreate(&projection->weight_layout, CUDA_R_32F,
                                        width, features, width);
  if (!status)
    status = cublasLtMatrixLayoutCreate(&projection->input_layout, CUDA_R_32F,
                                        width, rows, width);
  if (!status)
    status = cublasLtMatrixLayoutCreate(&projection->output_layout, CUDA_R_32F,
                                        features, rows, features);
  if (!status)
    status = cublasLtMatmulAlgoInit(
        context->handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F,
        CUDA_R_32F, CUDA_R_32F, 0, &projection->algorithm);
  const unsigned tile = qkv ? 16 : affine_first ? 20 : 14;
  const unsigned split = qkv ? (rows == 3000 ? 2 : 3) : 1;
  const unsigned reduction = qkv ? 1 : 0;
  for (auto attribute :
       {CUBLASLT_ALGO_CONFIG_TILE_ID, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
        CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME}) {
    const unsigned value = attribute == CUBLASLT_ALGO_CONFIG_TILE_ID ? tile
                           : attribute == CUBLASLT_ALGO_CONFIG_SPLITK_NUM
                               ? split
                               : reduction;
    if (!status)
      status = cublasLtMatmulAlgoConfigSetAttribute(
          &projection->algorithm, attribute, &value, sizeof(value));
  }
  cublasLtMatmulHeuristicResult_t result{};
  if (!status)
    status = cublasLtMatmulAlgoCheck(
        context->handle, projection->descriptor, projection->weight_layout,
        projection->input_layout, projection->output_layout,
        projection->output_layout, &projection->algorithm, &result);
  if (!status)
    status = result.state;
  if (!status && result.workspaceSize > WORKSPACE_BYTES)
    status = -4;
  if (status) {
    release_projection(projection);
    return status;
  }
  try {
    context->projections.push_back(projection);
  } catch (const std::bad_alloc &) {
    release_projection(projection);
    return -5;
  }
  *output = projection;
  return 0;
}
extern "C" int lt_projection_destroy(Context *context, Projection *projection) {
  if (!context || !projection)
    return -1;
  auto found = std::find(context->projections.begin(),
                         context->projections.end(), projection);
  if (found == context->projections.end())
    return -2;
  int status = static_cast<int>(cudaDeviceSynchronize());
  if (status)
    return status;
  context->projections.erase(found);
  return release_projection(projection);
}
extern "C" int lt_run(Context *context, Projection *projection,
                      const void *input, void *output, void *stream) {
  int device = -1;
  if (!context || !projection || !input || !output ||
      cudaGetDevice(&device) != cudaSuccess || device != context->device ||
      static_cast<cudaStream_t>(stream) != context->stream)
    return -1;
  if (std::find(context->projections.begin(), context->projections.end(),
                projection) == context->projections.end())
    return -2;
  const float alpha = 1.0f, beta = 0.0f;
  return cublasLtMatmul(context->handle, projection->descriptor, &alpha,
                        projection->weight, projection->weight_layout, input,
                        projection->input_layout, &beta, output,
                        projection->output_layout, output,
                        projection->output_layout, &projection->algorithm,
                        context->workspace, WORKSPACE_BYTES, context->stream);
}

} // namespace moss_cfm_lt

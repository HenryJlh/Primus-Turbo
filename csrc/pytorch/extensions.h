// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/HIPGeneratorImpl.h>
#include <c10/macros/Macros.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <ATen/hip/HIPGraphsUtils.cuh>

#include "primus_turbo/common.h"

namespace primus_turbo::pytorch {

at::Tensor grouped_gemm(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                        at::Tensor &group_offs, const bool transA, const bool transB,
                        c10::optional<int64_t> num_cu);

at::Tensor grouped_gemm_meta(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                             at::Tensor &group_offs, const bool transA, const bool transB,
                             c10::optional<int64_t> num_cu);

at::Tensor grouped_gemm_variable_k(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                                   at::Tensor &group_offs, const bool transA, const bool transB,
                                   c10::optional<int64_t> num_cu);

at::Tensor grouped_gemm_variable_k_meta(at::Tensor &a, at::Tensor &b, at::Tensor &group_lens,
                                        at::Tensor &group_offs, const bool transA,
                                        const bool transB, c10::optional<int64_t> num_cu);

at::Tensor grouped_gemm_compute_offs(at::Tensor &group_lens);

at::Tensor grouped_gemm_compute_offs_meta(at::Tensor &group_lens);

} // namespace primus_turbo::pytorch

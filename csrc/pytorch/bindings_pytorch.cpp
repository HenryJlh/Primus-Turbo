// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// See LICENSE for license information.

#include <torch/extension.h>

#include "extensions.h"

namespace primus_turbo::pytorch {

TORCH_LIBRARY(group_gemm_primus_cpp_extension, m) {
    m.def("grouped_gemm(Tensor a, Tensor b, Tensor group_lens, Tensor group_offs, bool transA, "
          "bool transB, int? num_cu=None) -> Tensor");
    m.def("grouped_gemm_variable_k(Tensor a, Tensor b, Tensor group_lens, Tensor group_offs, "
          "bool transA, bool transB, int? num_cu=None) -> Tensor");
    m.def("grouped_gemm_compute_offs(Tensor group_lens) -> Tensor");
}

TORCH_LIBRARY_IMPL(group_gemm_primus_cpp_extension, CUDA, m) {
    m.impl("grouped_gemm", grouped_gemm);
    m.impl("grouped_gemm_variable_k", grouped_gemm_variable_k);
    m.impl("grouped_gemm_compute_offs", grouped_gemm_compute_offs);
}

TORCH_LIBRARY_IMPL(group_gemm_primus_cpp_extension, Meta, m) {
    m.impl("grouped_gemm", grouped_gemm_meta);
    m.impl("grouped_gemm_variable_k", grouped_gemm_variable_k_meta);
    m.impl("grouped_gemm_compute_offs", grouped_gemm_compute_offs_meta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    (void)m;
}

} // namespace primus_turbo::pytorch

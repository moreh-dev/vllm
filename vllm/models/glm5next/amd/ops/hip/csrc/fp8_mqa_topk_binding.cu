// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "aiter_torch.h"
#include "fp8_mqa_topk.h"

using vllm_aiter::to_aiter;

namespace {

int64_t fp8_mqa_topk_workspace_size(int64_t m, int64_t n,
                                    int64_t total_cu_count) {
  aiter_detail::AiterThrowGuard throw_guard;
  return aiter::fp8_mqa_topk_workspace_size(m, n, total_cu_count);
}

void fp8_mqa_topk(const at::Tensor& q_fp8, const at::Tensor& k_fp8,
                  const at::Tensor& kv_scale, const at::Tensor& weights,
                  const at::Tensor& cu_seqlen_ks,
                  const at::Tensor& cu_seqlen_ke, at::Tensor& topk_indices,
                  at::Tensor& workspace, int64_t total_cu_count,
                  const std::optional<at::Tensor>& fallback_rows,
                  bool force_fallback) {
  vllm_aiter::EntryScope scope(topk_indices);
  aiter_tensor_t indices = to_aiter(topk_indices);
  aiter_tensor_t scratch = to_aiter(workspace);
  aiter::fp8_mqa_topk(to_aiter(q_fp8), to_aiter(k_fp8), to_aiter(kv_scale),
                      to_aiter(weights), to_aiter(cu_seqlen_ks),
                      to_aiter(cu_seqlen_ke), indices, scratch, total_cu_count,
                      to_aiter(fallback_rows), force_fallback);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_mqa_topk_workspace_size", &fp8_mqa_topk_workspace_size);
  m.def("fp8_mqa_topk", &fp8_mqa_topk);
}

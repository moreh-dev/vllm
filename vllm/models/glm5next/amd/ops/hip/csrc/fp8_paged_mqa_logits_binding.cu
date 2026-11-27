// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "aiter_torch.h"
#include "fp8_paged_mqa_logits.h"

using vllm_aiter::to_aiter;

namespace {

void fp8_paged_mqa_logits(at::Tensor& out, const at::Tensor& q_fp8,
                          const at::Tensor& kv_cache_fp8,
                          const at::Tensor& weights,
                          const at::Tensor& context_lens,
                          const at::Tensor& block_tables, int64_t max_model_len,
                          int64_t total_cu_count) {
  vllm_aiter::EntryScope scope(out);
  aiter_tensor_t logits = to_aiter(out);
  // ChunkK, SplitKV, num_warps and RowsPerBlock take the host heuristic.
  aiter::fp8_paged_mqa_logits(logits, to_aiter(q_fp8), to_aiter(kv_cache_fp8),
                              to_aiter(weights), to_aiter(context_lens),
                              to_aiter(block_tables), max_model_len, 0, 0, 0,
                              total_cu_count, 0);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_paged_mqa_logits", &fp8_paged_mqa_logits);
}

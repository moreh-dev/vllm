// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "aiter_torch.h"
#include "fp8_mqa_logits.h"

using vllm_aiter::to_aiter;

namespace {

void fp8_mqa_logits(at::Tensor& out, const at::Tensor& q_fp8,
                    const at::Tensor& k_fp8, const at::Tensor& kv_scale,
                    const at::Tensor& weights, const at::Tensor& cu_seqlen_ks,
                    const at::Tensor& cu_seqlen_ke, int64_t total_cu_count,
                    bool clean_logits) {
  vllm_aiter::EntryScope scope(out);
  aiter_tensor_t logits = to_aiter(out);
  // BlockM, SplitN, num_warps and reverse_rows take the host heuristic.
  aiter::fp8_mqa_logits(logits, to_aiter(q_fp8), to_aiter(k_fp8),
                        to_aiter(kv_scale), to_aiter(weights),
                        to_aiter(cu_seqlen_ks), to_aiter(cu_seqlen_ke), 0, 0, 0,
                        total_cu_count, clean_logits, -1);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp8_mqa_logits", &fp8_mqa_logits);
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "aiter_torch.h"

using vllm_aiter::to_aiter;

// C entry points of the vendored sparse_mla_fp8.cu (AITER ctypes ABI v2:
// status 0 on success, else the message is in aiter_get_last_error()).
extern "C" int sparse_mla_fp8_fwd_out(
    aiter_tensor_t* q, aiter_tensor_t* kv_buffer, aiter_tensor_t* kv_indices,
    aiter_tensor_t* out, aiter_tensor_t* lse, aiter_tensor_t* partial_o,
    aiter_tensor_t* partial_lse, float softmax_scale, float q_scale,
    float kv_scale, hipStream_t stream);
extern "C" const char* aiter_get_last_error();

namespace {

void sparse_mla_fp8_fwd(const at::Tensor& q, const at::Tensor& kv_buffer,
                        const at::Tensor& kv_indices, at::Tensor& out,
                        const std::optional<at::Tensor>& lse,
                        at::Tensor& partial_o, at::Tensor& partial_lse,
                        double softmax_scale, double q_scale, double kv_scale) {
  vllm_aiter::EntryScope scope(out);
  aiter_tensor_t q_t = to_aiter(q);
  aiter_tensor_t kv_t = to_aiter(kv_buffer);
  aiter_tensor_t ids_t = to_aiter(kv_indices);
  aiter_tensor_t out_t = to_aiter(out);
  std::optional<aiter_tensor_t> lse_t = to_aiter(lse);
  aiter_tensor_t po_t = to_aiter(partial_o);
  aiter_tensor_t pl_t = to_aiter(partial_lse);
  const int status = sparse_mla_fp8_fwd_out(
      &q_t, &kv_t, &ids_t, &out_t, lse_t ? &*lse_t : nullptr, &po_t, &pl_t,
      static_cast<float>(softmax_scale), static_cast<float>(q_scale),
      static_cast<float>(kv_scale), scope.stream());
  if (status != 0) {
    const char* message = aiter_get_last_error();
    TORCH_CHECK(false, "sparse_mla_fp8: ",
                message != nullptr ? message : "unknown error");
  }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_mla_fp8_fwd", &sparse_mla_fp8_fwd);
}

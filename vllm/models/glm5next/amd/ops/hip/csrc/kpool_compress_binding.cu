// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "aiter_torch.h"
#include "kpool_compress.h"

using vllm_aiter::to_aiter;

namespace {

void kpool_compress_insert(at::Tensor& kv_cache, const at::Tensor& k,
                           const at::Tensor& gate_score, const at::Tensor& ape,
                           const at::Tensor& slot_mapping, int64_t pool_size,
                           bool round_scale, int64_t lanes_per_pool) {
  vllm_aiter::EntryScope scope(kv_cache);
  aiter_tensor_t cache = to_aiter(kv_cache);
  aiter::kpool_compress_insert(cache, to_aiter(k), to_aiter(gate_score),
                               to_aiter(ape), to_aiter(slot_mapping),
                               static_cast<int>(pool_size), round_scale,
                               static_cast<int>(lanes_per_pool));
}

void kpool_decode_update(at::Tensor& kv_cache, at::Tensor& tail_kv_cache,
                         const at::Tensor& tail_slot_mapping,
                         const at::Tensor& key, const at::Tensor& slot_score,
                         const at::Tensor& ape, const at::Tensor& slot_mapping,
                         const at::Tensor& positions, int64_t pool_size,
                         bool round_scale) {
  vllm_aiter::EntryScope scope(kv_cache);
  aiter_tensor_t cache = to_aiter(kv_cache);
  aiter_tensor_t tail = to_aiter(tail_kv_cache);
  aiter::kpool_decode_update(cache, tail, to_aiter(tail_slot_mapping),
                             to_aiter(key), to_aiter(slot_score), to_aiter(ape),
                             to_aiter(slot_mapping), to_aiter(positions),
                             static_cast<int>(pool_size), round_scale);
}

void kpool_mixed_write(
    at::Tensor& kv_cache, const at::Tensor& ape, int64_t pool_size,
    bool round_scale, int64_t lanes_per_pool,
    const std::optional<at::Tensor>& prefill_k,
    const std::optional<at::Tensor>& prefill_gate,
    const std::optional<at::Tensor>& prefill_slot_mapping,
    const std::optional<at::Tensor>& prefill_tail_slot_mapping,
    const std::optional<at::Tensor>& tail_kv_cache,
    const std::optional<at::Tensor>& decode_tail_slot_mapping,
    const std::optional<at::Tensor>& decode_key,
    const std::optional<at::Tensor>& decode_gate,
    const std::optional<at::Tensor>& decode_slot_mapping,
    const std::optional<at::Tensor>& decode_positions) {
  vllm_aiter::EntryScope scope(kv_cache);
  aiter_tensor_t cache = to_aiter(kv_cache);
  aiter::kpool_mixed_write(
      cache, to_aiter(ape), static_cast<int>(pool_size), round_scale,
      static_cast<int>(lanes_per_pool), to_aiter(prefill_k),
      to_aiter(prefill_gate), to_aiter(prefill_slot_mapping),
      to_aiter(prefill_tail_slot_mapping), to_aiter(tail_kv_cache),
      to_aiter(decode_tail_slot_mapping), to_aiter(decode_key),
      to_aiter(decode_gate), to_aiter(decode_slot_mapping),
      to_aiter(decode_positions));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kpool_compress_insert", &kpool_compress_insert);
  m.def("kpool_decode_update", &kpool_decode_update);
  m.def("kpool_mixed_write", &kpool_mixed_write);
}

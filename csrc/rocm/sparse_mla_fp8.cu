// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// FP8 sparse MLA with fused BF16 query conversion and BF16 output, gfx950
// (GLM-5.x sparse attention over the shared FP8 latent K/V pool, H64 / D512,
// no appended RoPE):
//
//   O[q] = softmax(scale * Q[q] . KV[ids[q]]^T) . KV[ids[q]]
//
// Q is BF16 (encoded to FP8 as Q / q_scale inside the kernel) or FP8;
// kv_buffer is the flat FP8 [slots, 512] pool; kv_indices [Q, K] holds
// physical slot ids (negative / out-of-range ids are masked, K <= 2176);
// scores and values are decoded with q_scale and kv_scale; P is FP8 with
// scale 1/256; accumulation, softmax and LSE are FP32; split partials and
// the output are BF16. The kernels and the split dispatch are shared verbatim
// with aiter's csrc/kernels/mla/fp8 (sparse_mla_fp8_fwd); only this entry
// point differs: it validates torch tensors and always writes into `out`.
// Build flags (CMakeLists.txt, object library _rocm_C_sparse_mla_fp8):
//   -fno-gpu-flush-denormals-to-zero  gradual underflow in the FP8 encode

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "rocm/ops.h"
#include "rocm/sparse_mla_fp8/params.cuh"

#include <climits>
#include <cmath>
#include <cstdint>

namespace {

void check_tensor(const torch::Tensor& t, const char* name, at::ScalarType dtype,
                  const torch::Device& device) {
  TORCH_CHECK(t.is_cuda() && t.device() == device, name, ": expected the query device");
  TORCH_CHECK(t.scalar_type() == dtype, name, ": unexpected dtype ", t.scalar_type());
  TORCH_CHECK(t.numel() == 0 || t.data_ptr(), name, ": null storage");
}

void check_aligned(const torch::Tensor& t, const char* name, size_t alignment) {
  TORCH_CHECK(!t.numel() || reinterpret_cast<uintptr_t>(t.data_ptr()) % alignment == 0,
              name, ": unaligned storage");
}

uintptr_t end_address(const torch::Tensor& t) {
  if (!t.numel()) return 0;
  int64_t span = 1;
  for (int d = 0; d < t.dim(); ++d) span += (t.size(d) - 1) * t.stride(d);
  return reinterpret_cast<uintptr_t>(t.data_ptr()) + span * t.element_size();
}

void check_no_overlap(const torch::Tensor& a, const torch::Tensor& b) {
  if (!a.numel() || !b.numel()) return;
  TORCH_CHECK(reinterpret_cast<uintptr_t>(a.data_ptr()) >= end_address(b) ||
                  reinterpret_cast<uintptr_t>(b.data_ptr()) >= end_address(a),
              "sparse_mla_fp8: output/workspace storage must not overlap");
}

}  // namespace

void sparse_mla_fp8_fwd(const torch::Tensor& q,            // [Q, 64, 512] bf16 / fp8
                        const torch::Tensor& kv_buffer,    // [slots, 512] fp8
                        const torch::Tensor& kv_indices,   // [Q, K] int32
                        torch::Tensor& out,                // [Q, 64, 512] bf16
                        std::optional<torch::Tensor> lse,  // [Q, 64] fp32
                        torch::Tensor& partial_o,          // [rows * 512] bf16
                        torch::Tensor& partial_lse,        // [rows] fp32
                        double softmax_scale, double q_scale, double kv_scale) {
  TORCH_CHECK(q.is_cuda(), "q must be on a HIP device");
  const torch::Device device = q.device();
  const c10::cuda::OptionalCUDAGuard device_guard(device);
  for (double scale : {softmax_scale, q_scale, kv_scale})
    TORCH_CHECK(std::isfinite(scale) && scale > 0, "scales must be finite and positive");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 || q.scalar_type() == at::kFloat8_e4m3fn,
              "q must be BF16 or FP8 (float8_e4m3fn)");
  check_tensor(q, "q", q.scalar_type(), device);
  TORCH_CHECK(q.dim() == 3 && q.size(1) == 64 && q.size(2) == 512, "expected q=[Q,64,512]");
  const int64_t queries = q.size(0);
  TORCH_CHECK(queries <= 16384, "Q must be <= 16384");
  TORCH_CHECK(q.stride(2) == 1 && q.stride(1) >= 512 && q.stride(0) >= 63 * q.stride(1) + 512,
              "q must have non-overlapping, unit-inner-stride rows");
  const int64_t bytes = q.element_size();
  TORCH_CHECK((q.stride(0) * bytes) % 16 == 0 && (q.stride(1) * bytes) % 16 == 0,
              "q rows must be aligned to 16 bytes");
  check_aligned(q, "q", 16);
  check_tensor(kv_buffer, "kv_buffer", at::kFloat8_e4m3fn, device);
  TORCH_CHECK(kv_buffer.dim() == 2 && kv_buffer.size(1) == 512 && kv_buffer.size(0) > 0 &&
                  kv_buffer.size(0) <= INT_MAX && kv_buffer.is_contiguous(),
              "expected contiguous kv_buffer=[slots,512]");
  check_aligned(kv_buffer, "kv_buffer", 16);
  check_tensor(kv_indices, "kv_indices", at::kInt, device);
  TORCH_CHECK(kv_indices.dim() == 2 && kv_indices.size(0) == queries &&
                  kv_indices.size(1) <= 2176 && kv_indices.is_contiguous(),
              "expected contiguous kv_indices=[Q,K], K <= 2176");
  check_aligned(kv_indices, "kv_indices", 4);
  check_tensor(out, "out", at::kBFloat16, device);
  TORCH_CHECK(out.dim() == 3 && out.size(0) == queries && out.size(1) == 64 && out.size(2) == 512 &&
                  out.is_contiguous(),
              "out must be a contiguous bf16 [Q,64,512] tensor (unexpected shape)");
  check_aligned(out, "out", 16);
  if (lse) {
    check_tensor(*lse, "lse", at::kFloat, device);
    TORCH_CHECK(lse->dim() == 2 && lse->size(0) == queries && lse->size(1) == 64 && lse->is_contiguous(),
                "lse must be a contiguous fp32 [Q,64] tensor (unexpected shape)");
    check_aligned(*lse, "lse", 4);
  }
  const int splits = sparse_mla_fp8::split_count(static_cast<int>(queries));
  const int64_t rows = splits > 1 ? queries * splits * 64 : 0;
  check_tensor(partial_o, "partial_o", at::kBFloat16, device);
  check_tensor(partial_lse, "partial_lse", at::kFloat, device);
  TORCH_CHECK(partial_o.dim() == 1 && partial_o.size(0) == rows * 512 && partial_lse.dim() == 1 &&
                  partial_lse.size(0) == rows,
              "workspace has an unexpected shape: expected partial_o [", rows * 512,
              "] bf16 and partial_lse [", rows, "] fp32 for Q=", queries);
  TORCH_CHECK(partial_o.is_contiguous() && partial_lse.is_contiguous(), "workspace must be contiguous");
  check_aligned(partial_o, "partial_o", 16);
  check_aligned(partial_lse, "partial_lse", 4);
  {
    std::vector<const torch::Tensor*> writes{&out, &partial_o, &partial_lse};
    if (lse) writes.push_back(&*lse);
    for (size_t i = 0; i < writes.size(); ++i) {
      for (const torch::Tensor* input : {&q, &kv_buffer, &kv_indices}) check_no_overlap(*writes[i], *input);
      for (size_t j = 0; j < i; ++j) check_no_overlap(*writes[i], *writes[j]);
    }
  }
  if (!queries) return;

  sparse_mla_fp8::Params p{};
  p.q = static_cast<const uint8_t*>(q.data_ptr());
  p.kv = static_cast<const uint8_t*>(kv_buffer.data_ptr());
  p.indices = static_cast<const int32_t*>(kv_indices.data_ptr());
  p.partial_o = static_cast<uint16_t*>(partial_o.data_ptr());
  p.partial_lse = static_cast<float*>(partial_lse.data_ptr());
  p.out = static_cast<uint8_t*>(out.data_ptr());
  p.lse = lse ? static_cast<float*>(lse->data_ptr()) : nullptr;
  p.queries = static_cast<int>(queries);
  p.slots = static_cast<int>(kv_buffer.size(0));
  p.capacity = static_cast<int>(kv_indices.size(1));
  p.q_scale = static_cast<float>(q_scale);
  p.kv_scale = static_cast<float>(kv_scale);
  p.attention_scale = static_cast<float>(softmax_scale);
  p.q_stride_row = q.stride(0);
  p.q_stride_head = q.stride(1);
  p.bf16_q = q.scalar_type() == at::kBFloat16;

  const hipStream_t stream = at::cuda::getCurrentCUDAStream(device.index());
  hipError_t status;
  if (queries <= 6)
    status = sparse_mla_fp8::launch_small(p, stream);
  else if (queries <= 12)
    status = sparse_mla_fp8::launch_prefetch(p, stream);
  else if (queries <= 64)
    status = sparse_mla_fp8::launch_rotating(p, stream);
  else
    status = sparse_mla_fp8::launch_streaming(p, stream);
  TORCH_CHECK(status == hipSuccess, "sparse_mla_fp8_fwd launch failed: ", hipGetErrorString(status));
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Torch front end for the vendored AITER kernels (ops/third_party/aiter):
// they take `aiter_tensor_t` and read their stream from aiter's thread-local
// slot, and a failed AITER_CHECK aborts unless an AiterThrowGuard is live.
#pragma once

#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <torch/extension.h>

#include <optional>

#include "aiter_stream.h"
#include "aiter_tensor.h"

namespace vllm_aiter {

inline AiterDtype to_aiter_dtype(at::ScalarType t) {
  switch (t) {
    case at::kFloat8_e4m3fn:
      return AITER_DTYPE_fp8;
    case at::kFloat8_e8m0fnu:
      return AITER_DTYPE_fp8_e8m0;
    case at::kHalf:
      return AITER_DTYPE_fp16;
    case at::kBFloat16:
      return AITER_DTYPE_bf16;
    case at::kFloat:
      return AITER_DTYPE_fp32;
    case at::kUInt32:
      return AITER_DTYPE_u32;
    case at::kInt:
      return AITER_DTYPE_i32;
    case at::kShort:
      return AITER_DTYPE_i16;
    case at::kChar:
      return AITER_DTYPE_i8;
    case at::kByte:
      return AITER_DTYPE_u8;
    case at::kLong:
      return AITER_DTYPE_i64;
    case at::kUInt64:
      return AITER_DTYPE_u64;
    default:
      TORCH_CHECK(false, "the AITER kernels do not take dtype ", t);
  }
}

inline aiter_tensor_t to_aiter(const at::Tensor& t) {
  TORCH_CHECK(t.dim() <= 8, "aiter_tensor_t holds at most 8 dims, got ",
              t.dim());
  aiter_tensor_t a{};
  a.ptr = t.data_ptr();
  a.numel_ = static_cast<size_t>(t.numel());
  a.ndim = static_cast<int>(t.dim());
  for (int i = 0; i < a.ndim; ++i) {
    a.shape[i] = t.size(i);
    a.strides[i] = t.stride(i);
  }
  a.dtype_ = to_aiter_dtype(t.scalar_type());
  a.device_id = t.is_cuda() ? t.get_device() : -1;
  return a;
}

inline std::optional<aiter_tensor_t> to_aiter(
    const std::optional<at::Tensor>& t) {
  if (!t.has_value()) {
    return std::nullopt;
  }
  return to_aiter(*t);
}

// Scope of one call into AITER: the device of `t`, the current torch stream
// as aiter's stream, and AITER_CHECK failures thrown (and so raised in
// Python) instead of aborting the process.
class EntryScope {
 public:
  explicit EntryScope(const at::Tensor& t)
      : device_guard_(t.device()), prev_stream_(aiter::getCurrentHIPStream()) {
    aiter::setCurrentHIPStream(
        at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream());
  }
  ~EntryScope() { aiter::setCurrentHIPStream(prev_stream_); }
  EntryScope(const EntryScope&) = delete;
  EntryScope& operator=(const EntryScope&) = delete;

  hipStream_t stream() const { return aiter::getCurrentHIPStream(); }

 private:
  at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard_;
  aiter_detail::AiterThrowGuard throw_guard_;
  hipStream_t prev_stream_;
};

}  // namespace vllm_aiter

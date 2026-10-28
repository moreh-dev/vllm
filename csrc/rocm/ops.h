#pragma once

#include <torch/all.h>

torch::Tensor LLMM1(at::Tensor& in_a, at::Tensor& in_b,
                    const int64_t rows_per_block);

torch::Tensor wvSplitK(const at::Tensor& in_a, const at::Tensor& in_b,
                       const std::optional<at::Tensor>& in_bias,
                       const int64_t CuCount);

torch::Tensor wvSplitK_int4_g(const at::Tensor& in_a, const at::Tensor& in_b,
                              const at::Tensor& in_scale,
                              const std::optional<at::Tensor>& in_zero_points,
                              const std::optional<at::Tensor>& in_bias,
                              const int64_t CuCount, const int64_t group_size);

torch::Tensor wvSplitKrc(const at::Tensor& in_a, const at::Tensor& in_b,
                         const std::optional<at::Tensor>& in_bias,
                         const int64_t CuCount);

void wvSplitKQ(const at::Tensor& in_a, const at::Tensor& in_b,
               const std::optional<at::Tensor>& in_bias, at::Tensor& out_c,
               const at::Tensor& scale_a, const at::Tensor& scale_b,
               const int64_t CuCount);

torch::Tensor mxfp4_w4a8_gemv_rdna3(const at::Tensor& a, const at::Tensor& b_q,
                                    const at::Tensor& b_scale);

torch::Tensor gptq_gemm_rdna3(torch::Tensor a, torch::Tensor b_q_weight,
                              torch::Tensor b_qzeros, torch::Tensor b_scales,
                              bool use_v2_format);

torch::Tensor gptq_gemm_rdna3_wmma(torch::Tensor a, torch::Tensor b_q_weight,
                                   torch::Tensor b_qzeros,
                                   torch::Tensor b_scales, bool use_v2_format);

void moe_gptq_gemm_rdna3(torch::Tensor a, torch::Tensor c,
                         torch::Tensor b_q_weight, torch::Tensor b_scales,
                         torch::Tensor b_qzeros, torch::Tensor topk_weights,
                         torch::Tensor sorted_token_ids,
                         torch::Tensor expert_ids,
                         torch::Tensor num_tokens_post_padded, int64_t top_k,
                         int64_t block_size_m, bool mul_topk_weight,
                         int64_t output_topk);

void paged_attention(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens,
    const std::optional<torch::Tensor>& query_start_loc, int64_t block_size,
    int64_t max_seq_len, const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale, const std::optional<torch::Tensor>& fp8_out_scale,
    const std::string& mfma_type, bool use_interleaved_v_cache);

// GLM-5.3-Flash sparse-indexer kpool cache writers (csrc/rocm/kpool_compress.cu),
// byte-identical to the Triton writers in
// vllm/models/glm5next/amd/ops/kpool_compress.py. kv_cache is
// [num_blocks, page_size, 132] uint8 (128 fp8 e4m3fn keys per token in the
// MFMA 16x16 preshuffled layout when page_size > 1, then page_size fp32
// scales); ape [pool_size, 128] fp32; pool_size in {2, 3, 4, 8, 16};
// lanes_per_pool 0 (auto) or 16 / 32 / 64.
#ifndef VLLM_SKIP_KPOOL_COMPRESS
// Prefill insert: k, gate_score [n_tokens, 128] bf16 (rows may be strided),
// slot_mapping [n_tokens] int32 / int64 (>= 0 on the token that closes a pool).
void kpool_compress_insert(torch::Tensor& kv_cache, const torch::Tensor& k,
                           const torch::Tensor& gate_score,
                           const torch::Tensor& ape,
                           const torch::Tensor& slot_mapping,
                           int64_t pool_size, bool round_scale,
                           int64_t lanes_per_pool);

// Decode update: each token stashes its K / gate into the request's tail ring
// tail_kv_cache [num_blocks, 2, ring, 128] bf16 (ring a multiple of
// pool_size); a token that closes a pool compresses it into kv_cache. key,
// slot_score [num_requests, next_n, 128] bf16; tail_slot_mapping, slot_mapping,
// positions [num_requests, next_n] all int32 or all int64; next_n <= 16.
void kpool_decode_update(torch::Tensor& kv_cache, torch::Tensor& tail_kv_cache,
                         const torch::Tensor& tail_slot_mapping,
                         const torch::Tensor& key,
                         const torch::Tensor& slot_score,
                         const torch::Tensor& ape,
                         const torch::Tensor& slot_mapping,
                         const torch::Tensor& positions, int64_t pool_size,
                         bool round_scale);

// Mixed step in one launch: the prefill insert on prefill_*, the tail seeding
// (kpool_seed_tail_cache) when prefill_tail_slot_mapping [n_tokens] is given,
// and the decode update on decode_*. Either side may be absent. One index
// dtype per launch: the seed slots and the decode metadata must have the
// dtype of prefill_slot_mapping (int32 or int64).
void kpool_mixed_write(torch::Tensor& kv_cache, const torch::Tensor& ape,
                       int64_t pool_size, bool round_scale,
                       int64_t lanes_per_pool,
                       std::optional<torch::Tensor> prefill_k,
                       std::optional<torch::Tensor> prefill_gate,
                       std::optional<torch::Tensor> prefill_slot_mapping,
                       std::optional<torch::Tensor> prefill_tail_slot_mapping,
                       std::optional<torch::Tensor> tail_kv_cache,
                       std::optional<torch::Tensor> decode_tail_slot_mapping,
                       std::optional<torch::Tensor> decode_key,
                       std::optional<torch::Tensor> decode_gate,
                       std::optional<torch::Tensor> decode_slot_mapping,
                       std::optional<torch::Tensor> decode_positions);
#endif  // VLLM_SKIP_KPOOL_COMPRESS

#include "core/registration.h"
#include "rocm/ops.h"

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, rocm_ops) {
  // vLLM custom ops for rocm

// skinny_gemms.cu (LLMM1/wvSplitK/wvSplitKrc/wvSplitKQ) is excluded on gfx1250
// (gfx9/gfx11 ISA, unsupported there); skip these registrations to avoid
// undefined symbols. vLLM uses default/Triton GEMM for these ops on gfx1250.
#ifndef VLLM_SKIP_SKINNY_GEMMS
  // Custom gemm op for matrix-vector multiplication
  rocm_ops.def(
      "LLMM1(Tensor in_a, Tensor in_b, int rows_per_block) -> "
      "Tensor");
  rocm_ops.impl("LLMM1", torch::kCUDA, &LLMM1);

  // Custom gemm op for skinny matrix-matrix multiplication
  rocm_ops.def(
      "wvSplitK(Tensor in_a, Tensor in_b, Tensor? in_bias, int CuCount) -> "
      "Tensor");
  rocm_ops.impl("wvSplitK", torch::kCUDA, &wvSplitK);

  // W4A16 grouped skinny GEMM: packed int4 weights, per-group scales,
  // optional zero points [M/8, K/group_size] int32 for asymmetric
  // quantization
  rocm_ops.def(
      "wvSplitK_int4_g(Tensor in_a, Tensor in_b, Tensor in_scale, "
      "Tensor? in_zero_points, Tensor? in_bias, int CuCount, "
      "int group_size) -> Tensor");
  rocm_ops.impl("wvSplitK_int4_g", torch::kCUDA, &wvSplitK_int4_g);

  // Custom gemm op for skinny matrix-matrix multiplication
  rocm_ops.def(
      "wvSplitKrc(Tensor in_a, Tensor in_b, Tensor? in_bias, int CuCount) -> "
      "Tensor");
  rocm_ops.impl("wvSplitKrc", torch::kCUDA, &wvSplitKrc);

  // wvSplitK for fp8
  rocm_ops.def(
      "wvSplitKQ(Tensor in_a, Tensor in_b, Tensor? in_bias, Tensor! out_c, "
      "Tensor scale_a, "
      "          Tensor scale_b, int CuCount) -> ()");
  rocm_ops.impl("wvSplitKQ", torch::kCUDA, &wvSplitKQ);
#endif  // VLLM_SKIP_SKINNY_GEMMS

#ifndef VLLM_SKIP_KPOOL_COMPRESS
  // GLM-5.3-Flash sparse-indexer kpool cache writers (gfx942 / gfx950):
  // byte-identical to the Triton writers in
  // vllm/models/glm5next/amd/ops/kpool_compress.py.
  rocm_ops.def(
      "kpool_compress_insert(Tensor! kv_cache, Tensor k, Tensor gate_score, "
      "Tensor ape, Tensor slot_mapping, int pool_size, bool round_scale, "
      "int lanes_per_pool) -> ()");
  rocm_ops.impl("kpool_compress_insert", torch::kCUDA, &kpool_compress_insert);

  rocm_ops.def(
      "kpool_decode_update(Tensor! kv_cache, Tensor! tail_kv_cache, "
      "Tensor tail_slot_mapping, Tensor key, Tensor slot_score, Tensor ape, "
      "Tensor slot_mapping, Tensor positions, int pool_size, "
      "bool round_scale) -> ()");
  rocm_ops.impl("kpool_decode_update", torch::kCUDA, &kpool_decode_update);

  rocm_ops.def(
      "kpool_mixed_write(Tensor! kv_cache, Tensor ape, int pool_size, "
      "bool round_scale, int lanes_per_pool, Tensor? prefill_k, "
      "Tensor? prefill_gate, Tensor? prefill_slot_mapping, "
      "Tensor? prefill_tail_slot_mapping, Tensor!? tail_kv_cache, "
      "Tensor? decode_tail_slot_mapping, Tensor? decode_key, "
      "Tensor? decode_gate, Tensor? decode_slot_mapping, "
      "Tensor? decode_positions) -> ()");
  rocm_ops.impl("kpool_mixed_write", torch::kCUDA, &kpool_mixed_write);
#endif  // VLLM_SKIP_KPOOL_COMPRESS

#ifndef VLLM_SKIP_FP8_MQA_LOGITS
  // GLM-5.x lightning-indexer prefill logits over a gathered fp8 K buffer
  // (gfx950, n_heads=32, head_dim=128).
  rocm_ops.def(
      "fp8_mqa_logits(Tensor q_fp8, Tensor k_fp8, Tensor kv_scale, "
      "Tensor weights, Tensor cu_seqlen_ks, Tensor cu_seqlen_ke, Tensor! out, "
      "int split_n, int num_warps, int total_cu_count, bool clean_logits, "
      "int reverse_rows) -> ()");
  rocm_ops.impl("fp8_mqa_logits", torch::kCUDA, &fp8_mqa_logits);

  // GLM-5.x lightning-indexer decode logits over the paged fp8 indexer cache
  // (gfx950, n_heads=32, head_dim=128, block_size 1 or 64).
  rocm_ops.def(
      "fp8_paged_mqa_logits(Tensor q_fp8, Tensor kv_cache_fp8, Tensor weights, "
      "Tensor context_lens, Tensor block_tables, Tensor! out, "
      "int max_model_len, int chunk_k, int split_kv, int num_warps, "
      "int total_cu_count, int rows_per_block) -> ()");
  rocm_ops.impl("fp8_paged_mqa_logits", torch::kCUDA, &fp8_paged_mqa_logits);
#endif  // VLLM_SKIP_FP8_MQA_LOGITS

#ifndef VLLM_SKIP_SPARSE_MLA_FP8
  // FP8 sparse MLA (GLM-5.x sparse attention core) with fused BF16 query
  // conversion and BF16 output; gfx950, 64 heads x 512 latent dims.
  rocm_ops.def(
      "sparse_mla_fp8_fwd(Tensor q, Tensor kv_buffer, Tensor kv_indices, "
      "Tensor! out, Tensor!? lse, Tensor! partial_o, Tensor! partial_lse, "
      "float softmax_scale, float q_scale, float kv_scale) -> ()");
  rocm_ops.impl("sparse_mla_fp8_fwd", torch::kCUDA, &sparse_mla_fp8_fwd);
#endif  // VLLM_SKIP_SPARSE_MLA_FP8

#ifdef VLLM_ROCM_GFX1100
  // MXFP4 weights x int8 activations GEMV for RDNA3/RDNA3.5 (gfx11) decode,
  // 1 <= M <= 8. Weights [N, K/2] uint8 E2M1, scales [N, K/32] uint8 E8M0.
  rocm_ops.def(
      "mxfp4_w4a8_gemv_rdna3(Tensor a, Tensor b_q, Tensor b_scale) -> Tensor");
  rocm_ops.impl("mxfp4_w4a8_gemv_rdna3", torch::kCUDA, &mxfp4_w4a8_gemv_rdna3);

  // W4A16 GPTQ kernels for AMD RDNA3 (gfx1100).
  rocm_ops.def(
      "gptq_gemm_rdna3(Tensor a, Tensor b_q_weight, Tensor b_qzeros, "
      "Tensor b_scales, bool use_v2_format) -> Tensor");
  rocm_ops.impl("gptq_gemm_rdna3", torch::kCUDA, &gptq_gemm_rdna3);

  rocm_ops.def(
      "gptq_gemm_rdna3_wmma(Tensor a, Tensor b_q_weight, Tensor b_qzeros, "
      "Tensor b_scales, bool use_v2_format) -> Tensor");
  rocm_ops.impl("gptq_gemm_rdna3_wmma", torch::kCUDA, &gptq_gemm_rdna3_wmma);

  rocm_ops.def(
      "moe_gptq_gemm_rdna3(Tensor a, Tensor! c, Tensor b_q_weight, "
      "Tensor b_scales, Tensor b_qzeros, Tensor topk_weights, "
      "Tensor sorted_token_ids, Tensor expert_ids, "
      "Tensor num_tokens_post_padded, "
      "int top_k, int block_size_m, bool mul_topk_weight, "
      "int output_topk) -> ()");
  rocm_ops.impl("moe_gptq_gemm_rdna3", torch::kCUDA, &moe_gptq_gemm_rdna3);
#endif

  // Custom attention op
  // Compute the attention between an input query and the cached
  // keys/values using PagedAttention.
  rocm_ops.def(
      "paged_attention(Tensor! out, Tensor exp_sums,"
      "                Tensor max_logits, Tensor tmp_out,"
      "                Tensor query, Tensor key_cache,"
      "                Tensor value_cache, int num_kv_heads,"
      "                float scale, Tensor block_tables,"
      "                Tensor seq_lens,"
      "                Tensor? query_start_loc,"
      "                int block_size,"
      "                int max_seq_len,"
      "                Tensor? alibi_slopes,"
      "                str kv_cache_dtype,"
      "                Tensor k_scale, Tensor v_scale,"
      "                Tensor? fp8_out_scale,"
      "                str mfma_type,"
      "                bool use_interleaved_v_cache=False) -> ()");
  rocm_ops.impl("paged_attention", torch::kCUDA, &paged_attention);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)

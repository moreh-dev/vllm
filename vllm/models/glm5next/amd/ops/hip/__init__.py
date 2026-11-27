# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm/aiter's gfx950 DSA kernels for GLM-5.3-Flash, built in-tree.

The kernels (`../third_party/aiter`, see its README) are built against the
installed AITER on first use. Each is opt-in by an env var: its gate is False
off gfx950, and builds the kernel otherwise. A failed build raises, since a
silent fallback would make a benchmark of the kernel meaningless. Callers keep
their default path for a call a kernel cannot take (`decline`).
"""

import functools

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.models.glm5next.amd.ops.hip._build import load_module
from vllm.platforms import current_platform

logger = init_logger(__name__)

# The shapes the kernels are built for: the GLM-5.x indexer (32 heads of 128)
# and its 64-head MLA over a 512-wide latent without RoPE.
INDEXER_HEADS = 32
INDEXER_HEAD_DIM = 128
FUSED_TOPK_MAX_K = 512
PAGED_LOGITS_BLOCK_SIZES = (1, 64)
SPARSE_MLA_HEADS = 64
SPARSE_MLA_HEAD_DIM = 512
SPARSE_MLA_MAX_QUERIES = 16384
SPARSE_MLA_MAX_TOPK = 2176


@functools.cache
def on_gfx950() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import _ON_GFX950

    return _ON_GFX950


def _gate(env_name: str, module: str, what: str) -> bool:
    if not getattr(envs, env_name):
        return False
    if not on_gfx950():
        logger.warning_once(
            "%s=1 needs gfx950; %s stays on its default path", env_name, what
        )
        return False
    load_module(module)
    logger.info_once("%s: %s runs on the in-tree HIP kernel", env_name, what)
    return True


def decline(env_name: str, reason: str) -> None:
    """Log once that a call keeps its default path, and why."""
    logger.warning_once("%s: %s, so the default path runs", env_name, reason)


@functools.cache
def sparse_mla_fp8_enabled() -> bool:
    return _gate("VLLM_ROCM_DSA_HIP_SPARSE_MLA_FP8", "sparse_mla_fp8", "sparse MLA")


@functools.cache
def kpool_compress_enabled() -> bool:
    return _gate(
        "VLLM_ROCM_DSA_KPOOL_HIP_COMPRESS", "kpool_compress", "the kpool cache writers"
    )


@functools.cache
def fp8_mqa_logits_enabled() -> bool:
    return _gate(
        "VLLM_ROCM_DSA_HIP_FP8_MQA_LOGITS",
        "fp8_mqa_logits",
        "the prefill indexer logits",
    )


@functools.cache
def fp8_mqa_topk_enabled() -> bool:
    return _gate(
        "VLLM_ROCM_DSA_HIP_FP8_MQA_TOPK",
        "fp8_mqa_topk",
        "the prefill indexer logits + top-k",
    )


@functools.cache
def fp8_paged_mqa_logits_enabled() -> bool:
    return _gate(
        "VLLM_ROCM_DSA_HIP_FP8_PAGED_MQA_LOGITS",
        "fp8_paged_mqa_logits",
        "the decode indexer logits",
    )


@functools.cache
def _cu_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def kpool_compress_insert(
    kv_cache: torch.Tensor,
    k: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    slot_mapping: torch.Tensor,
    pool_size: int,
    round_scale: bool = True,
) -> None:
    """Prefill kpool write, byte-identical to `kpool_compress_and_write_cache`.

    Args:
        kv_cache: indexer K cache `[num_blocks, block_size, 132]` uint8.
        k: `[n_tokens, 128]` bf16, rows may be strided.
        gate_score: `[n_tokens, 128]` bf16.
        ape: `[pool_size, 128]` fp32.
        slot_mapping: `[n_tokens]` int32 or int64, the pool slot on the token
            that closes a pool and -1 elsewhere.
        pool_size: 2, 3, 4, 8 or 16.
        round_scale: Round each fp8 scale down to a power of two.

    """
    load_module("kpool_compress").kpool_compress_insert(
        kv_cache, k, gate_score, ape, slot_mapping, pool_size, round_scale, 0
    )


def kpool_decode_update(
    kv_cache: torch.Tensor,
    tail_kv_cache: torch.Tensor,
    tail_slot_mapping: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    pool_size: int,
    round_scale: bool = True,
) -> None:
    """Decode kpool update, byte-identical to
    `kpool_decode_update_and_maybe_write_cache_batched` (same arguments,
    `next_n <= 16`, the tail ring a power-of-two multiple of `pool_size`)."""
    load_module("kpool_compress").kpool_decode_update(
        kv_cache,
        tail_kv_cache,
        tail_slot_mapping,
        key,
        slot_score,
        ape,
        slot_mapping,
        positions,
        pool_size,
        round_scale,
    )


def kpool_insert_and_seed_tail(
    kv_cache: torch.Tensor,
    tail_kv_cache: torch.Tensor,
    k: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    slot_mapping: torch.Tensor,
    tail_slot_mapping: torch.Tensor,
    pool_size: int,
    round_scale: bool = True,
) -> None:
    """`kpool_compress_insert` and `kpool_seed_tail_cache` in one launch: also
    stores each token's K and gate score in the tail ring at
    `tail_slot_mapping` (-1 skips the token)."""
    load_module("kpool_compress").kpool_mixed_write(
        kv_cache,
        ape,
        pool_size,
        round_scale,
        0,
        k,
        gate_score,
        slot_mapping,
        tail_slot_mapping,
        tail_kv_cache,
        None,
        None,
        None,
        None,
        None,
    )


def fp8_mqa_logits(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    kv_scale: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Prefill indexer logits over gathered K, same contract as AITER's Triton
    `fp8_mqa_logits`: `q_fp8` `[M, 32, 128]`, `k_fp8` `[N, 128]`, `kv_scale`
    `[N]` fp32, `weights` `[M, 32]` fp32, row `m` valid on
    `[cu_seqlen_ks[m], cu_seqlen_ke[m])`. Returns `[M, N]` fp32, -inf outside
    each row's window when `clean_logits`, untouched there otherwise."""
    out = torch.empty(
        (q_fp8.shape[0], k_fp8.shape[0]), dtype=torch.float32, device=q_fp8.device
    )
    load_module("fp8_mqa_logits").fp8_mqa_logits(
        out,
        q_fp8.contiguous(),
        k_fp8.contiguous(),
        kv_scale.contiguous(),
        weights.contiguous(),
        cu_seqlen_ks.contiguous(),
        cu_seqlen_ke.contiguous(),
        _cu_count(q_fp8.device.index),
        clean_logits,
    )
    return out


@functools.lru_cache(maxsize=1024)
def _fp8_mqa_topk_workspace_size(m: int, n: int, cu_count: int) -> int:
    return load_module("fp8_mqa_topk").fp8_mqa_topk_workspace_size(m, n, cu_count)


def fp8_mqa_topk(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    kv_scale: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    """`fp8_mqa_logits` and a per-row top-k, without writing the logits.

    Writes `topk_indices` `[M, K]` int32 (`K <= 512`): row `m` gets the
    indices, relative to `max(cu_seqlen_ks[m], 0)`, of its `K` largest logits
    in no particular order, or `0..len-1` then -1 when the row is shorter
    than `K`. Not CUDA-graph capturable (host-sized grid and workspace).
    """
    cu_count = _cu_count(q_fp8.device.index)
    workspace = torch.empty(
        _fp8_mqa_topk_workspace_size(q_fp8.shape[0], k_fp8.shape[0], cu_count),
        dtype=torch.uint8,
        device=q_fp8.device,
    )
    load_module("fp8_mqa_topk").fp8_mqa_topk(
        q_fp8.contiguous(),
        k_fp8.contiguous(),
        kv_scale.contiguous(),
        weights.contiguous(),
        cu_seqlen_ks.contiguous(),
        cu_seqlen_ke.contiguous(),
        topk_indices,
        workspace,
        cu_count,
        None,
        False,
    )
    return topk_indices


def fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Decode indexer logits over the paged cache, a drop-in for AITER's
    `deepgemm_fp8_paged_mqa_logits`.

    Args:
        q_fp8: `[batch, next_n, 32, 128]` fp8.
        kv_cache_fp8: `[num_blocks, block_size, 1, 132]` uint8, block size 1
            (flat rows) or 64 (16x16 preshuffled keys).
        weights: `[batch * next_n, 32]` fp32.
        context_lens: `[batch]` or `[batch, next_n]` int32.
        block_tables: `[batch, max_blocks]` int32.
        max_model_len: Logits row width.
        out: `[batch * next_n, max_model_len]` fp32; only each row's causal
            window is written.

    """
    # The kernel addresses q, weights and the block table as dense rows
    # without checking their strides.
    load_module("fp8_paged_mqa_logits").fp8_paged_mqa_logits(
        out,
        q_fp8.contiguous(),
        kv_cache_fp8,
        weights.contiguous(),
        context_lens.contiguous(),
        block_tables.contiguous(),
        max_model_len,
        _cu_count(q_fp8.device.index),
    )
    return out


def _sparse_mla_split_count(queries: int) -> int:
    for limit, splits in (
        (6, 17),
        (12, 9),
        (21, 6),
        (23, 4),
        (42, 6),
        (64, 4),
        (128, 2),
    ):
        if queries <= limit:
            return splits
    return 1


def sparse_mla_fp8_fwd(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_indices: torch.Tensor,
    softmax_scale: float,
    q_scale: float = 1.0,
    kv_scale: float = 1.0,
    out: torch.Tensor | None = None,
    return_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sparse MLA over an FP8 latent cache, AITER's `sparse_mla_fp8_fwd`.

    Args:
        q: `[Q, 64, 512]` bf16 (encoded to fp8 as `q / q_scale` in the kernel)
            or fp8, `Q <= 16384`, rows 16-byte aligned with unit inner stride.
        kv_buffer: contiguous fp8 `[slots, 512]`.
        kv_indices: contiguous int32 `[Q, K]`, `K <= 2176`: physical slots,
            negative or out-of-range ones masked.
        softmax_scale: Scale of the QK scores.
        q_scale: Decoding scale of the fp8 query.
        kv_scale: Decoding scale of the cache.
        out: `[Q, 64, 512]` bf16, allocated when None.
        return_lse: Also return the natural-log LSE `[Q, 64]` fp32.

    """
    queries = q.shape[0]
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    lse = (
        torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    splits = _sparse_mla_split_count(queries)
    rows = queries * splits * SPARSE_MLA_HEADS if splits > 1 else 0
    partial_o = torch.empty(
        (rows * SPARSE_MLA_HEAD_DIM,), dtype=torch.bfloat16, device=q.device
    )
    partial_lse = torch.empty((rows,), dtype=torch.float32, device=q.device)
    load_module("sparse_mla_fp8").sparse_mla_fp8_fwd(
        q,
        kv_buffer,
        kv_indices,
        out,
        lse,
        partial_o,
        partial_lse,
        softmax_scale,
        q_scale,
        kv_scale,
    )
    return out, lse

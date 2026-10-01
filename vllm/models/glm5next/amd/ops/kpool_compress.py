# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""kpool (key-pooling) Triton kernels for the sparse-attention indexer.

The cache stores POOLS (1 entry per ``pool_size`` consecutive tokens) rather
than individual tokens. ``compress_ratio == pool_size`` on the kv_cache_spec
makes the metadata builder emit pool-granular slot_mapping / seq_lens /
cu_seq_lens / page_table for free; this file supplies the compress-write
kernel (replacing ``indexer_k_quant_and_cache``) and the pool-level topk
helpers (select pools -> expand to tokens -> append tail).
"""

from __future__ import annotations

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

# The GLM-5.3-Flash indexer head dimension is fixed at 128.
INDEX_HEAD_DIM = 128
FP8_DTYPE = current_platform.fp8_dtype()
FP8_MAX = torch.finfo(FP8_DTYPE).max


@triton.jit
def _cache_k_offset(
    token_offset,
    dim_offset,
    head_dim: tl.constexpr,
    preshuffle: tl.constexpr,
):
    if preshuffle:
        return (
            (token_offset // 16) * 16 * head_dim
            + (dim_offset // 16) * 16 * 16
            + (token_offset % 16) * 16
            + dim_offset % 16
        )
    return token_offset * head_dim + dim_offset


# Hadamard-128 rotation


@triton.jit
def _hadamard128_stage(x, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    x3 = tl.reshape(x, (GROUPS, 2, STRIDE))
    x3 = tl.trans(x3, 0, 2, 1)
    a, b = tl.split(x3)
    x3 = tl.join(a + b, a - b)
    x3 = tl.trans(x3, 0, 2, 1)
    return tl.reshape(x3, (128,))


@triton.jit
def _hadamard128(x):
    x = _hadamard128_stage(x, 64, 1)
    x = _hadamard128_stage(x, 32, 2)
    x = _hadamard128_stage(x, 16, 4)
    x = _hadamard128_stage(x, 8, 8)
    x = _hadamard128_stage(x, 4, 16)
    x = _hadamard128_stage(x, 2, 32)
    x = _hadamard128_stage(x, 1, 64)
    return x * 0.08838834764831845  # 1/sqrt(128)


# Fused pool compression and cache write.


@triton.jit
def _kpool_softmax_rotate_write_cache_kernel(
    buf_fp8_ptr,
    buf_fp32_ptr,
    slot_k_ptr,
    slot_score_ptr,
    ape_ptr,
    loc_ptr,
    write_mask_ptr,
    compressed_k_ptr,
    compressed_scale_ptr,
    slot_k_stride_0,
    slot_k_stride_1,
    slot_score_stride_0,
    slot_score_stride_1,
    ape_stride_0,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    PRESHUFFLE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    HAS_WRITE_MASK: tl.constexpr,
    RETURN_COMPRESSED: tl.constexpr,
    WRITE_CACHE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per pool. softmax(slot_score+ape)-weighted sum of slot_k ->
    Hadamard-128 -> per-vector fp8 absmax quant -> write to cache at ``loc``."""
    row = tl.program_id(0)
    do_write = True
    if HAS_WRITE_MASK:
        do_write = tl.load(write_mask_ptr + row)

    offs = tl.arange(0, BLOCK_D)
    mask = (offs < HEAD_DIM) & do_write

    # --- Pass 1: per-dim max over the pool (softmax numerical stability) ---
    max_score = tl.full((BLOCK_D,), -float("inf"), tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        score = tl.load(
            slot_score_ptr
            + row * slot_score_stride_0
            + slot * slot_score_stride_1
            + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        score += tl.load(ape_ptr + slot * ape_stride_0 + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        max_score = tl.maximum(max_score, score)

    # --- Pass 2: softmax-weighted sum of K ---
    acc = tl.full((BLOCK_D,), 0.0, tl.float32)
    denom = tl.full((BLOCK_D,), 0.0, tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        score = tl.load(
            slot_score_ptr
            + row * slot_score_stride_0
            + slot * slot_score_stride_1
            + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        score += tl.load(ape_ptr + slot * ape_stride_0 + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        prob = tl.exp(score - max_score)
        denom += prob
        k = tl.load(
            slot_k_ptr + row * slot_k_stride_0 + slot * slot_k_stride_1 + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += k * prob

    x = acc / denom
    x = tl.where(do_write, x, 0.0).to(tl.bfloat16).to(tl.float32)

    # Match the unfused pooled-K path's bf16 precision before quantization.
    x = _hadamard128(x).to(tl.bfloat16).to(tl.float32)

    # --- per-vector absmax fp8 quant ---
    fp8_max_inv = 1.0 / FP8_MAX
    absmax = tl.max(tl.abs(x), axis=0)
    absmax = tl.maximum(absmax, 1e-4)
    if ROUND_SCALE:
        scale = tl.exp2(tl.ceil(tl.log2(absmax * fp8_max_inv)))
    else:
        scale = absmax * fp8_max_inv
    quantized = x / scale
    quantized = tl.minimum(tl.maximum(quantized, -FP8_MAX), FP8_MAX)

    if WRITE_CACHE:
        loc = tl.load(loc_ptr + row, mask=do_write, other=0)
        loc_page_index = loc // PAGE_SIZE
        loc_token_offset_in_page = loc % PAGE_SIZE
        out_k_offsets = loc_page_index * BUF_NUMEL_PER_PAGE + _cache_k_offset(
            loc_token_offset_in_page,
            offs,
            HEAD_DIM,
            PRESHUFFLE,
        )
        out_s_offset = (
            loc_page_index * BUF_NUMEL_PER_PAGE // 4
            + S_OFFSET_NBYTES_IN_PAGE // 4
            + loc_token_offset_in_page
        )
        tl.store(buf_fp8_ptr + out_k_offsets, quantized, mask=mask)
        tl.store(buf_fp32_ptr + out_s_offset, scale, mask=do_write)

    if RETURN_COMPRESSED:
        tl.store(
            compressed_k_ptr + row * HEAD_DIM + offs,
            quantized,
            mask=offs < HEAD_DIM,
        )
        tl.store(compressed_scale_ptr + row, scale)


def kpool_compress_and_write_cache(
    kv_cache: torch.Tensor,
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    pool_size: int,
    head_dim: int = INDEX_HEAD_DIM,
    write_mask: torch.Tensor | None = None,
    round_scale: bool = True,
    return_compressed: bool = False,
    write_cache: bool = True,
):
    """Compress ``pool_size`` tokens into one fp8 K and write at ``loc``.

    Args:
        kv_cache: indexer K cache ``[num_blocks, block_size, head_dim+4]`` uint8.
        slot_k: ``[n_pools, pool_size, head_dim]`` bf16 — raw per-token K.
        slot_score: ``[n_pools, pool_size, head_dim]`` — per-token gate score.
        ape: ``[pool_size, head_dim]`` fp32 — per-slot position bias.
        loc: ``[n_pools]`` int64 — flat physical slot per pool.
        pool_size: Number of tokens compressed into one cache entry.
        head_dim: Indexer head dimension.
        write_mask: ``[n_pools]`` bool — pools to write, or None for all.
        round_scale: Round each fp8 scale down to a power of two.
        return_compressed: Also return the compressed K and scales.
        write_cache: Write the compressed result into ``kv_cache``.

    """
    assert slot_k.ndim == 3
    assert slot_score.shape == slot_k.shape
    assert ape.shape == slot_k.shape[1:]
    assert slot_k.shape[2] == head_dim
    assert slot_k.dtype == torch.bfloat16
    assert ape.dtype == torch.float32
    assert kv_cache.dtype == torch.uint8
    assert loc.dtype == torch.int64
    assert write_cache or return_compressed

    page_size = kv_cache.shape[1]
    buf = kv_cache
    slot_k = slot_k.contiguous()
    slot_score = slot_score.contiguous()
    ape = ape.contiguous()
    loc = loc.contiguous()
    if write_mask is None:
        write_mask = torch.empty((1,), dtype=torch.bool, device=slot_k.device)
        has_write_mask = False
    else:
        assert write_mask.shape == (slot_k.shape[0],)
        write_mask = write_mask.contiguous()
        has_write_mask = True
        assert not return_compressed

    if slot_k.shape[0] == 0:
        if return_compressed:
            return (
                torch.empty(
                    (0, head_dim),
                    dtype=FP8_DTYPE,
                    device=slot_k.device,
                ),
                torch.empty((0,), dtype=torch.float32, device=slot_k.device),
            )
        return None

    buf_fp8 = buf.view(FP8_DTYPE)
    buf_fp32 = buf.view(torch.float32)
    # bytes per page (last dim of kv_cache) viewed as uint8
    buf_numel_per_page = buf.stride(0)
    s_offset_nbytes_in_page = page_size * head_dim

    if return_compressed:
        compressed_k = torch.empty(
            (slot_k.shape[0], head_dim),
            dtype=FP8_DTYPE,
            device=slot_k.device,
        )
        compressed_scale = torch.empty(
            (slot_k.shape[0],), dtype=torch.float32, device=slot_k.device
        )
    else:
        compressed_k = buf_fp8
        compressed_scale = buf_fp32

    if page_size > 1:
        assert page_size % 16 == 0, "ROCm preshuffle requires 16-token tiles"

    _kpool_softmax_rotate_write_cache_kernel[(slot_k.shape[0],)](
        buf_fp8,
        buf_fp32,
        slot_k,
        slot_score,
        ape,
        loc,
        write_mask,
        compressed_k,
        compressed_scale,
        slot_k.stride(0),
        slot_k.stride(1),
        slot_score.stride(0),
        slot_score.stride(1),
        ape.stride(0),
        PAGE_SIZE=page_size,
        BUF_NUMEL_PER_PAGE=buf_numel_per_page,
        POOL_SIZE=slot_k.shape[1],
        HEAD_DIM=head_dim,
        S_OFFSET_NBYTES_IN_PAGE=s_offset_nbytes_in_page,
        FP8_MAX=FP8_MAX,
        PRESHUFFLE=page_size > 1,
        ROUND_SCALE=round_scale,
        HAS_WRITE_MASK=has_write_mask,
        RETURN_COMPRESSED=return_compressed,
        WRITE_CACHE=write_cache,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )

    if return_compressed:
        return compressed_k, compressed_scale
    return None


# Fused prefill insert: pool straight from the token-major K / gate score.


@triton.jit
def _fwht128_rows_stage(x, N: tl.constexpr, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
    # _hadamard128_stage on a flat [rows * 128] tensor. Every stage's pair
    # distance divides 128, so butterflies never straddle two rows and each row
    # sees exactly the ops (and fp32 rounding) of the single-row version.
    x3 = tl.reshape(x, (GROUPS, 2, STRIDE))
    x3 = tl.trans(x3, 0, 2, 1)
    a, b = tl.split(x3)
    x3 = tl.join(a + b, a - b)
    x3 = tl.trans(x3, 0, 2, 1)
    return tl.reshape(x3, (N,))


@triton.jit
def _hadamard128_rows(x, ROWS: tl.constexpr):
    N: tl.constexpr = ROWS * 128
    x = tl.reshape(x, (N,))
    x = _fwht128_rows_stage(x, N, ROWS * 64, 1)
    x = _fwht128_rows_stage(x, N, ROWS * 32, 2)
    x = _fwht128_rows_stage(x, N, ROWS * 16, 4)
    x = _fwht128_rows_stage(x, N, ROWS * 8, 8)
    x = _fwht128_rows_stage(x, N, ROWS * 4, 16)
    x = _fwht128_rows_stage(x, N, ROWS * 2, 32)
    x = _fwht128_rows_stage(x, N, ROWS, 64)
    x = x * 0.08838834764831845  # 1/sqrt(128)
    return tl.reshape(x, (ROWS, 128))


@triton.jit
def _kpool_insert_windows(
    pid,
    k_ptr,
    score_ptr,
    ape_ptr,
    slot_mapping_ptr,
    buf_fp8_ptr,
    buf_fp32_ptr,
    n_tokens,
    k_stride,
    score_stride,
    ape_stride,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    PRESHUFFLE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    BLOCK_P: tl.constexpr,
    POOL_PAD: tl.constexpr,
):
    """One program per ``BLOCK_P`` windows of ``POOL_SIZE`` tokens.

    ``slot_mapping`` is pool-granular: a token whose slot is >= 0 closes the
    pool made of it and its ``POOL_SIZE - 1`` predecessors. Closing tokens of
    disjoint pools are >= ``POOL_SIZE`` apart, so each window holds at most
    one; a per-window reduction turns the windows into a dense
    ``[BLOCK_P, HEAD_DIM]`` tile of real pools. K / gate are read in place (no
    ``[n, POOL_SIZE, HEAD_DIM]`` copy) and the pool math runs once per pool,
    not once per token. A pool may start in the previous window.

    A denser mapping (not produced by pool-aligned batches) just makes the
    loop below run more than once, which keeps the output identical to
    ``kpool_compress_and_write_cache`` for any ``slot_mapping``.
    """
    win = pid.to(tl.int64) * BLOCK_P + tl.arange(0, BLOCK_P)
    lane = tl.arange(0, POOL_PAD)
    tok = win[:, None] * POOL_SIZE + lane[None, :]
    slot = tl.load(
        slot_mapping_ptr + tok,
        mask=(lane[None, :] < POOL_SIZE) & (tok < n_tokens),
        other=-1,
    ).to(tl.int64)
    closes = ((slot >= 0) & (tok >= POOL_SIZE - 1)).to(tl.int32)
    rank = tl.cumsum(closes, axis=1) - closes
    n_rounds = tl.max(tl.sum(closes, axis=1), axis=0)

    offs = tl.arange(0, HEAD_DIM)
    fp8_max_inv = 1.0 / FP8_MAX
    for r in tl.range(0, n_rounds):
        pick = (closes != 0) & (rank == r)
        pool_ok = tl.max(pick.to(tl.int32), axis=1) > 0
        pool_last = tl.max(tl.where(pick, tok, -1), axis=1)
        loc = tl.max(tl.where(pick, slot, -1), axis=1)
        pool_first = pool_last - (POOL_SIZE - 1)
        mask = pool_ok[:, None]

        # --- Pass 1: per-dim max over the pool (softmax numerical stability) ---
        max_score = tl.full((BLOCK_P, HEAD_DIM), -float("inf"), tl.float32)
        for s in tl.static_range(0, POOL_SIZE):
            row = (pool_first + s)[:, None]
            score = tl.load(
                score_ptr + row * score_stride + offs[None, :], mask=mask, other=0.0
            ).to(tl.float32)
            score += tl.load(ape_ptr + s * ape_stride + offs)[None, :]
            max_score = tl.maximum(max_score, score)

        # --- Pass 2: softmax-weighted sum of K (same order as the unfused op) ---
        acc = tl.zeros((BLOCK_P, HEAD_DIM), tl.float32)
        denom = tl.zeros((BLOCK_P, HEAD_DIM), tl.float32)
        for s in tl.static_range(0, POOL_SIZE):
            row = (pool_first + s)[:, None]
            score = tl.load(
                score_ptr + row * score_stride + offs[None, :], mask=mask, other=0.0
            ).to(tl.float32)
            score += tl.load(ape_ptr + s * ape_stride + offs)[None, :]
            prob = tl.exp(score - max_score)
            denom += prob
            k = tl.load(
                k_ptr + row * k_stride + offs[None, :], mask=mask, other=0.0
            ).to(tl.float32)
            acc += k * prob

        x = (acc / denom).to(tl.bfloat16).to(tl.float32)
        x = _hadamard128_rows(x, BLOCK_P).to(tl.bfloat16).to(tl.float32)

        # --- per-vector absmax fp8 quant ---
        absmax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
        if ROUND_SCALE:
            scale = tl.exp2(tl.ceil(tl.log2(absmax * fp8_max_inv)))
        else:
            scale = absmax * fp8_max_inv
        quantized = tl.minimum(tl.maximum(x / scale[:, None], -FP8_MAX), FP8_MAX)

        loc_page_index = loc // PAGE_SIZE
        loc_token_offset_in_page = loc % PAGE_SIZE
        out_k_offsets = loc_page_index[:, None] * BUF_NUMEL_PER_PAGE + _cache_k_offset(
            loc_token_offset_in_page[:, None],
            offs[None, :],
            HEAD_DIM,
            PRESHUFFLE,
        )
        out_s_offset = (
            loc_page_index * BUF_NUMEL_PER_PAGE // 4
            + S_OFFSET_NBYTES_IN_PAGE // 4
            + loc_token_offset_in_page
        )
        tl.store(buf_fp8_ptr + out_k_offsets, quantized, mask=mask)
        tl.store(buf_fp32_ptr + out_s_offset, scale, mask=pool_ok)


@triton.jit(do_not_specialize=["n_tokens"])
def _kpool_compress_insert_kernel(
    k_ptr,
    score_ptr,
    ape_ptr,
    slot_mapping_ptr,
    buf_fp8_ptr,
    buf_fp32_ptr,
    n_tokens,
    k_stride,
    score_stride,
    ape_stride,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    PRESHUFFLE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    BLOCK_P: tl.constexpr,
    POOL_PAD: tl.constexpr,
):
    """See ``_kpool_insert_windows``."""
    _kpool_insert_windows(
        tl.program_id(0),
        k_ptr,
        score_ptr,
        ape_ptr,
        slot_mapping_ptr,
        buf_fp8_ptr,
        buf_fp32_ptr,
        n_tokens,
        k_stride,
        score_stride,
        ape_stride,
        PAGE_SIZE,
        BUF_NUMEL_PER_PAGE,
        POOL_SIZE,
        HEAD_DIM,
        S_OFFSET_NBYTES_IN_PAGE,
        FP8_MAX,
        PRESHUFFLE,
        ROUND_SCALE,
        BLOCK_P,
        POOL_PAD,
    )


def _kpool_insert_launch_config(n_tokens: int, pool_size: int) -> tuple[int, int]:
    """(pools per program, num_warps) for ``kpool_compress_insert``.

    Tuned on MI355X (GLM-5.3-Flash, pool_size 4): small batches want many
    tiny programs to fill the CUs, large ones want 16 pools per wave for
    memory-level parallelism. One wave per program throughout; wider
    programs were slower at every size.
    """
    n_pools = n_tokens // pool_size
    if n_pools <= 2048:
        return 1, 1
    if n_pools <= 8192:
        return 2, 1
    if n_pools <= 32768:
        return 8, 1
    return 16, 1


def kpool_compress_insert(
    kv_cache: torch.Tensor,
    k: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    slot_mapping: torch.Tensor,
    pool_size: int,
    head_dim: int = INDEX_HEAD_DIM,
    round_scale: bool = True,
    block_p: int | None = None,
    num_warps: int | None = None,
) -> None:
    """Pool a prefill batch straight into the indexer K cache.

    Drop-in for the gather + ``kpool_compress_and_write_cache`` sequence of the
    prefill path: every token whose pool-granular ``slot_mapping`` entry is
    >= 0 (and that has ``pool_size - 1`` predecessors in the batch) closes a
    pool; that pool is softmax(gate+ape)-weighted, Hadamard-rotated,
    fp8-quantized and written at its slot. Results are bit-identical to the
    unfused path.

    Args:
        kv_cache: indexer K cache ``[num_blocks, block_size, head_dim+4]`` uint8.
        k: ``[n_tokens, head_dim]`` bf16 raw per-token K (row stride free).
        gate_score: ``[n_tokens, head_dim]`` bf16 per-token gate score.
        ape: ``[pool_size, head_dim]`` fp32 per-slot position bias.
        slot_mapping: ``[n_tokens]`` int32/int64 pool-granular slots.
        pool_size: Number of tokens compressed into one cache entry.
        head_dim: Indexer head dimension (the Hadamard rotation needs 128).
        round_scale: Round each fp8 scale up to a power of two.
        block_p: Pools per program override (benchmarking only).
        num_warps: Warps per program override (benchmarking only). Triton's
            codegen is not bit-stable across every layout (``block_p=1,
            num_warps=2`` flips rare fp8 ulps, as
            ``kpool_compress_and_write_cache`` itself does at
            ``num_warps=2``); the defaults are verified bit-exact.

    """
    assert head_dim == 128, "Hadamard-128 rotation"
    assert k.ndim == 2 and k.shape[1] == head_dim
    assert gate_score.shape == k.shape
    assert k.dtype == torch.bfloat16
    assert k.stride(1) == 1 and gate_score.stride(1) == 1
    assert ape.shape == (pool_size, head_dim)
    assert ape.dtype == torch.float32
    assert kv_cache.dtype == torch.uint8
    assert slot_mapping.shape == (k.shape[0],)
    assert slot_mapping.dtype in (torch.int32, torch.int64)

    n_tokens = k.shape[0]
    if n_tokens < pool_size:
        return
    page_size = kv_cache.shape[1]
    if page_size > 1:
        assert page_size % 16 == 0, "ROCm preshuffle requires 16-token tiles"
    ape = ape.contiguous()
    slot_mapping = slot_mapping.contiguous()

    default_p, default_warps = _kpool_insert_launch_config(n_tokens, pool_size)
    block_p = block_p or default_p
    num_warps = num_warps or default_warps

    grid = (triton.cdiv(triton.cdiv(n_tokens, pool_size), block_p),)
    _kpool_compress_insert_kernel[grid](
        k,
        gate_score,
        ape,
        slot_mapping,
        kv_cache.view(FP8_DTYPE),
        kv_cache.view(torch.float32),
        n_tokens,
        k.stride(0),
        gate_score.stride(0),
        ape.stride(0),
        PAGE_SIZE=page_size,
        BUF_NUMEL_PER_PAGE=kv_cache.stride(0),
        POOL_SIZE=pool_size,
        HEAD_DIM=head_dim,
        S_OFFSET_NBYTES_IN_PAGE=page_size * head_dim,
        FP8_MAX=FP8_MAX,
        PRESHUFFLE=page_size > 1,
        ROUND_SCALE=round_scale,
        BLOCK_P=block_p,
        POOL_PAD=triton.next_power_of_2(pool_size),
        num_warps=num_warps,
    )


# Seed each request's incomplete pool into its paged tail during prefill.


@triton.jit
def _kpool_tail_seed_kernel(
    key_ptr,
    score_ptr,
    tslot_ptr,
    tail_ptr,
    n_tokens,
    TAIL_BLOCK_ELEMS: tl.constexpr,
    KPOOL_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KPOOL: tl.constexpr,
    RING: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Copy token ``i``'s raw K + gate into its request's tail ring.

    Token ``i`` is among its request's last KPOOL tokens iff the token KPOOL
    ahead belongs to a different tail block (or is past the batch / padding,
    slot < 0). ``tslot = block * RING + pos % RING``; the destination is
    ``tail[block, {0:K, 1:score}, pos % RING, :]``.
    """
    i = tl.program_id(0)
    t = tl.load(tslot_ptr + i).to(tl.int64)
    if t < 0:
        return
    blk = t // RING  # t >= 0 here, so trunc == floor
    ahead = tl.load(tslot_ptr + i + KPOOL, mask=i + KPOOL < n_tokens, other=-1).to(
        tl.int64
    )
    # Match the torch semantics exactly: a negative ahead slot floors to a
    # block id that differs from every real block -> token is in the tail.
    # Only divide non-negative slots (Triton int div truncates, torch floors).
    if ahead >= 0 and ahead // RING == blk:
        return
    offs = tl.arange(0, BLOCK_D)
    m = offs < HEAD_DIM
    block_base = blk * TAIL_BLOCK_ELEMS
    base = block_base + (t % RING) * HEAD_DIM
    k = tl.load(key_ptr + i * HEAD_DIM + offs, mask=m)
    s = tl.load(score_ptr + i * HEAD_DIM + offs, mask=m)
    tl.store(tail_ptr + base + offs, k, mask=m)
    tl.store(
        tail_ptr + block_base + KPOOL_HEAD + (t % RING) * HEAD_DIM + offs, s, mask=m
    )


def kpool_seed_tail_cache(
    tail_kv_cache: torch.Tensor,
    key: torch.Tensor,
    gate_score: torch.Tensor,
    tslot: torch.Tensor,
    kpool: int,
    head_dim: int = INDEX_HEAD_DIM,
) -> None:
    """Seed the paged tail cache from a prefill batch (see the kernel)."""
    assert tail_kv_cache.dtype == torch.bfloat16
    assert key.dtype == torch.bfloat16
    n = tslot.shape[0]
    if n == 0:
        return
    _kpool_tail_seed_kernel[(n,)](
        key,
        gate_score,
        tslot,
        tail_kv_cache,
        n,
        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),
        KPOOL_HEAD=tail_kv_cache.stride(1),
        HEAD_DIM=head_dim,
        KPOOL=kpool,
        RING=tail_kv_cache.shape[2],
        BLOCK_D=triton.next_power_of_2(head_dim),
    )


# Update each request's tail during decode and write completed pools.


@triton.jit
def _kpool_decode_update_batched_kernel(
    buf_fp8_ptr,
    buf_fp32_ptr,
    tail_kv_ptr,
    tail_slot_mapping_ptr,  # [B, NEXT_N] int32
    key_ptr,  # [B, NEXT_N, HEAD_DIM] bf16
    key_stride_b,
    key_stride_t,
    slot_score_ptr,  # [B, NEXT_N, HEAD_DIM] bf16
    ss_stride_b,
    ss_stride_t,
    ape_ptr,
    ape_stride_0,
    slot_mapping_ptr,  # [B, NEXT_N] int32
    positions_ptr,  # [B, NEXT_N] int32
    NEXT_N,  # runtime token count per request (no .item() needed)
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    RING: tl.constexpr,
    TAIL_BLOCK_ELEMS: tl.constexpr,
    KPOOL_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    PRESHUFFLE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program per request; iterates its NEXT_N verify tokens in order.

    Replaces the caller's per-token sequential launch loop. The intra-request
    iteration MUST stay in position order: a pool-completion at token t* reads
    the tail-ring slots that tokens t < t* (same request) just stashed in this
    same invocation. ``tl.range`` iterates sequentially within the program, so
    those stashes are visible to the later completion read. Cross-request
    programs are independent (distinct tail blocks). RING >= POOL_SIZE.
    """
    req = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    dim_mask = offs < HEAD_DIM

    for t in tl.range(0, NEXT_N):
        idx = req * NEXT_N + t
        cache_loc = tl.load(slot_mapping_ptr + idx)
        pos = tl.load(positions_ptr + idx)
        safe_pos = tl.maximum(pos, 0)
        pos_valid = (cache_loc >= 0) & (pos >= 0)

        slot = safe_pos % POOL_SIZE
        phys_slot = safe_pos % RING

        # Derive the tail block from THIS token's tail_slot (the request's block
        # is constant across a pool, but a padded / invalid entry carries a
        # negative sentinel -- reading it from token 0 would poison every
        # token's base address). Clamp so an invalid entry can never form an
        # out-of-bounds base; the accesses below are gated on pos_valid anyway.
        tail_slot = tl.load(tail_slot_mapping_ptr + idx)
        block = tl.maximum(tail_slot, 0).to(tl.int64) // RING
        block_base = block * TAIL_BLOCK_ELEMS

        # The tail-ring stash must run for EVERY real token, so it is gated on
        # the token-granular tail slot -- not on `pos_valid`, which keys off the
        # POOL-granular `slot_mapping` and is therefore only true on the pool's
        # last token. Gating the stash on pos_valid dropped every intra-pool
        # token, so a decode-built pool compressed 3 stale ring entries (the
        # prefill-seeded prompt tail, frozen forever) plus the current token.
        stash_valid = (pos >= 0) & (tail_slot >= 0)

        key = tl.load(
            key_ptr + req * key_stride_b + t * key_stride_t + offs,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        score_current = tl.load(
            slot_score_ptr + req * ss_stride_b + t * ss_stride_t + offs,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)

        if pos_valid & (slot == POOL_SIZE - 1):
            pool_logical_start = safe_pos - slot

            max_score = tl.full((BLOCK_D,), -float("inf"), tl.float32)
            for pool_slot in tl.static_range(0, POOL_SIZE):
                is_current = pool_slot == slot
                phys = (pool_logical_start + pool_slot) % RING
                score_buf = tl.load(
                    tail_kv_ptr + block_base + KPOOL_HEAD + phys * HEAD_DIM + offs,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                score = tl.where(is_current, score_current, score_buf)
                score += tl.load(
                    ape_ptr + pool_slot * ape_stride_0 + offs,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                max_score = tl.maximum(max_score, score)

            acc = tl.full((BLOCK_D,), 0.0, tl.float32)
            denom = tl.full((BLOCK_D,), 0.0, tl.float32)
            for pool_slot in tl.static_range(0, POOL_SIZE):
                is_current = pool_slot == slot
                phys = (pool_logical_start + pool_slot) % RING
                score_buf = tl.load(
                    tail_kv_ptr + block_base + KPOOL_HEAD + phys * HEAD_DIM + offs,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                score = tl.where(is_current, score_current, score_buf)
                score += tl.load(
                    ape_ptr + pool_slot * ape_stride_0 + offs,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                prob = tl.exp(score - max_score)
                denom += prob
                k_buf = tl.load(
                    tail_kv_ptr + block_base + phys * HEAD_DIM + offs,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                k = tl.where(is_current, key, k_buf)
                acc += k * prob

            x = (acc / denom).to(tl.bfloat16).to(tl.float32)
            x = _hadamard128(x).to(tl.bfloat16).to(tl.float32)

            fp8_max_inv = 1.0 / FP8_MAX
            absmax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-4)
            if ROUND_SCALE:
                scale = tl.exp2(tl.ceil(tl.log2(absmax * fp8_max_inv)))
            else:
                scale = absmax * fp8_max_inv
            quantized = tl.minimum(tl.maximum(x / scale, -FP8_MAX), FP8_MAX)

            loc = cache_loc.to(tl.int64)
            loc_page_index = loc // PAGE_SIZE
            loc_token_offset_in_page = loc % PAGE_SIZE
            out_k_offsets = loc_page_index * BUF_NUMEL_PER_PAGE + _cache_k_offset(
                loc_token_offset_in_page,
                offs,
                HEAD_DIM,
                PRESHUFFLE,
            )
            out_s_offset = (
                loc_page_index * BUF_NUMEL_PER_PAGE // 4
                + S_OFFSET_NBYTES_IN_PAGE // 4
                + loc_token_offset_in_page
            )
            tl.store(buf_fp8_ptr + out_k_offsets, quantized, mask=dim_mask)
            tl.store(buf_fp32_ptr + out_s_offset, scale)

        # Stash the current token AFTER any completion read so the completion
        # uses prior stashes (and the current token's own key/score via
        # is_current), then leaves this token for future pools. Order matches
        # the per-token kernel: completion read first, stash second.
        update_mask = dim_mask & stash_valid
        tl.store(
            tail_kv_ptr + block_base + phys_slot * HEAD_DIM + offs,
            key,
            mask=update_mask,
        )
        tl.store(
            tail_kv_ptr + block_base + KPOOL_HEAD + phys_slot * HEAD_DIM + offs,
            score_current,
            mask=update_mask,
        )


def kpool_decode_update_and_maybe_write_cache_batched(
    kv_cache: torch.Tensor,
    tail_kv_cache: torch.Tensor,
    tail_slot_mapping: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    pool_size: int,
    head_dim: int = INDEX_HEAD_DIM,
    round_scale: bool = True,
) -> None:
    """Batched decode-step kpool update for spec verify (``next_n > 1``).

    One launch replaces the caller's per-token loop. Inputs are grouped per
    request: ``[num_requests, next_n, ...]``. Each program handles one
    request's ``next_n`` tokens in position order (see the kernel docstring for
    why ordering is required for pool-completion correctness).

    Plain decode (``next_n == 1``) is handled here too — the kernel collapses
    to a single-iteration loop.

    Args:
        kv_cache: indexer K cache ``[num_blocks, block_size, head_dim+4]`` uint8.
        tail_kv_cache: paged tail cache ``[num_blocks, 2, pool_size, head_dim]``
            bf16 (K at half 0, gate score at half 1).
        tail_slot_mapping: ``[num_requests, next_n]`` int32.
        key: ``[num_requests, next_n, head_dim]`` bf16.
        slot_score: ``[num_requests, next_n, head_dim]`` bf16.
        ape: ``[pool_size, head_dim]`` fp32.
        slot_mapping: ``[num_requests, next_n]`` int32.
        positions: ``[num_requests, next_n]`` int32.
        pool_size: Number of tokens compressed into one cache entry.
        head_dim: Indexer head dimension.
        round_scale: Round each fp8 scale down to a power of two.

    """
    num_requests, next_n = key.shape[0], key.shape[1]
    if num_requests == 0 or next_n == 0:
        return
    assert tail_kv_cache.ndim == 4
    assert tail_kv_cache.shape[1] == 2
    ring = tail_kv_cache.shape[2]
    assert ring >= pool_size and ring % pool_size == 0, (ring, pool_size)
    assert tail_kv_cache.shape[3] == head_dim
    assert tail_kv_cache.dtype == torch.bfloat16
    assert key.ndim == 3 and key.shape[2] == head_dim
    assert slot_score.shape == key.shape
    assert ape.shape == (pool_size, head_dim)
    assert tail_slot_mapping.shape == (num_requests, next_n)
    assert slot_mapping.shape == (num_requests, next_n)
    assert positions.shape == (num_requests, next_n)
    assert key.dtype == torch.bfloat16
    assert slot_score.dtype == torch.bfloat16
    assert ape.dtype == torch.float32
    assert kv_cache.dtype == torch.uint8

    page_size = kv_cache.shape[1]
    buf = kv_cache
    buf_fp8 = buf.view(FP8_DTYPE)
    buf_fp32 = buf.view(torch.float32)

    # The kernel indexes the int tensors as ``req * next_n + t`` (row-major),
    # so they must be contiguous. Callers pass either a view of a contiguous
    # slice or a freshly scattered tensor, making these no-ops; the calls guard
    # against a future caller handing over a strided view.
    tail_slot_mapping = tail_slot_mapping.contiguous()
    slot_mapping = slot_mapping.contiguous()
    positions = positions.contiguous()

    if page_size > 1:
        assert page_size % 16 == 0, "ROCm preshuffle requires 16-token tiles"

    _kpool_decode_update_batched_kernel[(num_requests,)](
        buf_fp8,
        buf_fp32,
        tail_kv_cache,
        tail_slot_mapping,
        key,
        key.stride(0),
        key.stride(1),
        slot_score,
        slot_score.stride(0),
        slot_score.stride(1),
        ape,
        ape.stride(0),
        slot_mapping,
        positions,
        next_n,
        PAGE_SIZE=page_size,
        BUF_NUMEL_PER_PAGE=buf.stride(0),
        POOL_SIZE=pool_size,
        RING=ring,
        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),
        KPOOL_HEAD=tail_kv_cache.stride(1),
        HEAD_DIM=head_dim,
        S_OFFSET_NBYTES_IN_PAGE=page_size * head_dim,
        FP8_MAX=FP8_MAX,
        PRESHUFFLE=page_size > 1,
        ROUND_SCALE=round_scale,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )


# Parallel decode update: all of a request's verify tokens in one tile.


@triton.jit
def _kpool_decode_complete_one(
    c,
    active,
    t,
    offs,
    completes,
    rank,
    safe_pos,
    cache_loc,
    block,
    phys_slot,
    stash_valid,
    req,
    buf_fp8_ptr,
    buf_fp32_ptr,
    tail_kv_ptr,
    key_ptr,
    key_stride_b,
    key_stride_t,
    slot_score_ptr,
    ss_stride_b,
    ss_stride_t,
    ape_ptr,
    ape_stride_0,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    RING: tl.constexpr,
    TAIL_BLOCK_ELEMS: tl.constexpr,
    KPOOL_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    PRESHUFFLE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
):
    """Compress the pool closed by the ``c``-th completing token (if active)."""
    fp8_max_inv = 1.0 / FP8_MAX
    sel = completes & (rank == c)
    tc = tl.max(tl.where(sel, t, -1), axis=0)
    tpos = tl.max(tl.where(sel, safe_pos, -1), axis=0)
    tloc = tl.max(tl.where(sel, cache_loc, -1), axis=0).to(tl.int64)
    tblk = tl.max(tl.where(sel, block, -1), axis=0)
    tbase = tblk * TAIL_BLOCK_ELEMS
    slot = tpos % POOL_SIZE
    pool_logical_start = tpos - slot
    # Candidates t' < tc that stashed into tc's tail block. Each pool slot
    # reads token ``src`` of this launch (the current token for its own
    # slot) or, if none, the ring; the pointer is selected branch-free so
    # all slot loads are in flight together.
    earlier = (t < tc) & stash_valid & (block == tblk)
    in_k = key_ptr + req * key_stride_b + offs
    in_s = slot_score_ptr + req * ss_stride_b + offs
    ring = tail_kv_ptr + tbase + offs

    max_score = tl.full((HEAD_DIM,), -float("inf"), tl.float32)
    for pool_slot in tl.static_range(0, POOL_SIZE):
        phys = (pool_logical_start + pool_slot) % RING
        src = tl.max(tl.where(earlier & (phys_slot == phys), t, -1), axis=0)
        src = tl.where(pool_slot == slot, tc, src)
        new = src >= 0
        score = tl.where(
            new,
            tl.load(
                in_s + tl.maximum(src, 0) * ss_stride_t, mask=new & active, other=0.0
            ),
            tl.load(ring + KPOOL_HEAD + phys * HEAD_DIM, mask=~new & active, other=0.0),
        ).to(tl.float32)
        score += tl.load(ape_ptr + pool_slot * ape_stride_0 + offs)
        max_score = tl.maximum(max_score, score)

    acc = tl.zeros((HEAD_DIM,), tl.float32)
    denom = tl.zeros((HEAD_DIM,), tl.float32)
    for pool_slot in tl.static_range(0, POOL_SIZE):
        phys = (pool_logical_start + pool_slot) % RING
        src = tl.max(tl.where(earlier & (phys_slot == phys), t, -1), axis=0)
        src = tl.where(pool_slot == slot, tc, src)
        new = src >= 0
        row = tl.maximum(src, 0)
        score = tl.where(
            new,
            tl.load(in_s + row * ss_stride_t, mask=new & active, other=0.0),
            tl.load(ring + KPOOL_HEAD + phys * HEAD_DIM, mask=~new & active, other=0.0),
        ).to(tl.float32)
        k = tl.where(
            new,
            tl.load(in_k + row * key_stride_t, mask=new & active, other=0.0),
            tl.load(ring + phys * HEAD_DIM, mask=~new & active, other=0.0),
        ).to(tl.float32)
        score += tl.load(ape_ptr + pool_slot * ape_stride_0 + offs)
        prob = tl.exp(score - max_score)
        denom += prob
        acc += k * prob

    x = (acc / denom).to(tl.bfloat16).to(tl.float32)
    x = _hadamard128(x).to(tl.bfloat16).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-4)
    if ROUND_SCALE:
        scale = tl.exp2(tl.ceil(tl.log2(absmax * fp8_max_inv)))
    else:
        scale = absmax * fp8_max_inv
    quantized = tl.minimum(tl.maximum(x / scale, -FP8_MAX), FP8_MAX)

    loc_page_index = tloc // PAGE_SIZE
    loc_token_offset_in_page = tloc % PAGE_SIZE
    out_k_offsets = loc_page_index * BUF_NUMEL_PER_PAGE + _cache_k_offset(
        loc_token_offset_in_page, offs, HEAD_DIM, PRESHUFFLE
    )
    out_s_offset = (
        loc_page_index * BUF_NUMEL_PER_PAGE // 4
        + S_OFFSET_NBYTES_IN_PAGE // 4
        + loc_token_offset_in_page
    )
    tl.store(buf_fp8_ptr + out_k_offsets, quantized, mask=active)
    tl.store(buf_fp32_ptr + out_s_offset, scale, mask=active)


@triton.jit
def _kpool_decode_request(
    req,
    buf_fp8_ptr,
    buf_fp32_ptr,
    tail_kv_ptr,
    tail_slot_mapping_ptr,  # [B, NEXT_N] int32
    key_ptr,  # [B, NEXT_N, HEAD_DIM] bf16
    key_stride_b,
    key_stride_t,
    slot_score_ptr,  # [B, NEXT_N, HEAD_DIM] bf16
    ss_stride_b,
    ss_stride_t,
    ape_ptr,
    ape_stride_0,
    slot_mapping_ptr,  # [B, NEXT_N] int32
    positions_ptr,  # [B, NEXT_N] int32
    NEXT_N,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    RING: tl.constexpr,
    TAIL_BLOCK_ELEMS: tl.constexpr,
    KPOOL_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    PRESHUFFLE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """One program per request; its NEXT_N tokens are loaded as one tile.

    Same result as ``_kpool_decode_update_batched_kernel``, which walks the
    tokens in order so a pool completing at token t reads the ring as left by
    tokens < t. Here each ring read is resolved without that walk: if a token
    t' < t of this invocation stashed into the same (tail block, ring slot),
    the latest such t' is read straight from the inputs, otherwise the value
    predates the launch and comes from the ring. Only real completions (at
    most ``cdiv(NEXT_N, POOL_SIZE)`` for consecutive positions) run the pool
    math; every ring read lands before any stash (barrier), and each ring /
    cache slot is written only by its last writer, as in the ordered walk.
    """
    t = tl.arange(0, BLOCK_T)
    offs = tl.arange(0, HEAD_DIM)
    t_ok = t < NEXT_N
    idx = req * NEXT_N + t

    cache_loc = tl.load(slot_mapping_ptr + idx, mask=t_ok, other=-1)
    pos = tl.load(positions_ptr + idx, mask=t_ok, other=-1)
    tail_slot = tl.load(tail_slot_mapping_ptr + idx, mask=t_ok, other=-1)
    safe_pos = tl.maximum(pos, 0)
    phys_slot = safe_pos % RING
    block = tl.maximum(tail_slot, 0).to(tl.int64) // RING
    stash_valid = (pos >= 0) & (tail_slot >= 0)
    completes = (cache_loc >= 0) & (pos >= 0) & (safe_pos % POOL_SIZE == POOL_SIZE - 1)
    # A later completion into the same cache slot overwrites this one.
    later_loc = (
        (t[None, :] > t[:, None])
        & completes[None, :]
        & (cache_loc[None, :] == cache_loc[:, None])
    )
    completes = completes & (tl.max(later_loc.to(tl.int32), axis=1) == 0)
    c32 = completes.to(tl.int32)
    rank = tl.cumsum(c32, axis=0) - c32
    n_complete = tl.sum(c32, axis=0)

    row_k = key_ptr + req * key_stride_b + t[:, None] * key_stride_t + offs[None, :]
    row_s = (
        slot_score_ptr + req * ss_stride_b + t[:, None] * ss_stride_t + offs[None, :]
    )
    key = tl.load(row_k, mask=t_ok[:, None], other=0.0)
    score_all = tl.load(row_s, mask=t_ok[:, None], other=0.0)

    for c in tl.range(0, n_complete):
        _kpool_decode_complete_one(
            c,
            True,
            t,
            offs,
            completes,
            rank,
            safe_pos,
            cache_loc,
            block,
            phys_slot,
            stash_valid,
            req,
            buf_fp8_ptr,
            buf_fp32_ptr,
            tail_kv_ptr,
            key_ptr,
            key_stride_b,
            key_stride_t,
            slot_score_ptr,
            ss_stride_b,
            ss_stride_t,
            ape_ptr,
            ape_stride_0,
            PAGE_SIZE,
            BUF_NUMEL_PER_PAGE,
            POOL_SIZE,
            RING,
            TAIL_BLOCK_ELEMS,
            KPOOL_HEAD,
            HEAD_DIM,
            S_OFFSET_NBYTES_IN_PAGE,
            FP8_MAX,
            PRESHUFFLE,
            ROUND_SCALE,
        )

    # Every ring read above must land before any stash below overwrites it.
    tl.debug_barrier()
    later = (
        (t[None, :] > t[:, None])
        & stash_valid[None, :]
        & (block[None, :] == block[:, None])
        & (phys_slot[None, :] == phys_slot[:, None])
    )
    last_writer = stash_valid & (tl.max(later.to(tl.int32), axis=1) == 0)
    stash_off = (
        block[:, None] * TAIL_BLOCK_ELEMS
        + phys_slot[:, None] * HEAD_DIM
        + offs[None, :]
    )
    tl.store(tail_kv_ptr + stash_off, key, mask=last_writer[:, None])
    tl.store(tail_kv_ptr + KPOOL_HEAD + stash_off, score_all, mask=last_writer[:, None])


@triton.jit(do_not_specialize=["NEXT_N"])
def _kpool_decode_update_parallel_kernel(
    buf_fp8_ptr,
    buf_fp32_ptr,
    tail_kv_ptr,
    tail_slot_mapping_ptr,  # [B, NEXT_N] int32
    key_ptr,  # [B, NEXT_N, HEAD_DIM] bf16
    key_stride_b,
    key_stride_t,
    slot_score_ptr,  # [B, NEXT_N, HEAD_DIM] bf16
    ss_stride_b,
    ss_stride_t,
    ape_ptr,
    ape_stride_0,
    slot_mapping_ptr,  # [B, NEXT_N] int32
    positions_ptr,  # [B, NEXT_N] int32
    NEXT_N,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    RING: tl.constexpr,
    TAIL_BLOCK_ELEMS: tl.constexpr,
    KPOOL_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    PRESHUFFLE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """See ``_kpool_decode_request``."""
    _kpool_decode_request(
        tl.program_id(0),
        buf_fp8_ptr,
        buf_fp32_ptr,
        tail_kv_ptr,
        tail_slot_mapping_ptr,
        key_ptr,
        key_stride_b,
        key_stride_t,
        slot_score_ptr,
        ss_stride_b,
        ss_stride_t,
        ape_ptr,
        ape_stride_0,
        slot_mapping_ptr,
        positions_ptr,
        NEXT_N,
        PAGE_SIZE,
        BUF_NUMEL_PER_PAGE,
        POOL_SIZE,
        RING,
        TAIL_BLOCK_ELEMS,
        KPOOL_HEAD,
        HEAD_DIM,
        S_OFFSET_NBYTES_IN_PAGE,
        FP8_MAX,
        PRESHUFFLE,
        ROUND_SCALE,
        BLOCK_T,
    )


def kpool_decode_update_parallel(
    kv_cache: torch.Tensor,
    tail_kv_cache: torch.Tensor,
    tail_slot_mapping: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    pool_size: int,
    head_dim: int = INDEX_HEAD_DIM,
    round_scale: bool = True,
) -> None:
    """Drop-in for ``kpool_decode_update_and_maybe_write_cache_batched``.

    Same arguments and bit-identical kv / tail cache results; the request's
    ``next_n`` tokens are processed in parallel instead of one after another
    (see ``_kpool_decode_update_parallel_kernel``).
    """
    num_requests, next_n = key.shape[0], key.shape[1]
    if num_requests == 0 or next_n == 0:
        return
    assert head_dim == 128, "Hadamard-128 rotation"
    assert tail_kv_cache.ndim == 4
    assert tail_kv_cache.shape[1] == 2
    ring = tail_kv_cache.shape[2]
    assert ring >= pool_size and ring % pool_size == 0, (ring, pool_size)
    assert tail_kv_cache.shape[3] == head_dim
    assert tail_kv_cache.dtype == torch.bfloat16
    assert key.ndim == 3 and key.shape[2] == head_dim
    assert key.stride(2) == 1 and slot_score.stride(2) == 1
    assert slot_score.shape == key.shape
    assert ape.shape == (pool_size, head_dim)
    assert tail_slot_mapping.shape == (num_requests, next_n)
    assert slot_mapping.shape == (num_requests, next_n)
    assert positions.shape == (num_requests, next_n)
    assert key.dtype == torch.bfloat16
    assert slot_score.dtype == torch.bfloat16
    assert ape.dtype == torch.float32
    assert kv_cache.dtype == torch.uint8

    page_size = kv_cache.shape[1]
    if page_size > 1:
        assert page_size % 16 == 0, "ROCm preshuffle requires 16-token tiles"
    tail_slot_mapping = tail_slot_mapping.contiguous()
    slot_mapping = slot_mapping.contiguous()
    positions = positions.contiguous()
    ape = ape.contiguous()

    _kpool_decode_update_parallel_kernel[(num_requests,)](
        kv_cache.view(FP8_DTYPE),
        kv_cache.view(torch.float32),
        tail_kv_cache,
        tail_slot_mapping,
        key,
        key.stride(0),
        key.stride(1),
        slot_score,
        slot_score.stride(0),
        slot_score.stride(1),
        ape,
        ape.stride(0),
        slot_mapping,
        positions,
        next_n,
        PAGE_SIZE=page_size,
        BUF_NUMEL_PER_PAGE=kv_cache.stride(0),
        POOL_SIZE=pool_size,
        RING=ring,
        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),
        KPOOL_HEAD=tail_kv_cache.stride(1),
        HEAD_DIM=head_dim,
        S_OFFSET_NBYTES_IN_PAGE=page_size * head_dim,
        FP8_MAX=FP8_MAX,
        PRESHUFFLE=page_size > 1,
        ROUND_SCALE=round_scale,
        BLOCK_T=triton.next_power_of_2(next_n),
        num_warps=1,
    )


# One launch for a mixed step: decode update + prefill insert + tail seed.


@triton.jit
def _kpool_seed_windows(
    pid,
    k_ptr,
    score_ptr,
    k_stride,
    score_stride,
    tslot_ptr,
    tail_ptr,
    n_tokens,
    BLOCK_P: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    POOL_PAD: tl.constexpr,
    TAIL_BLOCK_ELEMS: tl.constexpr,
    KPOOL_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    RING: tl.constexpr,
):
    """``_kpool_tail_seed_kernel`` for the tokens of ``pid``'s windows.

    Token i is among its request's last POOL_SIZE tokens iff the token
    POOL_SIZE ahead has a different tail block (or is past the batch / has
    slot < 0); its raw K / gate go to ``tail[block, {0, 1}, tslot % RING]``.
    """
    TOK: tl.constexpr = BLOCK_P * POOL_PAD
    j = tl.arange(0, TOK)
    offs = tl.arange(0, HEAD_DIM)
    tok = (pid.to(tl.int64) * BLOCK_P + j // POOL_PAD) * POOL_SIZE + j % POOL_PAD
    ok = (j % POOL_PAD < POOL_SIZE) & (tok < n_tokens)
    ts = tl.load(tslot_ptr + tok, mask=ok, other=-1).to(tl.int64)
    ahead = tl.load(
        tslot_ptr + tok + POOL_SIZE,
        mask=ok & (tok + POOL_SIZE < n_tokens),
        other=-1,
    ).to(tl.int64)
    safe_ts = tl.maximum(ts, 0)
    blk = safe_ts // RING
    # A negative ahead slot never matches a real block -> token is in the tail.
    in_tail = ok & (ts >= 0) & ~((ahead >= 0) & (tl.maximum(ahead, 0) // RING == blk))
    m = in_tail[:, None]
    dst = blk[:, None] * TAIL_BLOCK_ELEMS + (safe_ts % RING)[:, None] * HEAD_DIM
    dst += offs[None, :]
    k = tl.load(k_ptr + tok[:, None] * k_stride + offs[None, :], mask=m)
    s = tl.load(score_ptr + tok[:, None] * score_stride + offs[None, :], mask=m)
    tl.store(tail_ptr + dst, k, mask=m)
    tl.store(tail_ptr + KPOOL_HEAD + dst, s, mask=m)


@triton.jit(do_not_specialize=["n_tokens", "NEXT_N", "num_decode"])
def _kpool_mixed_write_kernel(
    buf_fp8_ptr,
    buf_fp32_ptr,
    ape_ptr,
    ape_stride,
    # prefill slice
    k_ptr,
    score_ptr,
    k_stride,
    score_stride,
    slot_mapping_ptr,
    n_tokens,
    seed_tslot_ptr,
    # decode requests (+ the tail cache shared with seeding)
    tail_kv_ptr,
    dec_tail_slot_ptr,
    dec_key_ptr,
    dec_key_stride_b,
    dec_key_stride_t,
    dec_score_ptr,
    dec_score_stride_b,
    dec_score_stride_t,
    dec_slot_mapping_ptr,
    dec_positions_ptr,
    NEXT_N,
    num_decode,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    PRESHUFFLE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    BLOCK_P: tl.constexpr,
    POOL_PAD: tl.constexpr,
    RING: tl.constexpr,
    TAIL_BLOCK_ELEMS: tl.constexpr,
    KPOOL_HEAD: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_DECODE: tl.constexpr,
    HAS_PREFILL: tl.constexpr,
    SEED_TAIL: tl.constexpr,
):
    """Programs [0, num_decode) run one decode request each
    (``_kpool_decode_request``); the rest run prefill windows
    (``_kpool_insert_windows``, then ``_kpool_seed_windows``). The roles touch
    disjoint state: decode requests own their tail blocks, prefill requests
    theirs, and every pool has its own cache slot.
    """
    pid = tl.program_id(0)
    if pid < num_decode:
        if HAS_DECODE:
            _kpool_decode_request(
                pid,
                buf_fp8_ptr,
                buf_fp32_ptr,
                tail_kv_ptr,
                dec_tail_slot_ptr,
                dec_key_ptr,
                dec_key_stride_b,
                dec_key_stride_t,
                dec_score_ptr,
                dec_score_stride_b,
                dec_score_stride_t,
                ape_ptr,
                ape_stride,
                dec_slot_mapping_ptr,
                dec_positions_ptr,
                NEXT_N,
                PAGE_SIZE,
                BUF_NUMEL_PER_PAGE,
                POOL_SIZE,
                RING,
                TAIL_BLOCK_ELEMS,
                KPOOL_HEAD,
                HEAD_DIM,
                S_OFFSET_NBYTES_IN_PAGE,
                FP8_MAX,
                PRESHUFFLE,
                ROUND_SCALE,
                BLOCK_T,
            )
    else:
        if HAS_PREFILL:
            _kpool_insert_windows(
                pid - num_decode,
                k_ptr,
                score_ptr,
                ape_ptr,
                slot_mapping_ptr,
                buf_fp8_ptr,
                buf_fp32_ptr,
                n_tokens,
                k_stride,
                score_stride,
                ape_stride,
                PAGE_SIZE,
                BUF_NUMEL_PER_PAGE,
                POOL_SIZE,
                HEAD_DIM,
                S_OFFSET_NBYTES_IN_PAGE,
                FP8_MAX,
                PRESHUFFLE,
                ROUND_SCALE,
                BLOCK_P,
                POOL_PAD,
            )
            if SEED_TAIL:
                _kpool_seed_windows(
                    pid - num_decode,
                    k_ptr,
                    score_ptr,
                    k_stride,
                    score_stride,
                    seed_tslot_ptr,
                    tail_kv_ptr,
                    n_tokens,
                    BLOCK_P,
                    POOL_SIZE,
                    POOL_PAD,
                    TAIL_BLOCK_ELEMS,
                    KPOOL_HEAD,
                    HEAD_DIM,
                    RING,
                )


def _kpool_mixed_block_p(n_tokens: int, pool_size: int) -> int:
    """Pools per prefill program for ``kpool_mixed_write``.

    Chunked-prefill steps (<= 32k tokens on MI355X) favour one pool per
    program: the tail seeding and decode programs gain more from parallelism
    than the insert loses (block_p=1 within ~3% of the best at 2k-32k, 4-8%
    ahead of the standalone heuristic at 16k). Larger unchunked batches use
    the standalone insert tiers.
    """
    if n_tokens // pool_size <= 8192:
        return 1
    return _kpool_insert_launch_config(n_tokens, pool_size)[0]


def kpool_mixed_write(
    kv_cache: torch.Tensor,
    ape: torch.Tensor,
    pool_size: int,
    head_dim: int = INDEX_HEAD_DIM,
    round_scale: bool = True,
    *,
    prefill_k: torch.Tensor | None = None,
    prefill_gate: torch.Tensor | None = None,
    prefill_slot_mapping: torch.Tensor | None = None,
    prefill_tail_slot_mapping: torch.Tensor | None = None,
    tail_kv_cache: torch.Tensor | None = None,
    decode_tail_slot_mapping: torch.Tensor | None = None,
    decode_key: torch.Tensor | None = None,
    decode_gate: torch.Tensor | None = None,
    decode_slot_mapping: torch.Tensor | None = None,
    decode_positions: torch.Tensor | None = None,
    block_p: int | None = None,
) -> None:
    """Every kpool cache write of one indexer step in a single launch.

    Equivalent to (and bit-identical with) running, in any order:
      * ``kpool_compress_insert`` on the prefill slice (``prefill_*``),
      * ``kpool_seed_tail_cache`` on it when ``prefill_tail_slot_mapping`` is
        given,
      * ``kpool_decode_update_parallel`` on the decode requests
        (``decode_*``, grouped ``[num_requests, next_n]`` as for that op).
    Either side may be absent. ``tail_kv_cache`` is required for decode and
    for seeding.

    Args:
        kv_cache: indexer K cache ``[num_blocks, block_size, head_dim+4]`` uint8.
        ape: ``[pool_size, head_dim]`` fp32 per-slot position bias.
        pool_size: Number of tokens compressed into one cache entry.
        head_dim: Indexer head dimension (the Hadamard rotation needs 128).
        round_scale: Round each fp8 scale up to a power of two.
        prefill_k: ``[n_prefill, head_dim]`` bf16 prefill K.
        prefill_gate: ``[n_prefill, head_dim]`` bf16 prefill gate score.
        prefill_slot_mapping: ``[n_prefill]`` pool-granular slots.
        prefill_tail_slot_mapping: ``[n_prefill]`` token-granular tail slots.
        tail_kv_cache: ``[num_blocks, 2, ring, head_dim]`` bf16 tail cache.
        decode_tail_slot_mapping: ``[num_requests, next_n]`` int32.
        decode_key: ``[num_requests, next_n, head_dim]`` bf16.
        decode_gate: ``[num_requests, next_n, head_dim]`` bf16.
        decode_slot_mapping: ``[num_requests, next_n]`` int32.
        decode_positions: ``[num_requests, next_n]`` int32.
        block_p: Prefill pools-per-program override (benchmarking only).

    """
    assert head_dim == 128, "Hadamard-128 rotation"
    assert kv_cache.dtype == torch.uint8
    assert ape.shape == (pool_size, head_dim) and ape.dtype == torch.float32
    page_size = kv_cache.shape[1]
    if page_size > 1:
        assert page_size % 16 == 0, "ROCm preshuffle requires 16-token tiles"
    ape = ape.contiguous()

    n_tokens = 0 if prefill_k is None else prefill_k.shape[0]
    has_prefill = n_tokens >= pool_size or (
        n_tokens > 0 and prefill_tail_slot_mapping is not None
    )
    num_decode = 0 if decode_key is None else decode_key.shape[0]
    next_n = 0 if decode_key is None else decode_key.shape[1]
    has_decode = num_decode > 0 and next_n > 0
    seed = has_prefill and prefill_tail_slot_mapping is not None
    if not (has_prefill or has_decode):
        return
    if not has_prefill:
        # Decode-only steps: the dedicated kernel keeps a lighter program.
        kpool_decode_update_parallel(
            kv_cache,
            tail_kv_cache,
            decode_tail_slot_mapping,
            decode_key,
            decode_gate,
            ape,
            decode_slot_mapping,
            decode_positions,
            pool_size,
            head_dim,
            round_scale=round_scale,
        )
        return
    if has_decode or seed:
        assert tail_kv_cache is not None
        assert tail_kv_cache.ndim == 4 and tail_kv_cache.shape[1] == 2
        assert tail_kv_cache.shape[3] == head_dim
        assert tail_kv_cache.dtype == torch.bfloat16
        ring = tail_kv_cache.shape[2]
        assert ring >= pool_size and ring % pool_size == 0, (ring, pool_size)
        tail_block_elems, kpool_head = tail_kv_cache.stride(0), tail_kv_cache.stride(1)
    else:
        tail_kv_cache = kv_cache  # unused placeholder
        ring, tail_block_elems, kpool_head = pool_size, 1, 1

    if has_prefill:
        assert prefill_k.ndim == 2 and prefill_k.shape[1] == head_dim
        assert prefill_gate.shape == prefill_k.shape
        assert prefill_k.dtype == torch.bfloat16
        assert prefill_k.stride(1) == 1 and prefill_gate.stride(1) == 1
        assert prefill_slot_mapping.shape == (n_tokens,)
        prefill_slot_mapping = prefill_slot_mapping.contiguous()
        if seed:
            assert prefill_tail_slot_mapping.shape == (n_tokens,)
            prefill_tail_slot_mapping = prefill_tail_slot_mapping.contiguous()
        else:
            prefill_tail_slot_mapping = prefill_slot_mapping  # unused
        block_p = block_p or _kpool_mixed_block_p(n_tokens, pool_size)
        n_prefill_programs = triton.cdiv(triton.cdiv(n_tokens, pool_size), block_p)
    else:
        prefill_k = prefill_gate = ape  # unused placeholders
        prefill_slot_mapping = prefill_tail_slot_mapping = ape
        block_p, n_prefill_programs = 1, 0

    if has_decode:
        assert decode_key.ndim == 3 and decode_key.shape[2] == head_dim
        assert decode_key.dtype == torch.bfloat16
        assert decode_gate.shape == decode_key.shape
        assert decode_gate.dtype == torch.bfloat16
        assert decode_key.stride(2) == 1 and decode_gate.stride(2) == 1
        for m in (decode_tail_slot_mapping, decode_slot_mapping, decode_positions):
            assert m.shape == (num_decode, next_n)
        decode_tail_slot_mapping = decode_tail_slot_mapping.contiguous()
        decode_slot_mapping = decode_slot_mapping.contiguous()
        decode_positions = decode_positions.contiguous()
        dkey, dgate = decode_key, decode_gate
    else:
        num_decode = 0
        dkey = dgate = tail_kv_cache  # unused placeholders
        decode_tail_slot_mapping = decode_slot_mapping = decode_positions = ape

    _kpool_mixed_write_kernel[(num_decode + n_prefill_programs,)](
        kv_cache.view(FP8_DTYPE),
        kv_cache.view(torch.float32),
        ape,
        ape.stride(0),
        prefill_k,
        prefill_gate,
        prefill_k.stride(0),
        prefill_gate.stride(0),
        prefill_slot_mapping,
        n_tokens,
        prefill_tail_slot_mapping,
        tail_kv_cache,
        decode_tail_slot_mapping,
        dkey,
        dkey.stride(0),
        dkey.stride(1) if has_decode else 0,
        dgate,
        dgate.stride(0),
        dgate.stride(1) if has_decode else 0,
        decode_slot_mapping,
        decode_positions,
        next_n,
        num_decode,
        PAGE_SIZE=page_size,
        BUF_NUMEL_PER_PAGE=kv_cache.stride(0),
        POOL_SIZE=pool_size,
        HEAD_DIM=head_dim,
        S_OFFSET_NBYTES_IN_PAGE=page_size * head_dim,
        FP8_MAX=FP8_MAX,
        PRESHUFFLE=page_size > 1,
        ROUND_SCALE=round_scale,
        BLOCK_P=block_p,
        POOL_PAD=triton.next_power_of_2(pool_size),
        RING=ring,
        TAIL_BLOCK_ELEMS=tail_block_elems,
        KPOOL_HEAD=kpool_head,
        BLOCK_T=triton.next_power_of_2(max(next_n, 1)),
        HAS_DECODE=has_decode,
        HAS_PREFILL=has_prefill,
        SEED_TAIL=seed,
        num_warps=1,
    )


# Pool-level top-k helpers.


def history_group_budget_for_topk(topk: int, pool_size: int) -> int:
    """Number of pools to select so that expanding yields ``topk`` tokens."""
    assert topk % pool_size == 0
    return topk // pool_size


def expand_pools_to_tokens(
    group_ids: torch.Tensor,
    group_valid: torch.Tensor,
    topk: int,
    pool_size: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand selected full-pool ids to a strict-width token topk tensor."""
    assert group_ids.ndim == 2
    assert group_valid.shape == group_ids.shape
    assert topk % pool_size == 0
    assert group_ids.shape[1] == history_group_budget_for_topk(topk, pool_size)
    assert page_table is None or topk_offsets is None

    device = group_ids.device
    offsets = torch.arange(pool_size, device=device, dtype=torch.int64)
    token_ids = group_ids.to(torch.int64).unsqueeze(-1) * pool_size + offsets
    token_ids = token_ids.reshape(group_ids.shape[0], topk)
    valid = (
        group_valid.unsqueeze(-1)
        .expand(-1, -1, pool_size)
        .reshape(group_ids.shape[0], topk)
    )

    if page_table is not None:
        assert page_table.ndim == 2
        safe_ids = token_ids.clamp(min=0, max=page_table.shape[1] - 1)
        output = torch.gather(page_table, dim=1, index=safe_ids).to(torch.int32)
    elif topk_offsets is not None:
        if topk_offsets.ndim == 2:
            assert topk_offsets.shape[1] == 1
            topk_offsets = topk_offsets.squeeze(1)
        output = (token_ids + topk_offsets.to(torch.int64).unsqueeze(1)).to(torch.int32)
    else:
        output = token_ids.to(torch.int32)

    return torch.where(valid, output, torch.full_like(output, -1))


def append_tail_to_topk(
    topk_result: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_lens: torch.Tensor,
    pool_size: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Append non-pooled tail tokens after expanded history tokens.

    ``index_kpool_always_select_tail`` keeps the (incomplete) trailing pool so
    the most recent tokens are always attended to.
    """
    assert topk_result.dtype == torch.int32
    assert seq_lens.ndim == 1
    assert pool_lens.ndim == 1

    tail_pool = pool_size - 1
    if tail_pool == 0:
        return topk_result

    rows, n_cols = topk_result.shape
    out_cols = n_cols + tail_pool
    out = torch.empty(
        (rows, out_cols), dtype=topk_result.dtype, device=topk_result.device
    )

    # tail tokens: [pool_len*pool_size, seq_len) for each row.
    pool_len = pool_lens.to(torch.int32)
    tail_start = pool_len * pool_size
    seq_len = seq_lens.to(torch.int32)
    tail_count = seq_len - tail_start  # in [0, pool_size)

    cols = torch.arange(out_cols, device=topk_result.device)[None, :]
    history_len = n_cols
    is_history = cols < history_len
    tail_off = cols - history_len
    is_tail = (tail_off >= 0) & (tail_off < tail_count[:, None])

    # safe_hist must be per-row [rows, out_cols] so the gather reads each row's
    # OWN history. cols is [1, out_cols]; if used directly, gather (which does
    # NOT broadcast the index) would read only row 0 of topk_result, making every
    # query inherit row 0's history (empty for the first token) and lose all its
    # selected tokens — only the per-row tail would survive. This only manifests
    # for multi-row sparse PREFILL (decode has 1 row, so it reads its own row 0).
    safe_hist = torch.minimum(cols, torch.full_like(cols, n_cols - 1)).expand(
        rows, out_cols
    )
    history_val = torch.gather(topk_result, 1, safe_hist)

    tail_raw = tail_start[:, None] + tail_off
    tail_val = tail_raw.to(torch.int32)
    if page_table is not None:
        safe_tail = tail_raw.clamp(min=0, max=page_table.shape[1] - 1)
        tail_val = torch.gather(page_table, 1, safe_tail).to(torch.int32)
    elif topk_offsets is not None:
        tail_val = (tail_raw + topk_offsets.to(torch.int64).unsqueeze(1)).to(
            torch.int32
        )

    out = torch.where(is_history, history_val, -1)
    out = torch.where(is_tail, tail_val, out)
    return out


@triton.jit
def _expand_pools_and_append_tail_kernel(
    pool_ids_ptr,  # [rows, n_groups], int (any int dtype)
    seq_lens_ptr,  # [rows], int32 (token-granular seq_len)
    out_ptr,  # [rows, out_cols], int32
    topk,  # n_groups * pool_size
    out_cols,  # topk + pool_size - 1
    POOL_SIZE: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    pid_s0,
    out_s0,
):
    # Fuses expand_pools_to_tokens + append_tail_to_topk (identity path) into a
    # single kernel. Each program writes one (row, column-tile) of the output.
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = cols < out_cols

    seq_len = tl.load(seq_lens_ptr + row)
    pool_len = seq_len // POOL_SIZE
    tail_start = pool_len * POOL_SIZE
    tail_count = seq_len - tail_start  # in [0, POOL_SIZE)

    # History region [0, topk): expand selected pool g = cols // POOL_SIZE.
    is_history = cols < topk
    g = cols // POOL_SIZE
    o = cols % POOL_SIZE
    pid = tl.load(pool_ids_ptr + row * pid_s0 + g, mask=mask & is_history, other=-1)
    hist_val = (pid * POOL_SIZE + o).to(tl.int32)
    hist_out = tl.where(pid >= 0, hist_val, -1)

    # Tail region [topk, out_cols): the request's trailing incomplete pool.
    tail_off = cols - topk
    is_tail = (tail_off >= 0) & (tail_off < tail_count)
    tail_val = (tail_start + tail_off).to(tl.int32)
    tail_out = tl.where(is_tail, tail_val, -1)

    result = tl.where(is_history, hist_out, tail_out)
    tl.store(out_ptr + row * out_s0 + cols, result, mask=mask)


def expand_pools_and_append_tail(
    pool_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """Fuse ``expand_pools_to_tokens`` + ``append_tail_to_topk`` (identity path).

    Produces the same ``[rows, topk + pool_size - 1]`` int32 output as calling
    the two functions in sequence when neither ``page_table`` nor
    ``topk_offsets`` is passed — the only path used by the GLM-5.3-Flash indexer.
    The kernel derives ``pool_len = seq_len // pool_size`` internally, so the
    caller no longer needs to precompute it. Replaces ~25 elementwise kernels
    with one Triton launch.
    """
    rows, n_groups = pool_ids.shape
    topk = n_groups * pool_size
    out_cols = topk + pool_size - 1
    out = torch.empty((rows, out_cols), dtype=torch.int32, device=pool_ids.device)
    BLOCK_COLS = 128
    n_tiles = triton.cdiv(out_cols, BLOCK_COLS)
    _expand_pools_and_append_tail_kernel[(rows, n_tiles)](
        pool_ids,
        seq_lens,
        out,
        topk,
        out_cols,
        POOL_SIZE=pool_size,
        BLOCK_COLS=BLOCK_COLS,
        pid_s0=pool_ids.stride(0),
        out_s0=out.stride(0),
    )
    return out

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The HIP decode indexer logits kernel of _rocm_C
(csrc/rocm/fp8_paged_mqa_logits.cu):

    logits[m, p] = sum_h relu(Q[m,h,:] . K[p,:]) * w[m,h] * kv_scale[p]

over the paged fp8 indexer cache ([blocks, block_size, 1, 132] uint8: the keys
then the per-token fp32 scales; block_size 64 in the 16x16 preshuffled layout
the indexer writes, block_size 1 plain), gathered through a block table.

Every case is graded against an fp32 torch reference that reads the plain
(unshuffled) cache. The kernel writes only each row's causal window and leaves
the rest of the buffer alone, so the output is pre-filled with -inf and the
mask is checked exactly. Context lengths come in both forms the indexer uses:
one per sequence ([B], row n sees context_lens[b] - next_n + n + 1 tokens) and
one per query row ([B, next_n], native spec decode and the pooled kpool
lengths, which are not consecutive).

gfx950 only (v_mfma_scale_f32_32x32x64_f8f6f4, v_permlane32_swap).
"""

import random

import pytest
import torch

from vllm.platforms import current_platform

if current_platform.is_rocm():
    from vllm import _custom_ops as ops
    from vllm.platforms.rocm import on_gfx950

    _SUPPORTED = (
        hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "fp8_paged_mqa_logits")
        and on_gfx950()
    )
else:
    _SUPPORTED = False

pytestmark = pytest.mark.skipif(
    not _SUPPORTED, reason="needs the _rocm_C fp8_paged_mqa_logits op on gfx950"
)

DEVICE = "cuda"
NUM_HEADS = 32
HEAD_DIM = 128
FP8 = torch.float8_e4m3fn
TILE = 16


def kv_cache_cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
    """Co-pack a bf16 [blocks, block_size, 1, 128] cache into the fp8+scale byte
    layout: per block, block_size*128 fp8 bytes then block_size fp32 scales."""
    num_blocks, block_size, one, head_dim = x.shape
    assert one == 1
    amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = amax / 240.0
    x_fp8 = (x * (1.0 / sf)).to(FP8)
    out = torch.empty(
        (num_blocks, block_size * (head_dim + 4)), device=x.device, dtype=torch.uint8
    )
    out[:, : block_size * head_dim] = x_fp8.view(num_blocks, -1).view(torch.uint8)
    out[:, block_size * head_dim :] = sf.view(num_blocks, block_size).view(torch.uint8)
    return out.view(num_blocks, block_size, 1, head_dim + 4)


def preshuffle_kv_data(kv_cache_fp8: torch.Tensor) -> torch.Tensor:
    """The 16x16 MFMA preshuffle of the key bytes (aiter shuffle_weight
    layout=(16, 16) on fp8, what indexer_k_quant_and_cache writes for
    block_size > 1): byte (tok, d) of a block moves to
    (tok // 16) * 16 * 128 + (d // 16) * 256 + (tok % 16) * 16 + d % 16.
    The scale tail is untouched."""
    num_blocks, block_size, one, index_dim = kv_cache_fp8.shape
    assert block_size % TILE == 0
    flat = kv_cache_fp8.reshape(num_blocks, block_size * index_dim).clone()
    keys = flat[:, : block_size * HEAD_DIM].view(num_blocks, block_size, HEAD_DIM)
    shuffled = (
        keys.view(num_blocks, block_size // TILE, TILE, HEAD_DIM // TILE, TILE)
        .permute(0, 1, 3, 2, 4)
        .reshape(num_blocks, block_size * HEAD_DIM)
    )
    flat[:, : block_size * HEAD_DIM] = shuffled
    return flat.view(num_blocks, block_size, one, index_dim)


def ref_logits(q_fp8, kv_cache_fp8, weights, context_lens, block_tables, max_model_len):
    """fp32 reference over the plain cache; -inf outside each row's causal window.
    context_lens is [B] (row n sees ctx[b] - next_n + n + 1 tokens) or
    [B, next_n] (row n sees ctx[b, n] tokens)."""
    batch_size, next_n, _, dim = q_fp8.shape
    num_blocks, block_size, _, index_dim = kv_cache_fp8.shape
    flat = kv_cache_fp8.reshape(num_blocks, block_size * index_dim)
    keys = flat[:, : block_size * dim].contiguous().view(num_blocks, block_size, dim)
    scales = (
        flat[:, block_size * dim :].contiguous().view(torch.float32).view(num_blocks, block_size)
    )
    q = q_fp8.float()
    out = torch.full(
        (batch_size * next_n, max_model_len), float("-inf"), dtype=torch.float32, device=DEVICE
    )
    row_lens = context_lens.dim() == 2
    for b in range(batch_size):
        if row_lens:
            ends = context_lens[b].to(torch.int64)  # exclusive, per row
        else:
            ctx = int(context_lens[b])
            ends = ctx - next_n + 1 + torch.arange(next_n, device=DEVICE)
        n_tok = int(ends.max().item())
        if n_tok <= 0:
            continue
        pos = torch.arange(n_tok, device=DEVICE)
        blk = block_tables[b, pos // block_size].to(torch.int64)
        tok = pos % block_size
        k = keys[blk, tok].view(FP8).float() * scales[blk, tok][:, None]
        s = torch.relu(torch.einsum("nhd,pd->nhp", q[b], k))
        s = torch.einsum("nhp,nh->np", s, weights[b * next_n : (b + 1) * next_n])
        s = s.masked_fill(pos[None, :] >= ends[:, None], float("-inf"))
        out[b * next_n : (b + 1) * next_n, :n_tok] = s
    return out


def build_inputs(batch_size, next_n, avg_kv_len, block_size, var_ratio=0.3, seed=0,
                 per_row_lens=False):
    torch.manual_seed(seed)
    random.seed(seed)
    max_model_len = 2 * avg_kv_len
    if var_ratio == 0.0:
        context_lens = torch.full((batch_size,), avg_kv_len, dtype=torch.int32, device=DEVICE)
    else:
        lo = max(1, int((1 - var_ratio) * avg_kv_len))
        hi = int((1 + var_ratio) * avg_kv_len) + 1
        context_lens = torch.randint(lo, hi, (batch_size,), device=DEVICE).to(torch.int32)
    context_lens = context_lens.clamp(min=next_n)

    blocks_per_seq = (context_lens.to(torch.int64) + block_size - 1) // block_size
    max_block_len = int(blocks_per_seq.max().item())
    num_blocks = int(blocks_per_seq.sum().item())

    q = torch.randn((batch_size, next_n, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
    kv = torch.randn((num_blocks, block_size, 1, HEAD_DIM), dtype=torch.bfloat16, device=DEVICE)
    weights = torch.randn((batch_size * next_n, NUM_HEADS), dtype=torch.float32, device=DEVICE)

    # a shuffled block pool, so a kernel that walks the cache linearly is caught
    pool = list(range(num_blocks))
    random.shuffle(pool)
    pool_t = torch.tensor(pool, device=DEVICE, dtype=torch.int32)
    col = torch.arange(max_block_len, device=DEVICE, dtype=torch.int64)
    starts = torch.cumsum(blocks_per_seq, 0) - blocks_per_seq
    block_tables = torch.where(
        col[None, :] < blocks_per_seq[:, None],
        pool_t[(starts[:, None] + col[None, :]) % num_blocks],
        torch.zeros((), device=DEVICE, dtype=torch.int32),
    ).to(torch.int32)

    if per_row_lens:
        # the kpool indexer's pooled lengths: not consecutive, each row its own
        g = torch.Generator(device=DEVICE).manual_seed(seed + 1)
        drop = torch.randint(0, 3, (batch_size, next_n), generator=g, device=DEVICE)
        ends = context_lens[:, None] - drop.cumsum(1).flip(1)
        context_lens = ends.clamp(min=1).to(torch.int32).contiguous()

    kv_fp8 = kv_cache_cast_to_fp8(kv)
    return q.to(FP8), kv_fp8, weights, context_lens, block_tables, max_model_len


def calc_diff(x, y):
    x, y = x.double(), y.double()
    return 1 - 2 * (x * y).sum() / (x * x + y * y).sum()


def run_and_check(q_fp8, kv_fp8, weights, context_lens, block_tables, max_model_len,
                  block_size, **tunables):
    ref = ref_logits(q_fp8, kv_fp8, weights, context_lens, block_tables, max_model_len)
    kv_kernel = preshuffle_kv_data(kv_fp8) if block_size > 1 else kv_fp8
    out = torch.full_like(ref, float("-inf"))
    got = ops.fp8_paged_mqa_logits(
        q_fp8, kv_kernel, weights, context_lens, block_tables, max_model_len, out=out, **tunables
    )
    assert got is out
    torch.cuda.synchronize()
    ref_mask, out_mask = ref == float("-inf"), out == float("-inf")
    assert torch.equal(out_mask, ref_mask), (
        f"-inf mask mismatch at {int((out_mask != ref_mask).sum())} positions"
    )
    assert not torch.isnan(out).any()
    o, r = out.masked_fill(out_mask, 0), ref.masked_fill(ref_mask, 0)
    torch.testing.assert_close(o, r, rtol=1e-2, atol=5.0)
    assert calc_diff(o, r) < 1e-3


@pytest.mark.parametrize("batch_size", [1, 4, 16, 64])
@pytest.mark.parametrize("next_n", [1, 2, 4, 6])
@pytest.mark.parametrize("avg_kv_len, var_ratio", [(1024, 0.3), (8192, 0.3), (8192, 0.0)])
@torch.inference_mode()
def test_fp8_paged_mqa_logits_block64(batch_size, next_n, avg_kv_len, var_ratio):
    """Production layout: block_size 64, preshuffled keys, one length per sequence."""
    block_size = 64
    inp = build_inputs(batch_size, next_n, avg_kv_len, block_size, var_ratio)
    run_and_check(*inp, block_size)


@pytest.mark.parametrize("batch_size, next_n, avg_kv_len", [(1, 1, 32768), (4, 4, 32768), (1, 2, 131072), (16, 6, 32768)])
@torch.inference_mode()
def test_fp8_paged_mqa_logits_long_context(batch_size, next_n, avg_kv_len):
    """Long contexts: the small-chunk / non-temporal / R >= 3 LDS-ring configs."""
    block_size = 64
    inp = build_inputs(batch_size, next_n, avg_kv_len, block_size, 0.3)
    run_and_check(*inp, block_size)


@pytest.mark.parametrize("batch_size, next_n, avg_kv_len", [(4, 1, 1024), (16, 4, 1024), (2, 6, 8192)])
@torch.inference_mode()
def test_fp8_paged_mqa_logits_block1(batch_size, next_n, avg_kv_len):
    """block_size 1: the plain (unshuffled) cache and the register double buffer."""
    block_size = 1
    inp = build_inputs(batch_size, next_n, avg_kv_len, block_size, 0.3)
    run_and_check(*inp, block_size)


@pytest.mark.parametrize("batch_size, next_n, avg_kv_len", [(4, 2, 1024), (16, 4, 8192), (8, 6, 8192), (2, 5, 32768), (1, 7, 8192)])
@torch.inference_mode()
def test_fp8_paged_mqa_logits_per_row_lens(batch_size, next_n, avg_kv_len):
    """[B, next_n] context lengths, one per query row (native spec decode and
    the kpool indexer's pooled lengths); includes next_n 5 / 7, which round up
    to the R = 6 / 8 kernels with rows past next_n masked."""
    block_size = 64
    inp = build_inputs(batch_size, next_n, avg_kv_len, block_size, 0.3, per_row_lens=True)
    assert inp[3].shape == (batch_size, next_n)
    run_and_check(*inp, block_size)


@pytest.mark.parametrize(
    "tunables",
    [
        dict(num_warps=8, chunk_k=256),
        dict(num_warps=2, chunk_k=128, split_kv=3),
        dict(rows_per_block=1),
        dict(rows_per_block=3, chunk_k=64),
        dict(rows_per_block=6),
    ],
)
@torch.inference_mode()
def test_fp8_paged_mqa_logits_tunables(tunables):
    """Explicit grid choices give the same logits as the host heuristics."""
    block_size = 64
    inp = build_inputs(8, 6, 8192, block_size, 0.3, seed=3)
    run_and_check(*inp, block_size, **tunables)


@torch.inference_mode()
def test_fp8_paged_mqa_logits_allocates_and_rejects_bad_inputs():
    block_size = 64
    q_fp8, kv_fp8, weights, context_lens, block_tables, max_model_len = build_inputs(
        2, 1, 1024, block_size, 0.3
    )
    kv_kernel = preshuffle_kv_data(kv_fp8)
    out = ops.fp8_paged_mqa_logits(q_fp8, kv_kernel, weights, context_lens, block_tables, max_model_len)
    assert out.shape == (2, max_model_len) and out.dtype == torch.float32
    ref = ref_logits(q_fp8, kv_fp8, weights, context_lens, block_tables, max_model_len)
    valid = ref != float("-inf")
    torch.testing.assert_close(out[valid], ref[valid], rtol=1e-2, atol=5.0)
    with pytest.raises(RuntimeError, match="n_heads=32"):
        ops.fp8_paged_mqa_logits(q_fp8[:, :, :16], kv_kernel, weights, context_lens, block_tables, max_model_len)
    with pytest.raises(RuntimeError, match="block_size must be 1"):
        ops.fp8_paged_mqa_logits(
            q_fp8, kv_kernel.reshape(-1, 32, 1, 132), weights, context_lens, block_tables, max_model_len
        )
    with pytest.raises(RuntimeError, match="out must be"):
        ops.fp8_paged_mqa_logits(
            q_fp8, kv_kernel, weights, context_lens, block_tables, max_model_len,
            out=torch.empty(2, max_model_len - 1, dtype=torch.float32, device=DEVICE),
        )

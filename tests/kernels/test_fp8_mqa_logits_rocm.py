# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The HIP prefill indexer logits kernel of _rocm_C (csrc/rocm/fp8_mqa_logits.cu):

    logits[m, n] = sum_h relu(Q[m,h,:] . K[n,:]) * w[m,h] * kv_scale[n]
                   for n in [ks[m], ke[m]), -inf elsewhere

Every case is graded against an fp32 torch reference over exactly the fp8
inputs the kernel consumes. The output buffer is poisoned with NaN first, so
any position the kernel fails to write fails the -inf mask check instead of
inheriting a correct value from the caching allocator.

The window modes are the ones the indexer really hands the kernel: ``causal``
and ``cp`` (what the model produces), ``misaligned`` (windows that start off a
tile boundary), ``empty`` (rows with no window: cu_ends below zero or below
cu_starts), ``past_end`` (bounds beyond seq_len_kv, legal and simply empty over
the out-of-range part) and ``multi_req`` (several requests packed into one
chunk, so cu_starts jumps at every request boundary).

gfx950 only (v_mfma_f32_32x32x64_f8f6f4, v_permlane32_swap).
"""

import pytest
import torch

from vllm.platforms import current_platform

if current_platform.is_rocm():
    from vllm import _custom_ops as ops
    from vllm.platforms.rocm import on_gfx950

    _SUPPORTED = (
        hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "fp8_mqa_logits")
        and on_gfx950()
    )
else:
    _SUPPORTED = False

pytestmark = pytest.mark.skipif(
    not _SUPPORTED, reason="needs the _rocm_C fp8_mqa_logits op on gfx950"
)

DEVICE = "cuda"
NUM_HEADS = 32
HEAD_DIM = 128
FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max

# Peak fp32 bytes the reference may spend on its [heads, chunk, s_k] score
# tensor; rows are independent, so chunking over query rows is exact.
_REF_SCORE_BUDGET = 512 << 20


def per_token_cast_to_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-wise fp8 e4m3 quantization with an fp32 scale per row."""
    amax = x.abs().float().amax(dim=1, keepdim=True).clamp(1e-4)
    sf = amax / FP8_MAX
    return (x * (1.0 / sf)).to(FP8), sf.squeeze(1)


def ref_logits(q_fp8, k_fp8, scales, weights, ks, ke):
    """fp32 reference, chunked over query rows; also returns the valid count."""
    q = q_fp8.float()
    kv = k_fp8.float() * scales[:, None]
    s_q, s_k = q.shape[0], kv.shape[0]
    col = torch.arange(s_k, device=DEVICE)[None, :]
    mask = (col >= ks[:, None]) & (col < ke[:, None])
    out = torch.empty((s_q, s_k), dtype=torch.float32, device=DEVICE)
    chunk = max(1, _REF_SCORE_BUDGET // max(1, NUM_HEADS * s_k * 4))
    for m0 in range(0, s_q, chunk):
        m1 = min(m0 + chunk, s_q)
        score = torch.einsum("mhd,nd->hmn", q[m0:m1], kv)
        part = (score.relu() * weights[m0:m1].T[:, :, None]).sum(0)
        out[m0:m1] = part.masked_fill(~mask[m0:m1], float("-inf"))
        del score, part
    return out, int(mask.sum())


def _num_requests(s_q, s_k):
    return max(1, min(4, s_q // 2, s_k // 2))


def make_windows(s_q, s_k, window):
    rows = torch.arange(s_q, device=DEVICE)
    i32 = torch.int32
    if window == "causal":
        ks = torch.zeros(s_q, dtype=i32, device=DEVICE)
        ke = (torch.arange(s_q, device=DEVICE) + (s_k - s_q)).to(i32)
        return ks, ke
    if window == "cp":
        # context-parallel layout: one arbitrary CP rank's two half-chunks
        assert s_k % s_q == 0 and s_q % 2 == 0
        chunk, cp_size = s_q // 2, s_k // s_q
        cp_id = cp_size // 3
        i = torch.arange(chunk, device=DEVICE)
        ks = torch.zeros(s_q, dtype=i32, device=DEVICE)
        ke = torch.cat([cp_id * chunk + i, (cp_size * 2 - 1 - cp_id) * chunk + i])
        return ks, ke.to(i32)
    if window == "misaligned":
        ks = ((rows * 53 + 100) % max(1, s_k // 2)).to(i32)
        ke = torch.minimum(ks + max(1, s_k // 3), torch.full_like(ks, s_k))
        return ks, ke.to(i32)
    if window == "empty":
        ks = torch.where(rows % 3 == 1, min(100, s_k), 0)
        ke = torch.where(
            rows % 3 == 0,
            -1 - (rows % 7),
            torch.where(rows % 3 == 1, 0, torch.minimum(rows + 1, ks + s_k)),
        )
        return ks.to(i32), ke.to(i32)
    if window == "past_end":
        ks = torch.where(rows % 2 == 0, s_k + 17, 0)
        ke = torch.where(rows % 2 == 0, s_k + 64, torch.minimum(rows + 1, ks + s_k))
        return ks.to(i32), ke.to(i32)
    if window == "multi_req":
        nreq = _num_requests(s_q, s_k)
        rows_per_req = (s_q + nreq - 1) // nreq
        kv_per_req = s_k // nreq
        req = torch.clamp(rows // rows_per_req, max=nreq - 1)
        ks = req * kv_per_req
        local = rows - req * rows_per_req
        ke = ks + (kv_per_req - rows_per_req) + local + 1
        ke = torch.minimum(torch.clamp(ke, min=0), ks + kv_per_req)
        return ks.to(i32), torch.maximum(ke, ks).to(i32)
    raise ValueError(window)


def make_inputs(s_q, s_k, window, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    q = torch.randn(s_q, NUM_HEADS, HEAD_DIM, generator=g, device=DEVICE).bfloat16()
    kv = torch.randn(s_k, HEAD_DIM, generator=g, device=DEVICE).bfloat16()
    k_fp8, scales = per_token_cast_to_fp8(kv)
    weights = torch.randn(s_q, NUM_HEADS, generator=g, device=DEVICE)
    ks, ke = make_windows(s_q, s_k, window)
    return q.to(FP8), k_fp8, scales, weights, ks, ke


def rehydrate(out, ks, ke):
    """Re-apply the -inf mask a clean_logits=False launch did not write."""
    s_k = out.shape[1]
    col = torch.arange(s_k, device=DEVICE)[None, :]
    keep = (col >= ks.clamp(0, s_k)[:, None]) & (col < ke.clamp(0, s_k)[:, None])
    return torch.where(keep, out, torch.full_like(out, float("-inf")))


def calc_diff(x, y):
    x, y = x.double(), y.double()
    return 1 - 2 * (x * y).sum() / (x * x + y * y).sum()


def check(out, ref, n_valid, tag):
    out_mask, ref_mask = out == float("-inf"), ref == float("-inf")
    assert torch.equal(out_mask, ref_mask), (
        f"{tag}: -inf mask mismatch at {int((out_mask != ref_mask).sum())} positions"
    )
    assert not torch.isnan(out).any(), f"{tag}: unwritten (NaN) positions"
    o, r = out.masked_fill(out_mask, 0), ref.masked_fill(ref_mask, 0)
    torch.testing.assert_close(o, r, rtol=1e-2, atol=1e-2)
    if n_valid > 1:
        assert calc_diff(o, r) < 1e-3, f"{tag}: calc_diff too large"


SHAPES = [
    (1, 1),
    (1, 113),
    (17, 76),
    (61, 1024),
    (128, 1024),
    (1024, 1024),
    (1024, 1560),
    # small-M / long-KV: the launcher splits each row's window across grid.y
    (64, 2048),
    (64, 8192),
    # s_kv < s_q: cu_ends below zero on the leading rows
    (128, 64),
    (1024, 1000),
    # real prefill chunk shapes (M*N*4 under VLLM_SPARSE_INDEXER_MAX_LOGITS_MB)
    (4096, 32768),
    (1024, 131072),
]
WINDOWS = ["causal", "cp", "misaligned", "empty", "past_end", "multi_req"]


def _cases():
    for s_q, s_k in SHAPES:
        for window in WINDOWS:
            if window == "cp" and (s_k % s_q != 0 or s_q % 2 != 0):
                continue
            if window == "multi_req" and _num_requests(s_q, s_k) < 2:
                continue
            for clean in (True, False):
                yield s_q, s_k, window, clean


@pytest.mark.parametrize("s_q, s_k, window, clean_logits", list(_cases()))
@torch.inference_mode()
def test_fp8_mqa_logits(s_q, s_k, window, clean_logits):
    q_fp8, k_fp8, scales, weights, ks, ke = make_inputs(s_q, s_k, window)
    ref, n_valid = ref_logits(q_fp8, k_fp8, scales, weights, ks, ke)
    out = torch.full((s_q, s_k), float("nan"), dtype=torch.float32, device=DEVICE)
    got = ops.fp8_mqa_logits(
        q_fp8, k_fp8, scales, weights, ks, ke, clean_logits=clean_logits, out=out
    )
    assert got is out
    torch.cuda.synchronize()
    if not clean_logits:
        got = rehydrate(got, ks, ke)
    check(got, ref, n_valid, f"{s_q}x{s_k} {window} clean={clean_logits}")


@pytest.mark.parametrize("num_warps, split_n, reverse_rows", [(4, 0, -1), (8, 3, 0), (8, 16, 1)])
@torch.inference_mode()
def test_fp8_mqa_logits_tunables(num_warps, split_n, reverse_rows):
    """Explicit grid choices give the same logits as the host heuristics."""
    s_q, s_k = 1024, 8192
    q_fp8, k_fp8, scales, weights, ks, ke = make_inputs(s_q, s_k, "multi_req", seed=1)
    ref, n_valid = ref_logits(q_fp8, k_fp8, scales, weights, ks, ke)
    out = torch.full((s_q, s_k), float("nan"), dtype=torch.float32, device=DEVICE)
    ops.fp8_mqa_logits(
        q_fp8, k_fp8, scales, weights, ks, ke,
        num_warps=num_warps, split_n=split_n, reverse_rows=reverse_rows, out=out,
    )
    torch.cuda.synchronize()
    check(out, ref, n_valid, f"warps={num_warps} split={split_n} rev={reverse_rows}")


@torch.inference_mode()
def test_fp8_mqa_logits_allocates_and_rejects_bad_shapes():
    q_fp8, k_fp8, scales, weights, ks, ke = make_inputs(64, 2048, "causal")
    out = ops.fp8_mqa_logits(q_fp8, k_fp8, scales, weights, ks, ke)
    assert out.shape == (64, 2048) and out.dtype == torch.float32
    ref, n_valid = ref_logits(q_fp8, k_fp8, scales, weights, ks, ke)
    check(out, ref, n_valid, "allocating call")
    with pytest.raises(RuntimeError, match="n_heads=32"):
        ops.fp8_mqa_logits(q_fp8[:, :16], k_fp8, scales, weights, ks, ke)
    with pytest.raises(RuntimeError, match="float8_e4m3fn"):
        ops.fp8_mqa_logits(q_fp8.view(torch.int8), k_fp8, scales, weights, ks, ke)
    with pytest.raises(RuntimeError, match="out must be"):
        ops.fp8_mqa_logits(
            q_fp8, k_fp8, scales, weights, ks, ke,
            out=torch.empty(64, 2047, dtype=torch.float32, device=DEVICE),
        )

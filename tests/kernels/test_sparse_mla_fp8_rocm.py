# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The HIP FP8 sparse MLA kernel of _rocm_C (csrc/rocm/sparse_mla_fp8.cu):
sparse attention over a shared FP8 latent K/V pool with the BF16 query
encoded to FP8 inside the kernel and a BF16 output (GLM-5.x sparse attention
core, 64 heads x 512 latent dims, no appended RoPE).

Two independent references grade every case:

- ideal FP64 attention after the scalar Q encoding (quantization error of
  FP8 P and the BF16 split partials included): per-head relative RMS and
  elementwise error <= 0.07, LSE to 5e-4;
- an FP64 re-implementation of the kernel's arithmetic (FP8 P at scale
  1/256, BF16 split-store rounding, the kernel's split count) on dyadic
  inputs whose QK sums are exact: per-head relative RMS <= 0.005.

Also covered: the FP8-Q entry (bit-identical to the BF16 one), every dispatch
family and capacity, padded and tail-only selections, graph replay with the
inputs rewritten in place, and the metadata rejections.

gfx950 only.
"""

import math

import pytest
import torch

from vllm.platforms import current_platform

if current_platform.is_rocm():
    from vllm import _custom_ops as ops
    from vllm.platforms.rocm import on_gfx950

    _SUPPORTED = (
        hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "sparse_mla_fp8_fwd")
        and on_gfx950()
    )
else:
    _SUPPORTED = False

pytestmark = pytest.mark.skipif(
    not _SUPPORTED, reason="needs the _rocm_C sparse_mla_fp8_fwd op on gfx950"
)

DEVICE = "cuda"
FP8 = torch.float8_e4m3fn
TOLERANCE = 0.07
# Every dispatch family of the kernel (small / prefetch / rotating / streaming
# and their split counts), both sides of each boundary.
DISPATCH_QUERIES = [1, 6, 7, 8, 12, 13, 21, 22, 23, 24, 42, 43, 64, 65, 128, 129]
CAPACITIES = [0, 1, 63, 64, 127, 128, 129, 2048, 2051, 2111, 2112, 2113, 2176]


def inputs(n, capacity=2112, *, strided=False, seed=2031071, padded=False):
    torch.manual_seed(seed + n + capacity)
    shape = (n, 65, 520) if strided else (n, 64, 512)
    q = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)[:, :64, :512]
    kv = torch.randn((8192, 512), device=DEVICE, dtype=torch.bfloat16).to(FP8)
    ids = torch.randint(len(kv), (n, capacity), device=DEVICE, dtype=torch.int32)
    if capacity:
        if n > 1:
            ids[0].fill_(-1)
        if padded:
            ids[:, 2051:] = -1
        if n > 1:
            ids[1, : min(9, capacity)] = len(kv) - 1
            ids[1, 10::97] = len(kv) + 17
    return q, kv, ids


def reference_rows(n):
    """All small-Q rows; stratified rows and both sides of 2K boundaries otherwise."""
    if n <= 129:
        return list(range(n))
    rows = {0, 1, n - 1}
    rows.update(i * (n - 1) // 31 for i in range(32))
    for boundary in range(2048, n, 2048):
        rows.update([boundary - 1, boundary])
    return sorted(rows)


def assert_reference(out, lse, q, kv, ids, *, q_scale=1.0, kv_scale=1.0, scale=1 / 16,
                     all_rows=False):
    """Independent ideal FP64 attention after scalar Q encoding, in bounded batches."""
    assert out.shape == q.shape and out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
    rows = list(range(len(q))) if all_rows else reference_rows(len(q))
    for start in range(0, len(rows), 16):
        batch = rows[start : start + 16]
        encoded = q[batch] if q.dtype == FP8 else (q[batch].float() / q_scale).to(FP8)
        decoded = encoded.double() * q_scale
        selected = ids[batch].long()
        valid = (selected >= 0) & (selected < len(kv))
        values = kv.view(torch.uint8)[selected.clamp(0, len(kv) - 1)].view(kv.dtype).double() * kv_scale
        values[~valid] = 0
        scores = torch.bmm(decoded, values.transpose(1, 2)) * scale
        scores.masked_fill_(~valid[:, None, :], -math.inf)
        expected_lse = scores.logsumexp(-1)
        nonempty = valid.any(-1)
        probability = torch.where(nonempty[:, None, None], scores.softmax(-1), 0.0)
        expected = torch.bmm(probability, values)
        error = out[batch].double() - expected
        rrms = error.square().mean(-1).sqrt() / expected.square().mean(-1).sqrt().clamp_min(1e-3)
        assert rrms.max() <= TOLERANCE
        torch.testing.assert_close(out[batch].double(), expected, atol=TOLERANCE, rtol=TOLERANCE)
        if lse is not None:
            torch.testing.assert_close(lse[batch].double(), expected_lse, atol=5e-4, rtol=1e-5)
        assert torch.count_nonzero(out[batch][~nonempty]) == 0
        if lse is not None:
            assert torch.isneginf(lse[batch][~nonempty]).all()


@pytest.mark.parametrize("n", [1, 2, 3, 6, 7, 8, 12, 13, 21, 22, 23, 24, 42, 43, 64, 65, 128, 129, 2048, 4096])
@torch.inference_mode()
def test_quantized_reference(n):
    q, kv, ids = inputs(n)
    out, lse = ops.sparse_mla_fp8_fwd(q, kv, ids, 1 / 16, return_lse=True)
    assert_reference(out, lse, q, kv, ids)
    # the FP8-Q entry is the BF16 one after the same encoding, bit for bit
    control, control_lse = ops.sparse_mla_fp8_fwd(q.to(FP8), kv, ids, 1 / 16, return_lse=True)
    assert torch.equal(out, control)
    assert torch.equal(lse, control_lse)


@pytest.mark.parametrize("n", [1, 8, 13, 24, 65, 129])
@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("padded", [False, True])
@torch.inference_mode()
def test_capacity_stride_scale(n, capacity, padded):
    """Strided q rows, the 4-D kv / 3-D index views, decoding scales != 1."""
    q, kv, ids = inputs(n, capacity, strided=True, padded=padded)
    kwargs = {"q_scale": 0.5, "kv_scale": 1.25, "return_lse": True}
    out, lse = ops.sparse_mla_fp8_fwd(q, kv.view(-1, 1, 1, 512), ids[:, None, :], 1 / 16, **kwargs)
    assert_reference(out, lse, q, kv, ids, q_scale=0.5, kv_scale=1.25)
    control, control_lse = ops.sparse_mla_fp8_fwd((q.float() / 0.5).to(FP8), kv, ids, 1 / 16, **kwargs)
    assert torch.equal(out, control)
    assert torch.equal(lse, control_lse)


@pytest.mark.parametrize("n", [1, 8, 13, 24, 65, 129, 2048])
@torch.inference_mode()
def test_changed_graph(n):
    """A captured call replays on rewritten inputs, including masked tails."""
    q, kv, ids = inputs(n)
    workspace = ops.sparse_mla_fp8_workspace(n, q.device)
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    kwargs = {"out": out, "lse": lse, "return_lse": True, "workspace": workspace}
    ops.sparse_mla_fp8_fwd(q, kv, ids, 1 / 16, **kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured, captured_lse = ops.sparse_mla_fp8_fwd(q, kv, ids, 1 / 16, **kwargs)
    for replay in range(4):
        q.normal_()
        kv.copy_(torch.randn(kv.shape, device=kv.device, dtype=torch.bfloat16).to(kv.dtype))
        ids.random_(0, len(kv))
        if replay == 0:
            ids[:, 2051:] = -1
        elif replay == 1:
            ids[:, :128] = len(kv) - 1
        elif replay == 2:
            ids[:, 3::11] = -1
        else:
            ids.fill_(-1)
        out.fill_(math.nan)
        lse.fill_(math.nan)
        for t in workspace:
            t.fill_(math.nan)
        graph.replay()
        torch.cuda.synchronize()
        expected, expected_lse = ops.sparse_mla_fp8_fwd(q, kv, ids, 1 / 16, return_lse=True)
        assert torch.equal(captured, expected)
        assert torch.equal(captured_lse, expected_lse)
        assert_reference(out, lse, q, kv, ids)


@pytest.mark.parametrize("n", [4096, 16384])
@torch.inference_mode()
def test_large_query_chunk_equivalence(n):
    """Every row agrees with the 2048-row launches of the same inputs."""
    q, kv, ids = inputs(n, 2176, strided=True)
    kwargs = {"q_scale": 0.5, "kv_scale": 1.25, "return_lse": True}
    out, lse = ops.sparse_mla_fp8_fwd(q, kv, ids, 1 / 16, **kwargs)
    assert_reference(out, lse, q, kv, ids, q_scale=0.5, kv_scale=1.25, all_rows=True)
    for start in range(0, n, 2048):
        chunk_out, chunk_lse = ops.sparse_mla_fp8_fwd(
            q[start : start + 2048], kv, ids[start : start + 2048], 1 / 16, **kwargs
        )
        assert torch.equal(out[start : start + 2048], chunk_out)
        assert torch.equal(lse[start : start + 2048], chunk_lse)


def arithmetic_inputs(n, capacity, *, seed):
    """Dyadic Q/KV keep QK sums exact, isolating P and partial-store rounding."""
    q, kv, ids = inputs(n, capacity, strided=True, seed=seed)
    q.copy_((torch.randint(-8, 9, q.shape, device=q.device).float() / 8).to(q.dtype))
    kv.copy_((torch.randint(-8, 9, kv.shape, device=kv.device).float() / 8).to(kv.dtype))
    return q, kv, ids


def arithmetic_reference(q, kv, ids, *, q_scale=1.0, kv_scale=1.0, scale=1 / 16):
    """FP64 matmuls with the kernel's FP8 P (scale 1/256), BF16 split-store
    rounding and split count; no device layouts."""
    splits = ops.sparse_mla_fp8_split_count(len(q))
    outputs, lses = [], []
    for row in range(len(q)):
        encoded = q[row] if q.dtype == FP8 else (q[row].float() / q_scale).to(FP8)
        decoded = encoded.double() * q_scale
        selected = ids[row].long()
        valid = (selected >= 0) & (selected < len(kv))
        values = kv.view(torch.uint8)[selected.clamp(0, len(kv) - 1)].view(kv.dtype).double() * kv_scale
        values[~valid] = 0
        scores = (decoded @ values.T) * scale
        scores[:, ~valid] = -math.inf
        parts, part_lses = [], []
        tile_count = (len(selected) + 127) // 128
        quotient, remainder = divmod(tile_count, splits)
        begin = 0
        for split in range(splits):
            end = begin + quotient + (split < remainder)
            maximum = torch.full((64,), -math.inf, device=q.device, dtype=torch.float64)
            denominator = torch.zeros_like(maximum)
            numerator = torch.zeros((64, 512), device=q.device, dtype=torch.float64)
            for tile in range(begin, end):
                lo, hi = tile * 128, min((tile + 1) * 128, len(selected))
                block = scores[:, lo:hi]
                new_maximum = torch.maximum(maximum, block.amax(-1))
                alpha = torch.where(torch.isfinite(maximum), (maximum - new_maximum).exp(), 0.0)
                probability = torch.where(
                    torch.isfinite(new_maximum[:, None]), (block - new_maximum[:, None]).exp(), 0.0
                )
                rounded_p = (probability.float() * 256).to(FP8).double() / 256
                numerator = numerator * alpha[:, None] + rounded_p @ values[lo:hi]
                denominator = denominator * alpha + probability.sum(-1)
                maximum = new_maximum
            partial = numerator / denominator.clamp_min(1e-300)[:, None]
            if splits > 1:
                partial = partial.to(torch.bfloat16).double()
            parts.append(partial)
            part_lses.append(maximum + denominator.log())
            begin = end
        logs = torch.stack(part_lses)
        lse = logs.logsumexp(0)
        weights = torch.where(torch.isfinite(lse)[None, :], (logs - lse).exp(), 0.0)
        outputs.append((torch.stack(parts) * weights[:, :, None]).sum(0).to(torch.bfloat16))
        lses.append(lse)
    return torch.stack(outputs), torch.stack(lses)


def assert_arithmetic_reference(out, lse, expected, expected_lse):
    error = out.double() - expected.double()
    rrms = error.square().mean(-1).sqrt() / expected.double().square().mean(-1).sqrt().clamp_min(1e-3)
    assert rrms.max() <= 0.005
    torch.testing.assert_close(out.double(), expected.double(), atol=0.002, rtol=0.005)
    torch.testing.assert_close(lse.double(), expected_lse, atol=1e-5, rtol=1e-6)


@pytest.mark.parametrize("n", DISPATCH_QUERIES)
@pytest.mark.parametrize("capacity", [2112, 2176])
@pytest.mark.parametrize("tail_only", [False, True])
@torch.inference_mode()
def test_arithmetic_reference_valid_tail(n, capacity, tail_only):
    q, kv, ids = arithmetic_inputs(n, capacity, seed=38119)
    ids[:, 2051:] = torch.randint(len(kv), (n, capacity - 2051), device=q.device, dtype=torch.int32)
    if tail_only:
        ids[:, :2051] = -1
    kwargs = {"q_scale": 0.5, "kv_scale": 1.25, "return_lse": True}
    out, lse = ops.sparse_mla_fp8_fwd(q, kv, ids, 1 / 16, **kwargs)
    expected, expected_lse = arithmetic_reference(q, kv, ids, q_scale=0.5, kv_scale=1.25)
    assert_arithmetic_reference(out, lse, expected, expected_lse)
    assert_reference(out, lse, q, kv, ids, q_scale=0.5, kv_scale=1.25)


@torch.inference_mode()
def test_arithmetic_gate_rejects_small_error():
    q, kv, ids = arithmetic_inputs(8, 2176, seed=91737)
    expected, expected_lse = arithmetic_reference(q, kv, ids)
    out, lse = ops.sparse_mla_fp8_fwd(q, kv, ids, 1 / 16, return_lse=True)
    assert_arithmetic_reference(out, lse, expected, expected_lse)
    with pytest.raises(AssertionError):
        assert_arithmetic_reference(out.float() * 1.02, lse, expected, expected_lse)


@pytest.mark.parametrize("n", [4096, 16384])
@pytest.mark.parametrize("capacity", [2112, 2176])
@torch.inference_mode()
def test_large_query_all_rows_single_tail(n, capacity):
    """Analytic oracle for every element: one selected slot per row, in the tail."""
    torch.manual_seed(81771 + n + capacity)
    q = torch.randn((n, 64, 512), device=DEVICE, dtype=torch.bfloat16)
    kv = torch.randn((257, 512), device=DEVICE, dtype=torch.bfloat16).to(FP8)
    rows = torch.arange(n, device=q.device)
    slots = (rows * 37 + 11) % len(kv)
    positions = 2051 + rows % (capacity - 2051)
    positions[-1] = capacity - 1
    ids = torch.full((n, capacity), -1, device=q.device, dtype=torch.int32)
    ids[rows, positions] = slots.to(torch.int32)
    out, lse = ops.sparse_mla_fp8_fwd(q, kv, ids, 1 / 16, q_scale=0.5, kv_scale=1.25, return_lse=True)
    for start in range(0, n, 32):
        end = min(start + 32, n)
        values = kv.view(torch.uint8)[slots[start:end]].view(kv.dtype).double() * 1.25
        expected = values[:, None, :].expand(-1, 64, -1).to(torch.bfloat16)
        encoded = (q[start:end].float() / 0.5).to(FP8).double() * 0.5
        expected_lse = (encoded * values[:, None, :]).sum(-1) / 16
        torch.testing.assert_close(out[start:end], expected, atol=0, rtol=0)
        torch.testing.assert_close(lse[start:end].double(), expected_lse, atol=5e-4, rtol=1e-5)


@pytest.mark.parametrize("scale", [0.0, -1.0, math.inf, math.nan])
@pytest.mark.parametrize("argument", ["q_scale", "kv_scale", "softmax_scale"])
def test_bad_scales(argument, scale):
    q, kv, ids = inputs(8, 1)
    kwargs = {"softmax_scale": 1 / 16, argument: scale}
    with pytest.raises(RuntimeError, match="scale"):
        ops.sparse_mla_fp8_fwd(q, kv, ids, **kwargs)


def test_query_limit_and_metadata():
    q, kv, ids = inputs(8, 1)
    big_q = torch.empty((16385, 64, 512), device=DEVICE, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="16384"):
        ops.sparse_mla_fp8_fwd(
            big_q, kv, torch.empty((16385, 0), device=DEVICE, dtype=torch.int32), 1 / 16
        )
    with pytest.raises(RuntimeError, match="dtype|FP8|float8"):
        ops.sparse_mla_fp8_fwd(q, kv.to(torch.bfloat16), ids, 1 / 16)
    with pytest.raises(RuntimeError, match="dtype"):
        ops.sparse_mla_fp8_fwd(q, kv, ids.long(), 1 / 16)
    with pytest.raises(RuntimeError, match="overlap"):
        ops.sparse_mla_fp8_fwd(q, kv, ids, 1 / 16, out=q)
    unaligned = torch.empty((q.numel() + 1,), device=DEVICE, dtype=q.dtype)[1:].view(q.shape)
    with pytest.raises(RuntimeError, match="align"):
        ops.sparse_mla_fp8_fwd(unaligned, kv, ids, 1 / 16)
    with pytest.raises(RuntimeError, match="shape"):
        ops.sparse_mla_fp8_fwd(q, kv, ids, 1 / 16, workspace=ops.sparse_mla_fp8_workspace(1, DEVICE))
    oversized = torch.empty((8, 2177), device=DEVICE, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="2176"):
        ops.sparse_mla_fp8_fwd(q, kv, oversized, 1 / 16)
    empty, empty_lse = ops.sparse_mla_fp8_fwd(q[:0], kv, ids[:0], 1 / 16, return_lse=True)
    assert empty.shape == (0, 64, 512) and empty_lse.shape == (0, 64)

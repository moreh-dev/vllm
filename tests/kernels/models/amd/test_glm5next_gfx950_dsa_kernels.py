# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The in-tree HIP DSA kernels of GLM-5.3-Flash (ROCm/aiter ports in
`vllm/models/glm5next/amd/ops/hip`) against what they replace on gfx950.

The kpool writers must be byte-identical to the Triton writers, since both
fill the same cache. The logits kernels are compared with references at
AITER's own tolerances, the fused top-k with a top-k of the unfused logits,
and sparse MLA with FP64 attention after the kernel's FP8 query encoding.
"""

import math

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm.models.glm5next.amd.ops import hip  # noqa: E402

if not hip.on_gfx950():
    pytest.skip("the HIP DSA kernels need gfx950", allow_module_level=True)

from vllm.models.glm5next.amd.ops import kpool_compress as triton_kpool  # noqa: E402
from vllm.models.glm5next.amd.sparse_indexer import (  # noqa: E402
    _kpool_compress_insert,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (  # noqa: E402
    fp8_paged_mqa_logits_torch,
    indexer_k_quant_and_cache_triton,
)

POOL = 4
PAGE = 64
HEAD = 128
NUM_BLOCKS = 64
FP8 = torch.float8_e4m3fn


def _prefill_batch(request_lens: list[int], seed: int):
    """K, gate score and pool-granular slot mapping of a prefill batch whose
    requests start pool-aligned; each closing token gets a distinct slot."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    n = sum(request_lens)
    k = torch.randn(n, HEAD, generator=g, device="cuda").to(torch.bfloat16)
    gate = torch.randn(n, HEAD, generator=g, device="cuda").to(torch.bfloat16)
    ape = torch.randn(POOL, HEAD, generator=g, device="cuda")
    positions = torch.cat([torch.arange(length) for length in request_lens]).cuda()
    closing = positions % POOL == POOL - 1
    slots = torch.randperm(NUM_BLOCKS * PAGE, generator=g, device="cuda")
    slot_mapping = torch.full((n,), -1, dtype=torch.int64, device="cuda")
    slot_mapping[closing] = slots[: int(closing.sum())]
    return k, gate, ape, positions, slot_mapping


def _caches(ring: int) -> tuple[torch.Tensor, torch.Tensor]:
    kv = torch.zeros(NUM_BLOCKS, PAGE, HEAD + 4, dtype=torch.uint8, device="cuda")
    tail = torch.zeros(NUM_BLOCKS, 2, ring, HEAD, dtype=torch.bfloat16, device="cuda")
    return kv, tail


def _assert_same_bytes(actual: torch.Tensor, expected: torch.Tensor) -> None:
    mismatch = actual.view(torch.uint8) != expected.view(torch.uint8)
    assert not mismatch.any(), f"{int(mismatch.sum())} bytes differ"


def test_kpool_insert_matches_triton():
    k, gate, ape, _, slot_mapping = _prefill_batch([10, 37, 64, 5, 128], seed=0)
    expected, _ = _caches(POOL)
    actual = expected.clone()
    _kpool_compress_insert(
        k, gate, ape, expected, slot_mapping, POOL, HEAD, round_scale=True
    )
    hip.kpool_compress_insert(actual, k, gate, ape, slot_mapping, POOL)
    _assert_same_bytes(actual, expected)


def test_kpool_insert_and_seed_tail_matches_triton():
    request_lens = [10, 37, 64, 5, 128]
    k, gate, ape, positions, slot_mapping = _prefill_batch(request_lens, seed=1)
    tail_blocks = torch.repeat_interleave(
        torch.arange(len(request_lens), device="cuda") * 3 + 2,
        torch.tensor(request_lens, device="cuda"),
    )
    tail_slot_mapping = tail_blocks * POOL + positions % POOL
    kv_expected, tail_expected = _caches(POOL)
    kv_actual, tail_actual = kv_expected.clone(), tail_expected.clone()
    _kpool_compress_insert(
        k, gate, ape, kv_expected, slot_mapping, POOL, HEAD, round_scale=True
    )
    triton_kpool.kpool_seed_tail_cache(
        tail_expected, k, gate, tail_slot_mapping, POOL, HEAD
    )
    hip.kpool_insert_and_seed_tail(
        kv_actual, tail_actual, k, gate, ape, slot_mapping, tail_slot_mapping, POOL
    )
    _assert_same_bytes(kv_actual, kv_expected)
    _assert_same_bytes(tail_actual, tail_expected)


@pytest.mark.parametrize("next_n,ring", [(1, 4), (2, 8)])
def test_kpool_decode_update_matches_triton(next_n: int, ring: int):
    num_requests = 24
    g = torch.Generator(device="cuda").manual_seed(next_n)
    kv_expected, tail_expected = _caches(ring)
    tail_expected.copy_(torch.randn(tail_expected.shape, generator=g, device="cuda"))
    kv_actual, tail_actual = kv_expected.clone(), tail_expected.clone()
    shape = (num_requests, next_n)
    start = torch.randint(0, 4096, (num_requests, 1), generator=g, device="cuda")
    positions = start + torch.arange(next_n, device="cuda")
    # int64 metadata, as the indexer metadata hands it over.
    tail_blocks = torch.randperm(NUM_BLOCKS, generator=g, device="cuda")
    tail_slot = tail_blocks[:num_requests, None] * ring + positions % ring
    slots = torch.randperm(NUM_BLOCKS * PAGE, generator=g, device="cuda")
    slot_mapping = torch.where(
        positions % POOL == POOL - 1, slots[: num_requests * next_n].view(shape), -1
    )
    key = torch.randn(*shape, HEAD, generator=g, device="cuda").to(torch.bfloat16)
    score = torch.randn(*shape, HEAD, generator=g, device="cuda").to(torch.bfloat16)
    ape = torch.randn(POOL, HEAD, generator=g, device="cuda")
    triton_kpool.kpool_decode_update_and_maybe_write_cache_batched(
        kv_expected,
        tail_expected,
        tail_slot,
        key,
        score,
        ape,
        slot_mapping,
        positions,
        POOL,
        HEAD,
        round_scale=True,
    )
    hip.kpool_decode_update(
        kv_actual,
        tail_actual,
        tail_slot,
        key,
        score,
        ape,
        slot_mapping,
        positions,
        POOL,
    )
    _assert_same_bytes(kv_actual, kv_expected)
    _assert_same_bytes(tail_actual, tail_expected)


def _prefill_logits_inputs(request_lens: list[int], seed: int):
    """Indexer q/k/weights of a prefill chunk whose rows see their request's
    keys causally."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    m = n = sum(request_lens)
    q = torch.randn(m, hip.INDEXER_HEADS, HEAD, generator=g, device="cuda").to(FP8)
    k = torch.randn(n, HEAD, generator=g, device="cuda").to(FP8)
    kv_scale = torch.rand(n, generator=g, device="cuda") + 0.5
    weights = torch.randn(m, hip.INDEXER_HEADS, generator=g, device="cuda")
    starts = torch.cumsum(torch.tensor([0] + request_lens[:-1]), 0)
    ks = torch.repeat_interleave(starts, torch.tensor(request_lens)).int().cuda()
    ke = (torch.arange(m, dtype=torch.int32, device="cuda") + 1).contiguous()
    return q, k, kv_scale, weights, ks, ke


def _mqa_logits_reference(q, k, kv_scale, weights, ks, ke) -> torch.Tensor:
    scores = torch.einsum("mhd,nd->mhn", q.float(), k.float()).relu()
    logits = (scores * weights[:, :, None]).sum(1) * kv_scale[None, :]
    cols = torch.arange(k.shape[0], device=q.device)
    window = (cols[None, :] >= ks[:, None]) & (cols[None, :] < ke[:, None])
    return logits.masked_fill(~window, -math.inf)


def test_fp8_mqa_logits_matches_reference():
    inputs = _prefill_logits_inputs([700, 1, 300, 1100], seed=2)
    expected = _mqa_logits_reference(*inputs)
    actual = hip.fp8_mqa_logits(*inputs, clean_logits=True)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_fp8_mqa_topk_matches_logits_topk():
    topk = hip.FUSED_TOPK_MAX_K
    q, k, kv_scale, weights, ks, ke = _prefill_logits_inputs([700, 1, 300, 1100], 3)
    logits = hip.fp8_mqa_logits(q, k, kv_scale, weights, ks, ke, clean_logits=True)
    indices = torch.empty(q.shape[0], topk, dtype=torch.int32, device="cuda")
    hip.fp8_mqa_topk(q, k, kv_scale, weights, ks, ke, indices)
    lengths = (ke - ks).long()
    for row in range(q.shape[0]):
        length = int(lengths[row])
        selected = indices[row]
        if length <= topk:
            expected = torch.arange(topk, device="cuda", dtype=torch.int32)
            expected[length:] = -1
            torch.testing.assert_close(selected, expected)
            continue
        assert (selected >= 0).all() and selected.unique().numel() == topk
        row_logits = logits[row, ks[row] : ke[row]]
        kth = row_logits.topk(topk).values[-1]
        # Ties at the k-th value may be broken either way.
        assert (row_logits[selected.long()] >= kth).all()


def test_fp8_paged_mqa_logits_matches_reference():
    batch, max_model_len = 20, 4096
    g = torch.Generator(device="cuda").manual_seed(4)
    context_lens = torch.randint(
        1, max_model_len, (batch,), generator=g, device="cuda", dtype=torch.int32
    )
    blocks_per_seq = max_model_len // PAGE
    kv_cache = torch.zeros(
        batch * blocks_per_seq, PAGE, HEAD + 4, dtype=torch.uint8, device="cuda"
    )
    block_tables = (
        torch.randperm(batch * blocks_per_seq, generator=g, device="cuda")
        .view(batch, blocks_per_seq)
        .int()
    )
    k = torch.randn(batch * max_model_len, HEAD, generator=g, device="cuda")
    pages = block_tables.long().repeat_interleave(PAGE, dim=1)
    offsets = torch.arange(max_model_len, device="cuda") % PAGE
    slot_mapping = (pages * PAGE + offsets).flatten()
    indexer_k_quant_and_cache_triton(
        k.to(torch.bfloat16), kv_cache, slot_mapping, HEAD, "ue8m0"
    )
    q = torch.randn(batch, 1, hip.INDEXER_HEADS, HEAD, generator=g, device="cuda")
    q = q.to(FP8)
    weights = torch.randn(batch, hip.INDEXER_HEADS, generator=g, device="cuda")
    cache_view = kv_cache.unsqueeze(-2)
    expected = fp8_paged_mqa_logits_torch(
        q, cache_view, weights, context_lens, block_tables, max_model_len, shuffled=True
    )
    actual = torch.full_like(expected, -math.inf)
    hip.fp8_paged_mqa_logits(
        q, cache_view, weights, context_lens, block_tables, max_model_len, actual
    )
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=5.0)


def _sparse_mla_reference(q, kv, ids, scale, q_scale, kv_scale):
    """FP64 attention after the kernel's FP8 encoding of `q / q_scale`."""
    decoded = (q.float() / q_scale).to(FP8).double() * q_scale
    valid = (ids >= 0) & (ids < kv.shape[0])
    values = kv[ids.long().clamp(0, kv.shape[0] - 1)].double() * kv_scale
    values[~valid] = 0
    scores = torch.bmm(decoded, values.transpose(1, 2)) * scale
    scores.masked_fill_(~valid[:, None, :], -math.inf)
    probs = torch.where(valid.any(-1)[:, None, None], scores.softmax(-1), 0.0)
    return torch.bmm(probs.nan_to_num(), values)


@pytest.mark.parametrize("queries", [1, 7, 64, 300])
def test_sparse_mla_fp8_matches_reference(queries: int):
    g = torch.Generator(device="cuda").manual_seed(queries)
    q = torch.randn(
        queries,
        hip.SPARSE_MLA_HEADS,
        hip.SPARSE_MLA_HEAD_DIM,
        generator=g,
        device="cuda",
    ).to(torch.bfloat16)
    kv = torch.randn(8192, hip.SPARSE_MLA_HEAD_DIM, generator=g, device="cuda").to(FP8)
    ids = torch.randint(
        0, kv.shape[0], (queries, 2048), generator=g, device="cuda", dtype=torch.int32
    )
    ids[:, 1900:] = -1
    ids[0, :7] = kv.shape[0] + 5
    if queries > 1:
        ids[1] = -1
    scale, q_scale, kv_scale = 1 / 16, 0.5, 1.0
    out, _ = hip.sparse_mla_fp8_fwd(q, kv, ids, scale, q_scale, kv_scale)
    expected = _sparse_mla_reference(q, kv, ids, scale, q_scale, kv_scale)
    torch.testing.assert_close(out.double(), expected, atol=0.07, rtol=0.07)

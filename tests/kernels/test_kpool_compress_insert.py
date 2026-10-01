# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Op test for the fused kpool prefill insert (``kpool_compress_insert``).

The fused kernel reads the token-major K / gate score in place and replaces
the prefill path's gather (``k[idx]``, ``gate_score[idx]``) +
``kpool_compress_and_write_cache``. Two checks:

* bit-exact against that unfused production path, byte for byte over the
  whole cache (so untouched entries must stay untouched too);
* close to an independent pure-torch reference (softmax pool -> bf16 ->
  Hadamard-128 -> bf16 -> per-vector fp8 absmax quant), decoded from the
  cache layout, so a shared bug in both Triton kernels would not pass.
"""

import math

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm kpool kernels", allow_module_level=True)

from vllm.models.glm5next.amd.ops.kpool_compress import (  # noqa: E402
    kpool_compress_and_write_cache,
    kpool_compress_insert,
    kpool_decode_update_and_maybe_write_cache_batched,
    kpool_mixed_write,
    kpool_seed_tail_cache,
)

HEAD_DIM = 128
FP8_DTYPE = current_platform.fp8_dtype()
FP8_MAX = torch.finfo(FP8_DTYPE).max


def _make_batch(lens, pool_size, page_size, slot_dtype, seed=0, lead_pad=0):
    """Varlen prefill batch; pools aligned to each request start.

    Returns k, gate, ape, pool-granular slot_mapping and a random cache.
    ``lead_pad`` prepends padding tokens (slot -1).
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    slots = [-1] * lead_pad
    n_pools = sum(length // pool_size for length in lens)
    num_blocks = max(1, math.ceil(n_pools / page_size)) + 2
    perm = torch.randperm(num_blocks * page_size, device="cuda", generator=g)
    perm = perm[:n_pools].tolist()
    for length in lens:
        for t in range(length):
            closes = t % pool_size == pool_size - 1
            slots.append(perm.pop() if closes else -1)
    n = len(slots)
    k = torch.randn(n, HEAD_DIM, generator=g, device="cuda").to(torch.bfloat16)
    gate = (torch.randn(n, HEAD_DIM, generator=g, device="cuda") * 3).to(torch.bfloat16)
    ape = torch.randn(pool_size, HEAD_DIM, generator=g, device="cuda")
    slot_mapping = torch.tensor(slots, dtype=slot_dtype, device="cuda")
    kv = torch.randint(
        0,
        256,
        (num_blocks, page_size, HEAD_DIM + 4),
        generator=g,
        dtype=torch.uint8,
        device="cuda",
    )
    return k, gate, ape, slot_mapping, kv


def _unfused_insert(kv, k, gate, ape, slot_mapping, pool_size, round_scale):
    """The production prefill path this kernel replaces."""
    n = slot_mapping.shape[0]
    if n < pool_size:
        return
    pos = torch.arange(n, device=k.device)
    write_mask = (slot_mapping >= 0) & (pos >= pool_size - 1)
    offs = torch.arange(pool_size, device=k.device)
    idx = (pos - (pool_size - 1)).clamp_min(0)[:, None] + offs[None, :]
    kpool_compress_and_write_cache(
        kv,
        k[idx],
        gate[idx],
        ape,
        slot_mapping.to(torch.int64),
        pool_size=pool_size,
        head_dim=HEAD_DIM,
        write_mask=write_mask,
        round_scale=round_scale,
    )


def _hadamard128(x):
    h = torch.ones(1, 1, device=x.device)
    while h.shape[0] < HEAD_DIM:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return x @ (h / math.sqrt(HEAD_DIM))


def _torch_reference(k, gate, ape, slot_mapping, pool_size, round_scale):
    """Independent pure-torch compressed K / scale for every closing token."""
    pos = torch.arange(k.shape[0], device=k.device)
    rows = torch.nonzero((slot_mapping >= 0) & (pos >= pool_size - 1)).squeeze(1)
    idx = rows[:, None] - (pool_size - 1) + torch.arange(pool_size, device=k.device)
    score = gate[idx].float() + ape[None]
    pooled = (torch.softmax(score, dim=1) * k[idx].float()).sum(1)
    x = _hadamard128(pooled.bfloat16().float()).bfloat16().float()
    absmax = x.abs().amax(-1).clamp_min(1e-4)
    scale = absmax / FP8_MAX
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    q = (x / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return slot_mapping[rows].long(), q, scale


def _read_cache(kv, locs):
    """Decode (fp8 K, fp32 scale) at pool slots from the ROCm cache layout."""
    page_size = kv.shape[1]
    page, off = locs // page_size, locs % page_size
    flat = kv.reshape(kv.shape[0], -1)
    d = torch.arange(HEAD_DIM, device=kv.device)
    if page_size > 1:
        k_off = (
            (off[:, None] // 16) * 16 * HEAD_DIM
            + (d[None] // 16) * 256
            + (off[:, None] % 16) * 16
            + d[None] % 16
        )
    else:
        k_off = off[:, None] * HEAD_DIM + d[None]
    q = torch.gather(flat[page], 1, k_off).view(FP8_DTYPE)
    s_off = page_size * HEAD_DIM + 4 * off[:, None] + torch.arange(4, device=kv.device)
    scale = torch.gather(flat[page], 1, s_off).contiguous().view(torch.float32)
    return q, scale.squeeze(1)


def _check(lens, pool_size, page_size, slot_dtype, round_scale=True, **kw):
    k, gate, ape, slot_mapping, kv = _make_batch(
        lens, pool_size, page_size, slot_dtype, **kw
    )
    expected = kv.clone()
    _unfused_insert(expected, k, gate, ape, slot_mapping, pool_size, round_scale)
    actual = kv.clone()
    kpool_compress_insert(
        actual, k, gate, ape, slot_mapping, pool_size, round_scale=round_scale
    )
    torch.accelerator.synchronize()
    assert torch.equal(actual, expected), (
        f"{int((actual != expected).sum())} cache bytes differ from unfused path"
    )

    locs, ref_q, ref_scale = _torch_reference(
        k, gate, ape, slot_mapping, pool_size, round_scale
    )
    if locs.numel() == 0:
        return
    q, scale = _read_cache(actual, locs)
    # torch's exp / sum order differs from Triton's, so the pooled vector can
    # land one bf16 ulp apart before the rotation; the Hadamard spreads that
    # over the whole row. Allow one e4m3 step (<= 12.5% of the value) plus
    # one bf16 ulp of the row absmax, and require flips to stay rare.
    torch.testing.assert_close(
        scale, ref_scale, rtol=0 if round_scale else 1e-2, atol=0
    )
    deq = q.float() * scale[:, None]
    ref = ref_q.float() * ref_scale[:, None]
    tol = ref.abs() * 0.125 + ref.abs().amax(-1, keepdim=True) * 2**-8
    assert ((deq - ref).abs() <= tol).all()
    assert (q.float() != ref_q.float()).float().mean() < 0.02


@pytest.mark.parametrize("pool_size", [4, 2, 8])
@pytest.mark.parametrize("page_size", [16, 32, 64])
@pytest.mark.parametrize("slot_dtype", [torch.int64, torch.int32])
def test_single_request(pool_size, page_size, slot_dtype):
    _check([4096], pool_size, page_size, slot_dtype)


@pytest.mark.parametrize(
    "lens",
    [
        [1, 2, 3],  # no pool closes
        [5],
        [1, 7, 130, 1000, 33, 4, 5],  # pools straddle every tile offset
        [1024] * 4,
        [4099, 17, 2048, 3],
    ],
)
@pytest.mark.parametrize("pool_size", [4, 3])
def test_varlen_batch(lens, pool_size):
    _check(lens, pool_size, 64, torch.int64)


def test_page_size_one():
    _check([1, 7, 130, 1000], 4, 1, torch.int64)


@pytest.mark.parametrize("round_scale", [True, False])
def test_round_scale(round_scale):
    _check([300, 77], 4, 32, torch.int64, round_scale=round_scale)


def test_leading_padding():
    # Pools cut by the batch start are dropped (no pool_size-1 predecessors).
    _check([64, 9], 4, 32, torch.int64, lead_pad=2)
    k, gate, ape, slot_mapping, kv = _make_batch([8], 4, 32, torch.int64)
    slot_mapping[1] = 5  # closes a pool that would start before the batch
    expected = kv.clone()
    _unfused_insert(expected, k, gate, ape, slot_mapping, 4, True)
    actual = kv.clone()
    kpool_compress_insert(actual, k, gate, ape, slot_mapping, 4)
    assert torch.equal(actual, expected)


def test_large_batch():
    # Several full-grid tiles and multi-page caches (1k ISL x 64 requests).
    _check([1024] * 64, 4, 32, torch.int64)


def test_dense_slot_mapping_matches_unfused():
    # Not produced by pool-aligned batches, but the kernel must still agree
    # with the unfused path when several closing tokens share a window.
    n, pool_size, page_size = 512, 4, 16
    k, gate, ape, _, _ = _make_batch([n], pool_size, page_size, torch.int64)
    gen = torch.Generator(device="cuda").manual_seed(1)
    kv = torch.zeros(
        n // page_size, page_size, HEAD_DIM + 4, dtype=torch.uint8, device="cuda"
    )
    closes = torch.rand(n, generator=gen, device="cuda") < 0.6
    slots = torch.randperm(n, generator=gen, device="cuda")[: int(closes.sum())]
    slot_mapping = torch.full((n,), -1, dtype=torch.int64, device="cuda")
    slot_mapping[closes] = slots
    expected = kv.clone()
    _unfused_insert(expected, k, gate, ape, slot_mapping, pool_size, True)
    actual = kv.clone()
    kpool_compress_insert(actual, k, gate, ape, slot_mapping, pool_size)
    assert torch.equal(actual, expected)


def test_strided_inputs():
    # K / gate as column slices of a wider activation (row stride != 128).
    k, gate, ape, slot_mapping, kv = _make_batch([333, 1024], 4, 32, torch.int64)
    wide = torch.randn(k.shape[0], 3 * HEAD_DIM, device="cuda").to(torch.bfloat16)
    wide[:, HEAD_DIM : 2 * HEAD_DIM] = k
    wide[:, 2 * HEAD_DIM :] = gate
    k_view, gate_view = wide[:, HEAD_DIM : 2 * HEAD_DIM], wide[:, 2 * HEAD_DIM :]
    expected = kv.clone()
    _unfused_insert(expected, k, gate, ape, slot_mapping, 4, True)
    actual = kv.clone()
    kpool_compress_insert(actual, k_view, gate_view, ape, slot_mapping, 4)
    assert torch.equal(actual, expected)


def test_extreme_scores():
    # Large gate logits (softmax near one-hot) and tiny K (absmax clamp).
    k, gate, ape, slot_mapping, kv = _make_batch([256, 64], 4, 32, torch.int64)
    gate = (gate.float() * 30).bfloat16()
    k[:64] = (k[:64].float() * 1e-6).bfloat16()
    expected = kv.clone()
    _unfused_insert(expected, k, gate, ape, slot_mapping, 4, True)
    actual = kv.clone()
    kpool_compress_insert(actual, k, gate, ape, slot_mapping, 4)
    assert torch.equal(actual, expected)


def test_empty_and_short():
    k, gate, ape, slot_mapping, kv = _make_batch([3], 4, 32, torch.int64)
    before = kv.clone()
    kpool_compress_insert(kv, k, gate, ape, slot_mapping, 4)
    kpool_compress_insert(kv, k[:0], gate[:0], ape, slot_mapping[:0], 4)
    assert torch.equal(kv, before)


# --- Mixed step: decode update + prefill insert + tail seed in one launch ---


def _mixed_step(lens, dec_reqs, next_n, pool_size=4, page_size=32, seed=0):
    """A mixed indexer step with disjoint tail blocks and cache slots.

    Decode requests own tail blocks [0, dec_reqs), prefill requests the next
    ones; every pool (decode or prefill) gets a distinct random cache slot.
    Prefill requests start at a block-aligned cached-prefix offset.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    spec = max(next_n - 1, 0)
    ring = (
        pool_size * 1
        << max(0, math.ceil((pool_size + spec) / pool_size) - 1).bit_length()
    )
    n_pre = sum(lens)
    n_pools = sum(length // pool_size for length in lens) + dec_reqs * next_n
    num_pages = math.ceil(n_pools / page_size) + 2
    locs = torch.randperm(num_pages * page_size, generator=g, device="cuda").tolist()
    kv = torch.randint(
        0,
        256,
        (num_pages, page_size, HEAD_DIM + 4),
        generator=g,
        dtype=torch.uint8,
        device="cuda",
    )
    n_blocks = dec_reqs + len(lens) + 1
    tail = torch.randn(
        n_blocks, 2, ring, HEAD_DIM, generator=g, device="cuda"
    ).bfloat16()
    ape = torch.randn(pool_size, HEAD_DIM, generator=g, device="cuda")

    # prefill slice
    slot, tslot = [], []
    for r, length in enumerate(lens):
        prefix = 256 * int(torch.randint(0, 64, (1,), generator=g, device="cuda"))
        blk = dec_reqs + r
        for t in range(length):
            p = prefix + t
            slot.append(locs.pop() if t % pool_size == pool_size - 1 else -1)
            tslot.append(blk * ring + p % ring)
    k = torch.randn(n_pre, HEAD_DIM, generator=g, device="cuda").bfloat16()
    gate = (torch.randn(n_pre, HEAD_DIM, generator=g, device="cuda") * 3).bfloat16()
    slot = torch.tensor(slot, dtype=torch.int64, device="cuda")
    tslot = torch.tensor(tslot, dtype=torch.int32, device="cuda")

    # decode requests
    ctx = torch.randint(pool_size, 100000, (dec_reqs,), generator=g, device="cuda")
    pos = (ctx[:, None] + torch.arange(next_n, device="cuda")[None]).to(torch.int32)
    blocks = torch.arange(dec_reqs, device="cuda", dtype=torch.int32)[:, None]
    dtail = (blocks * ring + pos % ring).to(torch.int32)
    dloc = torch.tensor(
        [locs.pop() for _ in range(dec_reqs * next_n)], dtype=torch.int32, device="cuda"
    ).view(dec_reqs, next_n)
    dslot = torch.where(pos % pool_size == pool_size - 1, dloc, -1).to(torch.int32)
    dkey = torch.randn(
        dec_reqs, next_n, HEAD_DIM, generator=g, device="cuda"
    ).bfloat16()
    dgate = torch.randn(
        dec_reqs, next_n, HEAD_DIM, generator=g, device="cuda"
    ).bfloat16()
    return dict(
        kv=kv,
        tail=tail,
        ape=ape,
        k=k,
        gate=gate,
        slot=slot,
        tslot=tslot,
        dtail=dtail,
        dkey=dkey,
        dgate=dgate,
        dslot=dslot,
        pos=pos,
        pool_size=pool_size,
    )


def _mixed_reference(c):
    """Production sequence: prefill op, tail seed, ordered decode writer."""
    kv, tail, ps = c["kv"].clone(), c["tail"].clone(), c["pool_size"]
    if c["k"].shape[0] > 0:
        _unfused_insert(kv, c["k"], c["gate"], c["ape"], c["slot"], ps, True)
        kpool_seed_tail_cache(tail, c["k"], c["gate"], c["tslot"], ps, HEAD_DIM)
    if c["dkey"].shape[0] > 0:
        kpool_decode_update_and_maybe_write_cache_batched(
            kv,
            tail,
            c["dtail"],
            c["dkey"],
            c["dgate"],
            c["ape"],
            c["dslot"],
            c["pos"],
            ps,
            HEAD_DIM,
            round_scale=True,
        )
    return kv, tail


def _mixed_fused(c, seed_tail=True):
    kv, tail = c["kv"].clone(), c["tail"].clone()
    kpool_mixed_write(
        kv,
        c["ape"],
        c["pool_size"],
        prefill_k=c["k"],
        prefill_gate=c["gate"],
        prefill_slot_mapping=c["slot"],
        prefill_tail_slot_mapping=c["tslot"] if seed_tail else None,
        tail_kv_cache=tail,
        decode_tail_slot_mapping=c["dtail"],
        decode_key=c["dkey"],
        decode_gate=c["dgate"],
        decode_slot_mapping=c["dslot"],
        decode_positions=c["pos"],
    )
    return kv, tail


@pytest.mark.parametrize(
    "lens",
    [[], [3], [8192], [1024] * 8, [1, 7, 130, 1000, 33, 4, 5], [257, 64, 3, 999]],
)
@pytest.mark.parametrize("dec", [(0, 1), (3, 1), (64, 2), (16, 4), (5, 8)])
def test_mixed_write_matches_production_sequence(lens, dec):
    c = _mixed_step(lens, *dec, seed=len(lens) + dec[0])
    ref_kv, ref_tail = _mixed_reference(c)
    kv, tail = _mixed_fused(c)
    assert torch.equal(kv, ref_kv)
    assert torch.equal(tail, ref_tail)


@pytest.mark.parametrize("pool_size", [2, 8])
def test_mixed_write_pool_sizes(pool_size):
    c = _mixed_step([100, 37, 256], 7, 3, pool_size=pool_size, page_size=16)
    ref_kv, ref_tail = _mixed_reference(c)
    kv, tail = _mixed_fused(c)
    assert torch.equal(kv, ref_kv)
    assert torch.equal(tail, ref_tail)


def test_mixed_write_without_seeding():
    c = _mixed_step([513, 64], 4, 2)
    ref_kv = c["kv"].clone()
    _unfused_insert(ref_kv, c["k"], c["gate"], c["ape"], c["slot"], 4, True)
    ref_tail = c["tail"].clone()
    kpool_decode_update_and_maybe_write_cache_batched(
        ref_kv,
        ref_tail,
        c["dtail"],
        c["dkey"],
        c["dgate"],
        c["ape"],
        c["dslot"],
        c["pos"],
        4,
        HEAD_DIM,
        round_scale=True,
    )
    kv, tail = _mixed_fused(c, seed_tail=False)
    assert torch.equal(kv, ref_kv)
    assert torch.equal(tail, ref_tail)

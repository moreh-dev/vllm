# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The HIP kpool cache writers of _rocm_C (csrc/rocm/kpool_compress.cu):
``kpool_compress_insert``, ``kpool_decode_update`` and ``kpool_mixed_write``.

Every case runs a pure-torch reference of the vLLM semantics (prefill insert,
``kpool_seed_tail_cache``, the ordered decode walk) and compares the whole
kv_cache (dequantized fp8 keys and fp32 scales) and the whole tail ring. The
reference rounds at the same two bf16 points as the kernel; the only sources
of difference are 1-ulp fp32 choices (exp, division, log2), so the dequantized
keys agree to one fp8 ulp. The production invariant is checked byte for byte:
the HIP writers must produce exactly the bytes of the Triton writers in
``vllm/models/glm5next/amd/ops/kpool_compress.py``.

gfx942 / gfx950 only (fp8 e4m3fn conversions).
"""

import math

import pytest
import torch

from vllm.platforms import current_platform

if current_platform.is_rocm():
    from vllm import _custom_ops as ops
    from vllm.models.glm5next.amd.ops.kpool_compress import (
        kpool_compress_and_write_cache,
        kpool_decode_update_and_maybe_write_cache_batched,
        kpool_seed_tail_cache,
    )
    from vllm.platforms.rocm import on_gfx942, on_gfx950

    _HAS_OP = hasattr(torch.ops, "_rocm_C") and hasattr(
        torch.ops._rocm_C, "kpool_mixed_write"
    )
    _SUPPORTED = _HAS_OP and (on_gfx942() or on_gfx950())
else:
    _SUPPORTED = False

pytestmark = pytest.mark.skipif(
    not _SUPPORTED, reason="needs the _rocm_C kpool_compress ops on gfx942 / gfx950"
)

DEVICE = "cuda"
HEAD_DIM = 128
FP8_MAX = 448.0
HAD_SCALE = 0.08838834764831845  # 1/sqrt(128), the fp32 constant of the kernels
FP8 = torch.float8_e4m3fn
TILE = 16


# --------------------------------------------------------------------------
# torch reference of the pool math and the cache layout
# --------------------------------------------------------------------------


def hadamard128(x: torch.Tensor) -> torch.Tensor:
    """Walsh-Hadamard, stages of increasing stride (same order as the kernels)."""
    n = x.shape[-1]
    h = 1
    while h < n:
        v = x.view(*x.shape[:-1], n // (2 * h), 2, h)
        a, b = v[..., 0, :], v[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2).reshape(*x.shape)
        h *= 2
    return x


def compress_pools(k_rows, g_rows, ape, round_scale=True):
    """k_rows, g_rows [n, pool, 128] bf16; ape [pool, 128] fp32 ->
    fp8 bytes [n, 128] uint8 and fp32 scale [n]."""
    u = g_rows.float() + ape.float()[None]
    m = u.amax(dim=1, keepdim=True)
    e = torch.exp(u - m)
    z = (e * k_rows.float()).sum(1) / e.sum(1)
    x = z.to(torch.bfloat16).float()
    y = (hadamard128(x) * HAD_SCALE).to(torch.bfloat16).float()
    a = y.abs().amax(1).clamp_min(1e-4)
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(a / FP8_MAX)))
    else:
        scale = a / FP8_MAX
    q = (y / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return q.view(torch.uint8), scale


def _k_offsets(tok, page):
    """Byte offsets of the 128 keys of token(s) ``tok`` inside a page."""
    d = torch.arange(HEAD_DIM, device=DEVICE)
    if page > 1:
        return (
            (tok // TILE)[..., None] * (TILE * HEAD_DIM)
            + (d // TILE)[None] * (TILE * TILE)
            + (tok % TILE)[..., None] * TILE
            + (d % TILE)[None]
        )
    return d[None].expand(*tok.shape, HEAD_DIM)


def write_cache(kv, loc, q_u8, scale):
    """Store fp8 rows and scales at paged slots ``loc`` (int64, all >= 0)."""
    nb, page, cs = kv.shape
    flat = kv.view(nb, page * cs)
    pg, tok = loc // page, loc % page
    flat[pg[:, None], _k_offsets(tok, page)] = q_u8
    soff = (
        page * HEAD_DIM + tok[:, None] * 4 + torch.arange(4, device=DEVICE)[None]
    )
    flat[pg[:, None], soff] = scale.contiguous().view(torch.uint8).view(-1, 4)


def read_cache(kv):
    """kv_cache -> dequantized keys [nb, page, 128] fp32 and scales [nb, page]."""
    nb, page, cs = kv.shape
    flat = kv.view(nb, page * cs)
    kbytes = flat[:, : page * HEAD_DIM]
    k = kbytes[:, _k_offsets(torch.arange(page, device=DEVICE), page)]
    scale = flat[:, page * HEAD_DIM :].contiguous().view(torch.float32).view(nb, page)
    return k.view(FP8).float() * scale[..., None], scale


def ref_insert(kv, k, gate, ape, slot, pool, round_scale=True):
    n = k.shape[0]
    t = torch.arange(n, device=DEVICE)
    closing = (slot >= 0) & (t >= pool - 1)
    tc = t[closing]
    if tc.numel() == 0:
        return
    idx = tc[:, None] - (pool - 1) + torch.arange(pool, device=DEVICE)[None]
    q, s = compress_pools(k[idx], gate[idx], ape, round_scale)
    write_cache(kv, slot[closing].long(), q, s)


def ref_seed(tail, k, gate, tail_slot, pool):
    """kpool_seed_tail_cache: a token is among the last ``pool`` tokens of its
    request when the token ``pool`` ahead has no tail slot or another block."""
    n = k.shape[0]
    ring = tail.shape[2]
    ts = tail_slot.long()
    ahead = torch.full_like(ts, -1)
    if n > pool:
        ahead[: n - pool] = ts[pool:]
    write = (ts >= 0) & ((ahead < 0) | (ahead // ring != ts // ring))
    idx = write.nonzero().flatten()
    tail[ts[idx] // ring, 0, ts[idx] % ring] = k[idx]
    tail[ts[idx] // ring, 1, ts[idx] % ring] = gate[idx]


def ref_decode(kv, tail, tail_slot, key, score, ape, slot_map, pos, pool, round_scale=True):
    """The ordered walk: per request, token by token, read ring then stash."""
    B, next_n = key.shape[:2]
    ring = tail.shape[2]
    ts_c, sm_c, pos_c = tail_slot.cpu(), slot_map.cpu(), pos.cpu()
    for b in range(B):
        for i in range(next_n):
            p, ts, loc = int(pos_c[b, i]), int(ts_c[b, i]), int(sm_c[b, i])
            if p >= 0 and loc >= 0 and p % pool == pool - 1:
                blk, slot = max(ts, 0) // ring, p % pool
                k_rows = torch.empty(pool, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
                g_rows = torch.empty_like(k_rows)
                for s in range(pool):
                    if s == slot:
                        k_rows[s], g_rows[s] = key[b, i], score[b, i]
                    else:
                        phys = (p - slot + s) % ring
                        k_rows[s], g_rows[s] = tail[blk, 0, phys], tail[blk, 1, phys]
                q, sc = compress_pools(k_rows[None], g_rows[None], ape, round_scale)
                write_cache(kv, torch.tensor([loc], device=DEVICE), q, sc)
            if p >= 0 and ts >= 0:
                tail[ts // ring, 0, p % ring] = key[b, i]
                tail[ts // ring, 1, p % ring] = score[b, i]


# --------------------------------------------------------------------------
# byte-exact reference: the Triton writers
# --------------------------------------------------------------------------


def triton_insert(kv, k, gate, ape, slot, pool, round_scale=True):
    n = slot.shape[0]
    if n < pool:
        return
    t = torch.arange(n, device=DEVICE)
    write_mask = (slot >= 0) & (t >= pool - 1)
    idx = (t - (pool - 1)).clamp_min(0)[:, None] + torch.arange(pool, device=DEVICE)[None]
    kpool_compress_and_write_cache(
        kv,
        k[idx],
        gate[idx],
        ape,
        slot.to(torch.int64),
        pool_size=pool,
        head_dim=HEAD_DIM,
        write_mask=write_mask,
        round_scale=round_scale,
        write_cache=True,
        return_compressed=False,
    )


def triton_decode(kv, tail, tail_slot, key, score, ape, slot_map, pos, pool, round_scale=True):
    # one request at a time: the batched launch is not deterministic at B >= 1024
    for b in range(key.shape[0]):
        kpool_decode_update_and_maybe_write_cache_batched(
            kv,
            tail,
            tail_slot[b : b + 1],
            key[b : b + 1],
            score[b : b + 1],
            ape,
            slot_map[b : b + 1],
            pos[b : b + 1],
            pool,
            HEAD_DIM,
            round_scale=round_scale,
        )


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def make_prefill(lens, pool, page, slot_dtype=torch.int64, seed=0, ring=None, nblock_offset=0):
    """Varlen prefill batch, pools aligned to each request start, random paged
    slots. Also returns the tail slot mapping for seeding."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    slots, tslots = [], []
    n_pools = sum(length // pool for length in lens)
    num_blocks = max(1, math.ceil(n_pools / page)) + 2
    perm = torch.randperm(num_blocks * page, device=DEVICE, generator=g)[:n_pools].tolist()
    for r, length in enumerate(lens):
        for t in range(length):
            slots.append(perm.pop() if t % pool == pool - 1 else -1)
            tslots.append((nblock_offset + r) * (ring or pool) + t % (ring or pool))
    n = len(slots)
    k = torch.randn(n, HEAD_DIM, generator=g, device=DEVICE).bfloat16()
    gate = (torch.randn(n, HEAD_DIM, generator=g, device=DEVICE) * 3).bfloat16()
    ape = torch.randn(pool, HEAD_DIM, generator=g, device=DEVICE)
    slot = torch.tensor(slots, dtype=slot_dtype, device=DEVICE)
    tslot = torch.tensor(tslots, dtype=slot_dtype, device=DEVICE)
    kv = torch.zeros(num_blocks, page, HEAD_DIM + 4, dtype=torch.uint8, device=DEVICE)
    return k, gate, ape, slot, tslot, kv


def make_decode(
    B, next_n, pool, page=64, seed=0, ring=None, nblocks=None, idx_dtype=torch.int32, tail_pad=1
):
    """B requests x next_n consecutive tokens from random context lengths.
    ``idx_dtype`` is the dtype of the three metadata tensors (vLLM keeps its
    slot mappings and positions in int64). ``tail_pad`` > 1 pads every tail
    page to ``tail_pad`` rings (vLLM pads the tail page to the KV-cache page,
    so stride(0) >> 2 * ring * 128 and the tail can exceed 2^31 elements): the
    tail is then a strided view and the requests are spread over all blocks."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    ring = ring or pool
    nblocks = nblocks or B + 2
    tail_base = torch.randn(
        nblocks, 2, ring * tail_pad, HEAD_DIM, generator=g, device=DEVICE
    ).bfloat16()
    tail = tail_base[:, :, :ring] if tail_pad > 1 else tail_base
    ctx = torch.randint(pool, 100000, (B,), generator=g, device=DEVICE)
    pos = (ctx[:, None] + torch.arange(next_n, device=DEVICE)[None]).to(torch.int32)
    spread = max(1, (nblocks - 1) // B) if tail_pad > 1 else 1
    blocks = (torch.arange(B, dtype=torch.int32, device=DEVICE) * spread)[:, None]
    tail_slot = (blocks * ring + pos % ring).to(torch.int32)
    n_pages = math.ceil(B * next_n / page) + 2
    locs = torch.randperm(n_pages * page, generator=g, device=DEVICE)[: B * next_n]
    locs = locs.view(B, next_n).to(torch.int32)
    completes = pos % pool == pool - 1
    slot_map = torch.where(completes, locs, torch.full_like(locs, -1))
    key = torch.randn(B, next_n, HEAD_DIM, generator=g, device=DEVICE).bfloat16()
    score = torch.randn(B, next_n, HEAD_DIM, generator=g, device=DEVICE).bfloat16()
    ape = torch.randn(pool, HEAD_DIM, generator=g, device=DEVICE)
    kv = torch.zeros(n_pages, page, HEAD_DIM + 4, dtype=torch.uint8, device=DEVICE)
    return dict(
        kv=kv,
        tail=tail,
        tail_base=tail_base,
        tail_slot=tail_slot.to(idx_dtype),
        key=key,
        score=score,
        ape=ape,
        slot_map=slot_map.to(idx_dtype),
        pos=pos.to(idx_dtype),
        pool=pool,
    )


def tail_clone(d):
    """Copy of the tail that keeps a padded tail's strides (clone() would compact the view)."""
    base = d["tail_base"].clone()
    return base[:, :, : d["tail"].shape[2]] if d["tail"].shape[2] != base.shape[2] else base


def assert_cache_close(kv_ref, kv_got):
    y_ref, s_ref = read_cache(kv_ref)
    y_got, s_got = read_cache(kv_got)
    torch.testing.assert_close(y_got, y_ref, rtol=0.13, atol=2e-3)
    torch.testing.assert_close(s_got, s_ref, rtol=1e-6, atol=0)


def _free_gib():
    free, _ = torch.cuda.mem_get_info()
    return free / 2**30


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lens, pool, page, slot_dtype, lanes, round_scale",
    [
        *[([1024], pool, page, torch.int64, 0, True) for pool in (4, 2, 3, 8, 16) for page in (64, 1)],
        *[([2048], 4, 64, torch.int64, lanes, True) for lanes in (16, 32, 64)],
        ([1, 7, 130, 1000, 33, 4, 5], 4, 32, torch.int32, 0, True),
        ([4099, 17, 2048, 3], 3, 32, torch.int64, 0, True),
        ([300, 77, 2048], 4, 64, torch.int64, 0, False),
        ([8192] * 8, 4, 64, torch.int64, 0, True),
        ([131072] * 2, 4, 64, torch.int64, 0, True),
    ],
)
@torch.inference_mode()
def test_insert(lens, pool, page, slot_dtype, lanes, round_scale):
    k, gate, ape, slot, _, kv = make_prefill(lens, pool, page, slot_dtype)
    ref = kv.clone()
    ref_insert(ref, k, gate, ape, slot, pool, round_scale)
    got = kv.clone()
    ops.kpool_compress_insert(got, k, gate, ape, slot, pool, round_scale, lanes)
    torch.cuda.synchronize()
    assert_cache_close(ref, got)
    exact = kv.clone()
    triton_insert(exact, k, gate, ape, slot, pool, round_scale)
    assert torch.equal(exact, got), "HIP insert differs from the Triton writer"


@pytest.mark.parametrize(
    "B, next_n, pool, ring, idx_dtype, tail_pad, nblocks",
    [
        # ring == pool: the generic path
        *[(B, n, pool, None, torch.int32, 1, None)
          for B, n, pool in ((4, 1, 4), (3, 4, 4), (8, 2, 4), (5, 8, 4), (2, 4, 16), (16, 3, 8))],
        # production rings (pool * pow2(ceil((pool + next_n - 1) / pool)))
        *[(B, n, 4, ring, torch.int32, 1, None)
          for B, n, ring in ((8, 2, 8), (6, 3, 8), (5, 4, 8), (4, 5, 8), (4, 6, 16), (4, 7, 16), (5, 8, 16), (3, 16, 32))],
        # int64 metadata (vLLM): same kernels with an 8 B element stride
        *[(B, n, 4, ring, torch.int64, 1, None)
          for B, n, ring in ((4, 1, 4), (5, 4, 8), (3, 2, 4), (5, 8, 16))],
        (64, 4, 4, None, torch.int32, 1, None),
        (32, 16, 4, None, torch.int32, 1, None),
        (64, 1, 4, None, torch.int64, 1, None),
        # vLLM layout: tail pages padded to the KV page (strided view)
        (64, 1, 4, None, torch.int64, 16, None),
        (5, 4, 4, 8, torch.int64, 16, None),
        # ... with block offsets past 2^31 elements (8.6 GB tail, stride0 2^20)
        (512, 1, 4, None, torch.int64, 1024, 4096),
        (256, 2, 4, 8, torch.int64, 512, 4096),
    ],
)
@torch.inference_mode()
def test_decode(B, next_n, pool, ring, idx_dtype, tail_pad, nblocks):
    need_gib = 2 * (nblocks or B + 2) * 2 * (ring or pool) * tail_pad * HEAD_DIM * 2 / 2**30
    if need_gib > _free_gib() - 4:
        pytest.skip(f"needs ~{need_gib:.0f} GiB of GPU memory")
    d = make_decode(B, next_n, pool, 64, seed=B * 100 + next_n, ring=ring, idx_dtype=idx_dtype,
                    tail_pad=tail_pad, nblocks=nblocks)
    kv_ref, tail_ref = d["kv"].clone(), tail_clone(d)
    ref_decode(kv_ref, tail_ref, d["tail_slot"], d["key"], d["score"], d["ape"],
               d["slot_map"], d["pos"], pool)
    # one call: with next_n > 1 the update is not idempotent (a second run
    # reads ring slots the first run's stashes overwrote)
    kv_got, tail_got = d["kv"].clone(), tail_clone(d)
    ops.kpool_decode_update(kv_got, tail_got, d["tail_slot"], d["key"], d["score"],
                            d["ape"], d["slot_map"], d["pos"], pool, True)
    torch.cuda.synchronize()
    assert_cache_close(kv_ref, kv_got)
    assert torch.equal(tail_ref, tail_got), "tail ring mismatch"
    if d["tail"].shape[2] == pool:
        # the Triton walk asserts ring == pool
        kv_e, tail_e = d["kv"].clone(), tail_clone(d)
        triton_decode(kv_e, tail_e, d["tail_slot"], d["key"], d["score"], d["ape"],
                      d["slot_map"], d["pos"], pool)
        assert torch.equal(kv_e, kv_got) and torch.equal(tail_e, tail_got), \
            "HIP decode differs from the Triton writer"


@pytest.mark.parametrize(
    "lens, B, next_n, idx_dtype, tail_pad",
    [
        ([64, 9], 1, 1, torch.int64, 1),
        ([4096, 17, 300], 8, 4, torch.int64, 1),
        ([64, 9], 1, 1, torch.int32, 1),
        ([300, 17], 8, 4, torch.int32, 1),
        ([2048] * 4, 32, 2, torch.int64, 1),
        # vLLM layout: padded tail pages (strided view) for the seed + decode roles
        ([4096, 17, 300], 8, 1, torch.int64, 16),
        ([300, 77], 256, 1, torch.int64, 512),
    ],
)
@torch.inference_mode()
def test_mixed(lens, B, next_n, idx_dtype, tail_pad):
    pool = 4
    ring = pool
    # one index dtype per launch: prefill slot_mapping, seed slots and decode metadata
    d = make_decode(B, next_n, pool, 64, seed=7, ring=ring, nblocks=B + len(lens) + 2,
                    idx_dtype=idx_dtype, tail_pad=tail_pad)
    k, gate, ape, slot, tslot, kv_p = make_prefill(lens, pool, 64, idx_dtype, seed=11,
                                                    ring=ring, nblock_offset=B)
    # prefill cache slots live in their own pages, after the decode pages
    slot = torch.where(slot >= 0, slot + d["kv"].shape[0] * 64, slot)
    kv = torch.cat([d["kv"], kv_p], 0)
    ape = d["ape"]
    kv_ref, tail_ref = kv.clone(), tail_clone(d)
    ref_insert(kv_ref, k, gate, ape, slot, pool)
    ref_seed(tail_ref, k, gate, tslot, pool)
    ref_decode(kv_ref, tail_ref, d["tail_slot"], d["key"], d["score"], ape,
               d["slot_map"], d["pos"], pool)
    kv_got, tail_got = kv.clone(), tail_clone(d)
    ops.kpool_mixed_write(kv_got, ape, pool, True, 0, k, gate, slot, tslot, tail_got,
                          d["tail_slot"], d["key"], d["score"], d["slot_map"], d["pos"])
    torch.cuda.synchronize()
    assert_cache_close(kv_ref, kv_got)
    assert torch.equal(tail_ref, tail_got), "tail ring mismatch"
    # byte for byte against the three Triton writers, in the production order
    kv_e, tail_e = kv.clone(), tail_clone(d)
    triton_insert(kv_e, k, gate, ape, slot, pool)
    kpool_seed_tail_cache(tail_e, k, gate, tslot, pool, HEAD_DIM)
    triton_decode(kv_e, tail_e, d["tail_slot"], d["key"], d["score"], ape,
                  d["slot_map"], d["pos"], pool)
    assert torch.equal(kv_e, kv_got) and torch.equal(tail_e, tail_got), \
        "HIP mixed write differs from the Triton writers"


@torch.inference_mode()
def test_mixed_decode_only_and_empty():
    """Mixed entry with no prefill rows falls back to the decode update; with
    nothing to do it must not touch the cache."""
    pool = 4
    d = make_decode(6, 2, pool, 64, seed=3, ring=8, idx_dtype=torch.int64)
    kv_ref, tail_ref = d["kv"].clone(), tail_clone(d)
    ops.kpool_decode_update(kv_ref, tail_ref, d["tail_slot"], d["key"], d["score"],
                            d["ape"], d["slot_map"], d["pos"], pool, True)
    kv_got, tail_got = d["kv"].clone(), tail_clone(d)
    empty_k = torch.empty(0, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    ops.kpool_mixed_write(kv_got, d["ape"], pool, True, 0, empty_k, empty_k,
                          torch.empty(0, dtype=torch.int64, device=DEVICE), None, tail_got,
                          d["tail_slot"], d["key"], d["score"], d["slot_map"], d["pos"])
    torch.cuda.synchronize()
    assert torch.equal(kv_ref, kv_got) and torch.equal(tail_ref, tail_got)
    kv0 = d["kv"].clone()
    ops.kpool_mixed_write(kv0, d["ape"], pool)
    torch.cuda.synchronize()
    assert torch.equal(kv0, d["kv"])

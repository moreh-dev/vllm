# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.priority_eviction_queue import (
    PriorityEvictionQueue,
    RetentionMeta,
)


def _make_block(block_id: int) -> KVCacheBlock:
    return KVCacheBlock(block_id=block_id)


def _set_meta(
    queue: PriorityEvictionQueue,
    block: KVCacheBlock,
    *,
    priority: int,
    expiry: float | None = None,
    scope: str | None = None,
    last_freed: float = 0.0,
) -> None:
    """Test helper: install a sidecar entry directly without going through
    apply_directives. Reaches into the queue's private dict — acceptable
    in tests of the queue itself."""
    queue._meta[block.block_id] = RetentionMeta(
        priority=priority,
        expiry=expiry,
        scope=scope,
        last_freed_time=last_freed,
    )


class TestPriorityEvictionQueue:
    def test_empty_queue(self):
        queue = PriorityEvictionQueue()
        assert queue.num_blocks == 0
        assert queue.pop_lowest() is None

    def test_insert_and_pop_single(self):
        queue = PriorityEvictionQueue()
        block = _make_block(1)
        _set_meta(queue, block, priority=50)
        assert queue.try_insert(block) is True
        assert queue.num_blocks == 1
        popped = queue.pop_lowest()
        assert popped is block
        assert queue.num_blocks == 0

    def test_try_insert_returns_false_for_unprioritized_block(self):
        queue = PriorityEvictionQueue()
        block = _make_block(1)
        # No _set_meta call — sidecar entry absent.
        assert queue.try_insert(block) is False
        assert queue.num_blocks == 0

    def test_eviction_order_by_priority(self):
        queue = PriorityEvictionQueue()
        blocks = [_make_block(i) for i in range(3)]
        # Insert in non-ascending order to verify the heap reorders.
        for blk, p in zip(blocks, [80, 20, 50]):
            _set_meta(queue, blk, priority=p)
            queue.try_insert(blk)
        # Lowest priority must come out first.
        assert queue.pop_lowest() is blocks[1]  # priority 20
        assert queue.pop_lowest() is blocks[2]  # priority 50
        assert queue.pop_lowest() is blocks[0]  # priority 80

    def test_eviction_order_tiebreak_by_time(self):
        queue = PriorityEvictionQueue()
        blocks = [_make_block(i) for i in range(3)]
        # Same priority; differ only in last_freed_time.
        for blk, t in zip(blocks, [300.0, 100.0, 200.0]):
            _set_meta(queue, blk, priority=50, last_freed=t)
            queue.try_insert(blk)
        # Oldest-freed evicted first.
        assert queue.pop_lowest() is blocks[1]  # t=100
        assert queue.pop_lowest() is blocks[2]  # t=200
        assert queue.pop_lowest() is blocks[0]  # t=300

    def test_remove_keeps_sidecar(self):
        queue = PriorityEvictionQueue()
        block = _make_block(1)
        _set_meta(queue, block, priority=50)
        queue.try_insert(block)
        queue.remove(block)
        assert queue.num_blocks == 0
        # Sidecar entry survives so that priority returns if the block
        # is freed again later.
        assert block.block_id in queue._meta
        assert queue.try_insert(block) is True
        assert queue.num_blocks == 1

    def test_remove_nonexistent_is_noop(self):
        queue = PriorityEvictionQueue()
        block = _make_block(1)
        # No insert; remove must not raise.
        queue.remove(block)
        assert queue.num_blocks == 0

    def test_stale_heap_entries_are_skipped_in_pop_lowest(self):
        queue = PriorityEvictionQueue()
        # Insert two blocks; remove one (leaving a stale heap entry).
        block_a = _make_block(1)
        block_b = _make_block(2)
        _set_meta(queue, block_a, priority=10)
        _set_meta(queue, block_b, priority=50)
        queue.try_insert(block_a)
        queue.try_insert(block_b)
        queue.remove(block_a)  # block_a now stale in heap
        # pop_lowest must skip the stale entry and return block_b.
        assert queue.pop_lowest() is block_b
        assert queue.pop_lowest() is None

    def test_ttl_not_expired(self, monkeypatch):
        import time as time_mod

        queue = PriorityEvictionQueue()
        block = _make_block(1)
        # Set "now" to 100; expiry is at 200 (not yet reached).
        monkeypatch.setattr(time_mod, "monotonic", lambda: 100.0)
        _set_meta(queue, block, priority=50, expiry=200.0)
        queue.try_insert(block)
        # Even when "now" advances to 150, expiry (200) is in the future.
        monkeypatch.setattr(time_mod, "monotonic", lambda: 150.0)
        assert queue.pop_lowest() is block

    def test_ttl_expiry(self, monkeypatch):
        import time as time_mod

        queue = PriorityEvictionQueue()
        block = _make_block(1)
        # Insert with expiry=200.
        monkeypatch.setattr(time_mod, "monotonic", lambda: 100.0)
        _set_meta(queue, block, priority=50, expiry=200.0)
        queue.try_insert(block)
        # Advance past expiry — block is treated as unprioritized.
        monkeypatch.setattr(time_mod, "monotonic", lambda: 250.0)
        # pop_lowest discards expired entries and returns None when none
        # remain.
        assert queue.pop_lowest() is None
        assert queue.num_blocks == 0


class TestApplyDirectives:
    def _peek_meta(self, queue: PriorityEvictionQueue, block_id: int):
        return queue._meta.get(block_id)

    def test_apply_retention_to_block_single_match(self):
        queue = PriorityEvictionQueue()
        block = _make_block(0)  # tokens 0..15 (block_size=16)
        directives = [{"start": 0, "end": 16, "priority": 50}]
        queue.apply_directives([block], directives, scope=None, block_size=16)
        meta = self._peek_meta(queue, 0)
        assert meta is not None
        assert meta.priority == 50

    def test_apply_retention_no_match(self):
        queue = PriorityEvictionQueue()
        block = _make_block(0)  # tokens 0..15
        # Directive covers tokens 100..200 — no overlap.
        directives = [{"start": 100, "end": 200, "priority": 50}]
        queue.apply_directives([block], directives, scope=None, block_size=16)
        assert self._peek_meta(queue, 0) is None

    def test_apply_retention_highest_priority_wins(self):
        queue = PriorityEvictionQueue()
        block = _make_block(0)  # tokens 0..15
        directives = [
            {"start": 0, "end": 16, "priority": 30},
            {"start": 0, "end": 16, "priority": 80},  # higher wins
            {"start": 0, "end": 16, "priority": 50},
        ]
        queue.apply_directives([block], directives, scope=None, block_size=16)
        assert self._peek_meta(queue, 0).priority == 80

    def test_apply_retention_open_ended_range(self):
        queue = PriorityEvictionQueue()
        blocks = [_make_block(i) for i in range(3)]  # tokens 0..15, 16..31, 32..47
        # end=None means "from start to end of sequence".
        directives = [{"start": 16, "end": None, "priority": 70}]
        queue.apply_directives(blocks, directives, scope=None, block_size=16)
        assert self._peek_meta(queue, 0) is None  # tokens 0..15 not covered
        assert self._peek_meta(queue, 1).priority == 70
        assert self._peek_meta(queue, 2).priority == 70

    def test_apply_retention_with_duration(self, monkeypatch):
        import time as time_mod

        monkeypatch.setattr(time_mod, "monotonic", lambda: 1000.0)
        queue = PriorityEvictionQueue()
        block = _make_block(0)
        directives = [{"start": 0, "end": 16, "priority": 50, "duration": 60.0}]
        queue.apply_directives([block], directives, scope=None, block_size=16)
        meta = self._peek_meta(queue, 0)
        assert meta.expiry == 1060.0

    def test_escalation_from_different_scope(self):
        queue = PriorityEvictionQueue()
        block = _make_block(0)
        _set_meta(queue, block, priority=30, scope="alice")
        # bob escalates to 80 — allowed regardless of scope.
        queue.apply_directives(
            [block],
            [{"start": 0, "end": 16, "priority": 80}],
            scope="bob",
            block_size=16,
        )
        meta = self._peek_meta(queue, 0)
        assert meta.priority == 80
        assert meta.scope == "bob"

    def test_downgrade_blocked_from_different_scope(self):
        queue = PriorityEvictionQueue()
        block = _make_block(0)
        _set_meta(queue, block, priority=80, scope="alice")
        # bob tries to downgrade to 20 — denied (alice owns the block).
        queue.apply_directives(
            [block],
            [{"start": 0, "end": 16, "priority": 20}],
            scope="bob",
            block_size=16,
        )
        meta = self._peek_meta(queue, 0)
        assert meta.priority == 80
        assert meta.scope == "alice"

    def test_owner_can_downgrade(self):
        queue = PriorityEvictionQueue()
        block = _make_block(0)
        _set_meta(queue, block, priority=80, scope="alice")
        queue.apply_directives(
            [block],
            [{"start": 0, "end": 16, "priority": 20}],
            scope="alice",
            block_size=16,
        )
        meta = self._peek_meta(queue, 0)
        assert meta.priority == 20
        assert meta.scope == "alice"

    def test_owner_clear_on_no_match(self):
        queue = PriorityEvictionQueue()
        block = _make_block(0)
        _set_meta(queue, block, priority=50, scope="alice")
        # alice issues directives that don't cover this block — sidecar
        # entry is cleared.
        queue.apply_directives(
            [block],
            [{"start": 100, "end": 200, "priority": 90}],
            scope="alice",
            block_size=16,
        )
        assert self._peek_meta(queue, 0) is None

    def test_non_owner_no_clear_on_no_match(self):
        queue = PriorityEvictionQueue()
        block = _make_block(0)
        _set_meta(queue, block, priority=50, scope="alice")
        # bob's directives don't cover this block — alice's entry stays.
        queue.apply_directives(
            [block],
            [{"start": 100, "end": 200, "priority": 90}],
            scope="bob",
            block_size=16,
        )
        meta = self._peek_meta(queue, 0)
        assert meta is not None
        assert meta.scope == "alice"

    def test_no_scope_no_clear(self):
        queue = PriorityEvictionQueue()
        block = _make_block(0)
        _set_meta(queue, block, priority=50, scope="alice")
        # scope=None caller is anonymous — must not clear anyone's entry.
        queue.apply_directives(
            [block],
            [{"start": 100, "end": 200, "priority": 90}],
            scope=None,
            block_size=16,
        )
        assert self._peek_meta(queue, 0) is not None


class TestSidecarLifecycle:
    def test_sidecar_entry_cleared_on_pop_lowest(self):
        queue = PriorityEvictionQueue()
        block = _make_block(0)
        _set_meta(queue, block, priority=50)
        queue.try_insert(block)
        queue.pop_lowest()
        assert 0 not in queue._meta

    def test_sidecar_entry_cleared_on_clear_priority(self):
        queue = PriorityEvictionQueue()
        block = _make_block(0)
        _set_meta(queue, block, priority=50)
        queue.try_insert(block)
        queue.clear_priority(0)
        assert 0 not in queue._meta
        assert 0 not in queue._in_queue

    def test_sidecar_persists_through_reuse_cycle(self):
        queue = PriorityEvictionQueue()
        block = _make_block(0)
        _set_meta(queue, block, priority=50)
        queue.try_insert(block)
        queue.remove(block)  # block reused via touch
        # Sidecar entry is preserved so try_insert succeeds again on next free.
        assert queue.try_insert(block) is True
        # And the heap entry reflects the same priority.
        popped = queue.pop_lowest()
        assert popped is block

    def test_sidecar_cleared_on_clear_all(self):
        queue = PriorityEvictionQueue()
        for i in range(3):
            block = _make_block(i)
            _set_meta(queue, block, priority=50)
            queue.try_insert(block)
        queue.clear()
        assert len(queue._meta) == 0
        assert len(queue._in_queue) == 0
        assert queue.num_blocks == 0

    def test_clear_priority_for_unknown_block_is_noop(self):
        queue = PriorityEvictionQueue()
        queue.clear_priority(999)  # not present — must not raise

    def test_contains_reflects_heap_membership(self):
        queue = PriorityEvictionQueue()
        block = _make_block(0)
        _set_meta(queue, block, priority=50)
        assert block not in queue
        queue.try_insert(block)
        assert block in queue
        queue.remove(block)
        assert block not in queue


class TestBlockPoolPriorityEviction:
    def _make_pool(self, num_blocks=8, block_size=16):
        from vllm.v1.core.block_pool import BlockPool

        return BlockPool(
            num_gpu_blocks=num_blocks,
            enable_caching=True,
            hash_block_size=block_size,
            enable_kv_cache_events=False,
        )

    def test_pool_has_priority_eviction_queue(self):
        pool = self._make_pool()
        assert isinstance(pool.priority_eviction_queue, PriorityEvictionQueue)
        assert pool.priority_eviction_queue.num_blocks == 0

    def test_reset_prefix_cache_clears_priority_queue(self):
        pool = self._make_pool()
        # Seed the queue with one prioritized free block. Remove from the
        # LRU first so the block lives in exactly one queue, matching the
        # invariant that free_blocks (Task 13) will enforce.
        block = pool.blocks[1]
        pool.free_block_queue.remove(block)
        _set_meta(pool.priority_eviction_queue, block, priority=50)
        pool.priority_eviction_queue.try_insert(block)
        assert pool.priority_eviction_queue.num_blocks == 1
        pool.reset_prefix_cache()
        assert pool.priority_eviction_queue.num_blocks == 0

    def test_cache_full_blocks_routes_directives_to_queue(self):
        from vllm.sampling_params import SamplingParams

        pool = self._make_pool(num_blocks=8, block_size=16)
        sampling = SamplingParams(
            extra_args={
                "retention_directives": [
                    {"start": 0, "end": 16, "priority": 80},
                ],
                "retention_scope": "alice",
            }
        )

        # Minimal stub request — only what the hook needs.
        class _Req:
            sampling_params: SamplingParams

        request = _Req()
        request.sampling_params = sampling

        blocks = [pool.blocks[1]]
        # Drive the hook directly to isolate its behavior from the rest of
        # cache_full_blocks.
        pool._apply_retention_hook(request, blocks, num_full_blocks=1, block_size=16)

        meta = pool.priority_eviction_queue._meta.get(blocks[0].block_id)
        assert meta is not None
        assert meta.priority == 80
        assert meta.scope == "alice"

    def test_no_extra_args_zero_overhead_path(self):
        from vllm.sampling_params import SamplingParams

        pool = self._make_pool()
        sampling = SamplingParams()  # no extra_args

        class _Req:
            sampling_params: SamplingParams

        request = _Req()
        request.sampling_params = sampling
        blocks = [pool.blocks[1]]
        pool._apply_retention_hook(request, blocks, num_full_blocks=1, block_size=16)
        assert pool.priority_eviction_queue.num_blocks == 0
        assert blocks[0].block_id not in pool.priority_eviction_queue._meta

    def test_eviction_drains_lru_before_priority(self):
        pool = self._make_pool(num_blocks=8, block_size=16)
        # Mark 3 blocks as prioritized, remove them from LRU first so they
        # live exclusively in the priority queue (avoids double-allocation
        # before Task 13 integrates the free-path).
        prioritized_ids = [1, 2, 3]
        for bid in prioritized_ids:
            block = pool.blocks[bid]
            pool.free_block_queue.remove(block)
            _set_meta(pool.priority_eviction_queue, block, priority=50)
            pool.priority_eviction_queue.try_insert(block)
        # After removal from LRU, count the remaining LRU-only free blocks.
        free_lru_before = pool.free_block_queue.num_free_blocks
        # Allocate up to free_lru_before + 1 blocks — the +1 must come from
        # the priority queue.
        ret = pool.get_new_blocks(free_lru_before + 1)
        assert len(ret) == free_lru_before + 1
        # The last block returned should be the one we marked prioritized
        # (since the LRU drained first).
        assert ret[-1].block_id in prioritized_ids

    def test_get_num_free_blocks_sums_both(self):
        pool = self._make_pool(num_blocks=8)
        free_before = pool.get_num_free_blocks()
        block = pool.blocks[1]
        _set_meta(pool.priority_eviction_queue, block, priority=50)
        pool.priority_eviction_queue.try_insert(block)
        # The block is "in" both the LRU and the priority queue at this
        # point — but the LRU count remains the same; the priority count
        # adds.
        assert pool.get_num_free_blocks() == free_before + 1

    def test_touch_removes_from_priority_queue(self):
        pool = self._make_pool()
        block = pool.blocks[1]
        block.ref_cnt = 0  # free state
        # Remove from LRU first so the block lives in exactly one queue,
        # consistent with the Task-13 invariant.
        pool.free_block_queue.remove(block)
        _set_meta(pool.priority_eviction_queue, block, priority=50)
        pool.priority_eviction_queue.try_insert(block)
        assert pool.priority_eviction_queue.num_blocks == 1
        pool.touch([block])
        # touch() reuses the block: it must leave the priority queue.
        assert pool.priority_eviction_queue.num_blocks == 0
        assert block.ref_cnt == 1

    def test_touch_removes_from_lru(self):
        pool = self._make_pool()
        # Use a block already in LRU (not prioritized).
        block = pool.blocks[2]
        block.ref_cnt = 0
        # Block is in free_block_queue by default after init.
        pool.touch([block])
        assert block.ref_cnt == 1

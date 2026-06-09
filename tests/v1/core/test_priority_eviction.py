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

    def test_try_insert_below_threshold_drops_sidecar_routes_to_lru(self, monkeypatch):
        """A sidecar entry whose priority is below the configured
        threshold must be rejected: try_insert returns False and the
        sidecar is dropped so the caller (free_blocks) routes the block
        to the LRU free list. The threshold guarantees enough cached
        blocks stay in LRU to absorb get_new_blocks demand without
        draining the priority queue."""
        import vllm.v1.core.priority_eviction_queue as pq_mod

        monkeypatch.setattr(pq_mod, "_PRIORITY_THRESHOLD", 60)
        queue = PriorityEvictionQueue()
        block = _make_block(1)
        _set_meta(queue, block, priority=50)

        assert queue.try_insert(block) is False, (
            "priority below threshold must be rejected so the block "
            "goes to LRU; the threshold is the whole point of Fix 4.0."
        )
        assert queue.num_blocks == 0
        assert block.block_id not in queue._meta, (
            "rejected entry must drop its sidecar so a later "
            "apply_directives starts from a clean slate."
        )

    def test_try_insert_at_or_above_threshold_enters_queue(self, monkeypatch):
        """At-or-above the threshold the regular insertion path runs:
        sidecar stays, the block lands in _in_queue, num_blocks goes up."""
        import vllm.v1.core.priority_eviction_queue as pq_mod

        monkeypatch.setattr(pq_mod, "_PRIORITY_THRESHOLD", 60)
        queue = PriorityEvictionQueue()
        block = _make_block(1)
        _set_meta(queue, block, priority=70)

        assert queue.try_insert(block) is True, (
            "priority at/above threshold must enter the queue normally."
        )
        assert queue.num_blocks == 1
        assert block.block_id in queue._meta

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

    def test_reinsert_after_remove_orders_by_current_priority(self):
        """Regression (priority inversion): a block removed (touch/reuse) and
        re-inserted with an ESCALATED priority must be ordered by the current
        priority, not the stale heap tuple left behind by the lazy remove().

        Sequence: A enters at 50 -> remove() (lazy delete leaves a (50,A)
        tuple in the heap) -> A escalated to 90 -> re-inserted (pushes a
        (90,A) tuple; both A tuples now satisfy the block_id-in-_in_queue
        test). A live priority-70 block B must evict BEFORE A. The buggy
        code pops the stale (50,A) tuple first and evicts the
        escalated-to-90 block ahead of the 70 block."""
        queue = PriorityEvictionQueue()
        a = _make_block(1)
        b = _make_block(2)
        _set_meta(queue, a, priority=50)
        queue.try_insert(a)
        queue.remove(a)  # touch: lazy delete, stale (50,A) tuple stays
        _set_meta(queue, a, priority=90)  # escalation via apply_directives
        queue.try_insert(a)  # re-freed: pushes (90,A); (50,A) still in heap
        _set_meta(queue, b, priority=70)
        queue.try_insert(b)
        assert queue.pop_lowest() is b, (
            "priority-70 block must evict before the escalated-to-90 block; "
            "a stale heap tuple must not order eviction by the old priority."
        )
        assert queue.pop_lowest() is a
        assert queue.pop_lowest() is None

    def test_reinsert_after_remove_orders_by_current_time(self):
        """Regression (recency tiebreak): same priority, but a block removed
        and re-freed with a NEWER last_freed_time must be ordered by the new
        time. The stale tuple carries the OLD (smaller) time and would
        otherwise make the block look older-freed than it is, evicting it
        before a genuinely older block."""
        queue = PriorityEvictionQueue()
        a = _make_block(1)
        b = _make_block(2)
        _set_meta(queue, a, priority=50, last_freed=100.0)
        queue.try_insert(a, last_freed_time=100.0)
        queue.remove(a)  # stale (t=100,A) left in heap
        queue.try_insert(a, last_freed_time=300.0)  # re-freed later; A now newest
        _set_meta(queue, b, priority=50, last_freed=200.0)
        queue.try_insert(b, last_freed_time=200.0)
        # A was last freed at 300 (newer than B's 200) -> B evicts first.
        assert queue.pop_lowest() is b, (
            "block re-freed at t=300 must evict after the t=200 block; "
            "a stale t=100 tuple must not defeat the recency tiebreak."
        )
        assert queue.pop_lowest() is a
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

    def test_drain_expired_then_pop_lowest_workflow(self, monkeypatch):
        """The full eviction workflow with TTL: drain_expired first
        removes expired entries from the queue (caller demotes them to
        LRU). pop_lowest then sees only live entries.

        Expired entries no longer leak into pop_lowest — that was the
        limbo-fix-era band-aid and is superseded by drain_expired.
        """
        import time as time_mod

        queue = PriorityEvictionQueue()
        block = _make_block(1)
        # Insert with expiry=200.
        monkeypatch.setattr(time_mod, "monotonic", lambda: 100.0)
        _set_meta(queue, block, priority=50, expiry=200.0)
        queue.try_insert(block)

        # Advance past expiry.
        monkeypatch.setattr(time_mod, "monotonic", lambda: 250.0)

        # New semantic: drain_expired returns the block_id, queue is
        # empty afterwards. The caller (BlockPool.get_new_blocks) is
        # responsible for routing the corresponding block into the LRU.
        drained = queue.drain_expired()
        assert drained == [block.block_id]
        assert queue.num_blocks == 0
        # pop_lowest now sees an empty queue.
        assert queue.pop_lowest() is None

    def test_drain_expired_returns_only_expired_block_ids(self, monkeypatch):
        """drain_expired returns block_ids whose sidecar.expiry has passed,
        and removes those entries from _in_queue + _meta. Non-expired
        entries stay in the queue."""
        import time as time_mod

        queue = PriorityEvictionQueue()
        b_exp = _make_block(1)
        b_live = _make_block(2)
        b_no_exp = _make_block(3)

        monkeypatch.setattr(time_mod, "monotonic", lambda: 100.0)
        _set_meta(queue, b_exp, priority=50, expiry=150.0)  # will expire
        _set_meta(queue, b_live, priority=70, expiry=250.0)  # still alive
        _set_meta(queue, b_no_exp, priority=90, expiry=None)  # no TTL
        queue.try_insert(b_exp)
        queue.try_insert(b_live)
        queue.try_insert(b_no_exp)

        # Advance time past b_exp's expiry only.
        monkeypatch.setattr(time_mod, "monotonic", lambda: 200.0)
        drained = queue.drain_expired()

        assert drained == [b_exp.block_id]
        assert b_exp.block_id not in queue._in_queue
        assert b_exp.block_id not in queue._meta
        assert b_live.block_id in queue._in_queue
        assert b_no_exp.block_id in queue._in_queue
        assert queue.num_blocks == 2

    def test_drain_expired_empty_queue_is_noop(self):
        """drain_expired on an empty queue returns an empty list."""
        queue = PriorityEvictionQueue()
        assert queue.drain_expired() == []
        assert queue.num_blocks == 0

    def test_try_insert_expired_meta_routes_to_lru(self, monkeypatch):
        """try_insert must return False when the sidecar entry is already
        expired so the caller (free_blocks) routes the block to the LRU
        free list instead of the priority queue. Otherwise the block would
        live in the priority queue forever, or land in limbo on the next
        pop_lowest."""
        import time as time_mod

        queue = PriorityEvictionQueue()
        block = _make_block(1)
        monkeypatch.setattr(time_mod, "monotonic", lambda: 100.0)
        _set_meta(queue, block, priority=50, expiry=150.0)
        # Advance past expiry BEFORE try_insert.
        monkeypatch.setattr(time_mod, "monotonic", lambda: 200.0)
        assert queue.try_insert(block, last_freed_time=200.0) is False
        # The expired sidecar must be cleaned up — otherwise a later
        # apply_directives could re-prime the same block back into the
        # priority queue.
        assert block.block_id not in queue._meta
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

    def test_free_unprioritized_goes_to_lru(self):
        pool = self._make_pool()
        block = pool.blocks[1]
        block.ref_cnt = 1
        # Block must not be in LRU while ref_cnt > 0 — remove it first to
        # mirror the production state of an active block.
        pool.free_block_queue.remove(block)
        free_before = pool.free_block_queue.num_free_blocks
        pool.free_blocks([block])
        assert pool.free_block_queue.num_free_blocks == free_before + 1
        assert pool.priority_eviction_queue.num_blocks == 0

    def test_free_prioritized_goes_to_priority_queue(self, monkeypatch):
        import time as time_mod

        monkeypatch.setattr(time_mod, "monotonic", lambda: 12345.0)
        pool = self._make_pool()
        block = pool.blocks[1]
        block.ref_cnt = 1
        # Block must not be in LRU while ref_cnt > 0 — remove it first.
        pool.free_block_queue.remove(block)
        # Install a sidecar entry so try_insert recognizes the block as
        # prioritized.
        _set_meta(pool.priority_eviction_queue, block, priority=50)
        free_before = pool.free_block_queue.num_free_blocks
        pool.free_blocks([block])
        # Did NOT land in the LRU queue.
        assert pool.free_block_queue.num_free_blocks == free_before
        # Did land in the priority queue with updated last_freed_time.
        assert pool.priority_eviction_queue.num_blocks == 1
        assert (
            pool.priority_eviction_queue._meta[block.block_id].last_freed_time
            == 12345.0
        )

    def test_touch_on_limbo_block_does_not_raise(self):
        """A block in neither the priority queue nor the LRU free list
        must not crash touch(). This guards against the pre-fix scenario
        where pop_lowest silently dropped an expired entry, leaving the
        block in limbo and crashing the next prefix-cache hit."""
        pool = self._make_pool()
        block = pool.blocks[1]
        # Take it out of LRU by hand to simulate the post-pop_lowest
        # limbo: ref_cnt=0, not in priority queue, not in free list.
        pool.free_block_queue.remove(block)
        assert block.prev_free_block is None
        assert block.next_free_block is None
        assert block not in pool.priority_eviction_queue
        assert block.ref_cnt == 0
        # touch() must not raise.
        pool.touch([block])
        assert block.ref_cnt == 1

    def test_get_new_blocks_drains_all_expired_to_lru(self, monkeypatch):
        """drain_expired must move ALL expired entries to the LRU, not
        just enough to satisfy the current allocation. Otherwise the
        next get_new_blocks call would re-fire the cache-eviction storm
        on the entries left behind in the priority queue.

        Pre-fix behavior: get_new_blocks(1) pops 1 entry from the
        priority queue via pop_lowest, leaves the other 2 expired
        entries in the queue. Each subsequent get_new_blocks would
        evict another cached block from the map.

        Post-fix behavior: drain_expired moves all 3 expired blocks to
        the LRU tail BEFORE any pop happens. get_new_blocks(1) then
        pops 1 from the LRU and the other 2 expired blocks sit in the
        LRU with their cached hashes intact until normal LRU order
        reaches them.
        """
        import time as time_mod

        pool = self._make_pool()

        # Stash 3 blocks into the priority queue with expiring sidecars.
        monkeypatch.setattr(time_mod, "monotonic", lambda: 100.0)
        target_ids = []
        for bid in (1, 2, 3):
            block = pool.blocks[bid]
            pool.free_block_queue.remove(block)  # take out of LRU
            _set_meta(
                pool.priority_eviction_queue,
                block,
                priority=50,
                expiry=150.0,
                last_freed=100.0,
            )
            pool.priority_eviction_queue.try_insert(block, last_freed_time=100.0)
            target_ids.append(bid)
        assert pool.priority_eviction_queue.num_blocks == 3
        lru_before = pool.free_block_queue.num_free_blocks

        # Advance past expiry, then ask for ONE block.
        monkeypatch.setattr(time_mod, "monotonic", lambda: 200.0)
        allocated = pool.get_new_blocks(1)
        assert len(allocated) == 1

        # Post-fix invariant: the priority queue is fully drained (0
        # entries), AND the LRU has gained 2 entries (we drained 3 and
        # consumed 1).
        # Pre-fix would leave 2 entries in the priority queue and the
        # LRU would have lost 0 entries net (started empty for our 3
        # blocks, ended empty too).
        assert pool.priority_eviction_queue.num_blocks == 0, (
            f"priority queue should be drained, has "
            f"{pool.priority_eviction_queue.num_blocks} entries"
        )
        assert pool.free_block_queue.num_free_blocks == lru_before + 2, (
            f"LRU should have gained 2 demoted-from-priority entries; "
            f"got {pool.free_block_queue.num_free_blocks - lru_before} delta"
        )
        # Sidecars also cleaned up for all 3.
        for bid in target_ids:
            assert bid not in pool.priority_eviction_queue._meta

    def test_evict_blocks_clears_sidecar(self):
        pool = self._make_pool()
        block = pool.blocks[1]
        _set_meta(pool.priority_eviction_queue, block, priority=50)
        # Don't insert into heap — just install sidecar (simulating a
        # block whose ref_cnt > 0 but had a prior priority).
        pool.evict_blocks({block.block_id})
        assert block.block_id not in pool.priority_eviction_queue._meta

    def test_priority_queue_pop_preserves_cache_map(self, monkeypatch):
        """A block popped from the priority queue keeps its hash in the
        cache map. Today the storm: every priority-queue pop wipes the
        cache hash; under the fix, only LRU-popleft does. This is the
        unit-level guard for the 1B cache-eviction-storm regression.
        """
        import time as time_mod

        from vllm.v1.core.kv_cache_utils import (
            BlockHash,
            make_block_hash_with_group_id,
        )

        pool = self._make_pool()
        block = pool.blocks[1]
        # Promote to in-use + cached, then free into the priority queue.
        pool.free_block_queue.remove(block)
        block.ref_cnt = 1
        raw_hash = BlockHash((42).to_bytes(32, "little"))
        h = make_block_hash_with_group_id(raw_hash, 0)
        block.block_hash = h
        pool.cached_block_hash_to_block.insert(h, block)

        monkeypatch.setattr(time_mod, "monotonic", lambda: 100.0)
        _set_meta(
            pool.priority_eviction_queue,
            block,
            priority=50,
            expiry=None,
            last_freed=100.0,
        )
        pool.free_blocks([block])
        assert block in pool.priority_eviction_queue
        assert pool.get_cached_block(raw_hash, [0]) is not None

        # Drain the LRU so the next get_new_blocks must dip into the
        # priority queue.
        for b in list(pool.free_block_queue.get_all_free_blocks()):
            if b is not block and b is not pool.null_block:
                pool.free_block_queue.remove(b)
                b.ref_cnt = 1
        assert pool.free_block_queue.num_free_blocks == 0
        assert pool.priority_eviction_queue.num_blocks == 1

        # Now pop via get_new_blocks. The block must come back with its
        # hash intact and the cache map entry preserved.
        allocated = pool.get_new_blocks(1)
        assert len(allocated) == 1
        assert allocated[0] is block
        assert block.ref_cnt == 1
        # The fix's guarantee:
        assert block.block_hash == h, (
            "Block hash was reset on priority-queue pop; the fix should "
            "have left it intact."
        )
        cached = pool.get_cached_block(raw_hash, [0])
        assert cached is not None and cached[0] is block, (
            "cached_block_hash_to_block entry was evicted on "
            "priority-queue pop; the fix should preserve it for "
            "subsequent prefix hits."
        )

    def test_lru_popleft_still_clears_cache_map(self):
        """LRU eviction's semantics are unchanged: popleft on a cached
        block must clear the cache map entry and reset the hash. Guards
        against accidentally decoupling the LRU path along with the
        priority-queue path."""
        from vllm.v1.core.kv_cache_utils import (
            BlockHash,
            make_block_hash_with_group_id,
        )

        pool = self._make_pool()
        block = pool.blocks[1]
        # Cache the block but leave it in LRU (no retention meta).
        pool.free_block_queue.remove(block)
        block.ref_cnt = 1
        raw_hash = BlockHash((77).to_bytes(32, "little"))
        h = make_block_hash_with_group_id(raw_hash, 0)
        block.block_hash = h
        pool.cached_block_hash_to_block.insert(h, block)
        pool.free_blocks([block])
        # Block is now in LRU with cache map entry intact.
        assert block not in pool.priority_eviction_queue
        cached = pool.get_cached_block(raw_hash, [0])
        assert cached is not None and cached[0] is block

        # Force LRU drain: ask for everything in LRU.
        n_free = pool.free_block_queue.num_free_blocks
        pool.get_new_blocks(n_free)

        # LRU semantics: cache map entry is gone, hash is cleared.
        assert pool.get_cached_block(raw_hash, [0]) is None, (
            "LRU popleft must still clear cached_block_hash_to_block; "
            "the fix targets PQ pop only."
        )
        assert block.block_hash is None, (
            "LRU popleft must still reset block.block_hash; the fix "
            "targets PQ pop only."
        )

    def test_cache_full_blocks_lazy_cleanup(self, monkeypatch):
        """A block popped from the priority queue carries its old hash.
        When cache_full_blocks runs on it again with a new hash, the
        old hash must be lazily removed from the cache map before the
        new hash is registered. This is the integration test that pairs
        with the priority-queue-preserves-cache-map test.

        Drives `cache_full_blocks` end-to-end (not the helper sequence)
        so that removing the lazy-cleanup branch in block_pool.py would
        be caught here.
        """
        import time as time_mod

        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.kv_cache_utils import (
            BlockHash,
            make_block_hash_with_group_id,
        )

        pool = self._make_pool()
        block = pool.blocks[1]
        pool.free_block_queue.remove(block)
        block.ref_cnt = 1
        raw_old = BlockHash((123).to_bytes(32, "little"))
        h_old = make_block_hash_with_group_id(raw_old, 0)
        block.block_hash = h_old
        pool.cached_block_hash_to_block.insert(h_old, block)

        monkeypatch.setattr(time_mod, "monotonic", lambda: 100.0)
        _set_meta(
            pool.priority_eviction_queue,
            block,
            priority=50,
            expiry=None,
            last_freed=100.0,
        )
        pool.free_blocks([block])

        # Drain LRU + pop the priority-queue entry. Block now carries
        # h_old; cache map still has h_old → block (per Task 1's fix).
        for b in list(pool.free_block_queue.get_all_free_blocks()):
            if b is not block and b is not pool.null_block:
                pool.free_block_queue.remove(b)
                b.ref_cnt = 1
        pool.get_new_blocks(1)
        assert block.block_hash == h_old, (
            "PQ pop should preserve the old hash for prefix-hit purposes."
        )
        cached_old = pool.get_cached_block(raw_old, [0])
        assert cached_old is not None and cached_old[0] is block

        # Now drive cache_full_blocks with a NEW raw hash. The real code
        # path must lazy-clean h_old from the cache map before
        # registering h_new — this exercises the
        # `if blk.block_hash is not None: self._maybe_evict_cached_block(blk)`
        # branch in block_pool.cache_full_blocks.
        raw_new = BlockHash((456).to_bytes(32, "little"))
        h_new = make_block_hash_with_group_id(raw_new, 0)

        # Minimal stub request — same pattern as
        # test_cache_full_blocks_routes_directives_to_queue. Without
        # extra_args the retention hook is a no-op, and with
        # enable_kv_cache_events=False the events branch is skipped, so
        # only block_hashes is load-bearing.
        class _Req:
            sampling_params: SamplingParams
            block_hashes: list

        request = _Req()
        request.sampling_params = SamplingParams()
        request.block_hashes = [raw_new]

        pool.cache_full_blocks(
            request=request,
            blocks=[block],
            num_cached_blocks=0,
            num_full_blocks=1,
            block_size=pool.hash_block_size,
            kv_cache_group_id=0,
        )

        assert pool.get_cached_block(raw_old, [0]) is None, (
            "cache_full_blocks must lazy-clean the stale hash before "
            "assigning the new one."
        )
        assert block.block_hash == h_new, (
            "cache_full_blocks must register the new hash on the block."
        )
        cached_new = pool.get_cached_block(raw_new, [0])
        assert cached_new is not None and cached_new[0] is block

    def test_prefix_hit_after_priority_queue_pop(self, monkeypatch):
        """End-to-end behavior: after a block is popped via the
        priority queue (ref_cnt becomes 1), a subsequent
        get_cached_block call for the same hash still returns that
        block. This is what the eviction-storm fix is for: the cache
        map entry survives across a PQ-driven allocation so the next
        prefix-matching request can hit.
        """
        import time as time_mod

        from vllm.v1.core.kv_cache_utils import (
            BlockHash,
            make_block_hash_with_group_id,
        )

        pool = self._make_pool()
        block = pool.blocks[1]
        pool.free_block_queue.remove(block)
        block.ref_cnt = 1
        raw_hash = BlockHash((321).to_bytes(32, "little"))
        h = make_block_hash_with_group_id(raw_hash, 0)
        block.block_hash = h
        pool.cached_block_hash_to_block.insert(h, block)

        monkeypatch.setattr(time_mod, "monotonic", lambda: 100.0)
        _set_meta(
            pool.priority_eviction_queue,
            block,
            priority=50,
            expiry=None,
            last_freed=100.0,
        )
        pool.free_blocks([block])

        # Drain LRU + pop priority queue → block.ref_cnt = 1.
        for b in list(pool.free_block_queue.get_all_free_blocks()):
            if b is not block and b is not pool.null_block:
                pool.free_block_queue.remove(b)
                b.ref_cnt = 1
        pool.get_new_blocks(1)
        assert block.ref_cnt == 1

        # The cache hit must still resolve. (The application would now
        # touch() this block; touch() handles ref_cnt > 0 correctly
        # without trying to remove from any queue.)
        hit = pool.get_cached_block(raw_hash, [0])
        assert hit is not None and hit[0] is block, (
            "Prefix-cache hit must succeed after priority-queue pop; "
            "this is the cache-eviction-storm regression guard."
        )


class TestStructuralInvariants:
    """Lock in the spec's 'sidecar pattern' contract:
    - KVCacheBlock must not gain feature-specific fields for retention.
    - Request must not gain retention attributes.

    If these tests fail, you are about to break the additive-only feel
    of this PR. Move the new state into PriorityEvictionQueue's sidecar
    instead.
    """

    def test_kv_cache_block_has_no_priority_fields(self):
        from dataclasses import fields

        from vllm.v1.core.kv_cache_utils import KVCacheBlock

        names = {f.name for f in fields(KVCacheBlock)}
        forbidden = {
            "priority",
            "priority_expiry",
            "priority_scope",
            "last_freed_time",
        }
        leaks = names & forbidden
        assert not leaks, (
            f"KVCacheBlock has retention-specific fields {leaks!r}. "
            "Move them to PriorityEvictionQueue's sidecar (see "
            "docs/superpowers/specs/2026-05-14-retention-api-super-minimal-design.md)."
        )

    def test_request_has_no_retention_attributes(self):
        import inspect

        from vllm.v1.request import Request

        src = inspect.getsource(Request.__init__)
        forbidden = ("retention_directives", "retention_scope")
        leaks = [name for name in forbidden if f"self.{name}" in src]
        assert not leaks, (
            f"Request.__init__ assigns to {leaks!r}. Read retention from "
            "request.sampling_params.extra_args at the use site instead."
        )

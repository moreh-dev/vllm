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

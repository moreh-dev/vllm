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

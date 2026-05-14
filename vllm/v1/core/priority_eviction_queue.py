# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Priority-based eviction queue with sidecar storage of per-block
retention metadata."""

import heapq
import time
from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import KVCacheBlock


@dataclass(slots=True)
class RetentionMeta:
    priority: int
    expiry: float | None
    scope: str | None
    last_freed_time: float


class PriorityEvictionQueue:
    def __init__(self) -> None:
        self._meta: dict[int, RetentionMeta] = {}
        self._heap: list[tuple[int, float, int, KVCacheBlock]] = []
        self._in_queue: set[int] = set()

    @property
    def num_blocks(self) -> int:
        return len(self._in_queue)

    def try_insert(self, block: KVCacheBlock) -> bool:
        """If the block has a sidecar entry, insert into the heap and
        return True. Otherwise return False (caller routes elsewhere)."""
        meta = self._meta.get(block.block_id)
        if meta is None:
            return False
        heapq.heappush(
            self._heap,
            (meta.priority, meta.last_freed_time, block.block_id, block),
        )
        self._in_queue.add(block.block_id)
        return True

    def remove(self, block: KVCacheBlock) -> None:
        """Remove the block from the heap (lazy delete: just drops from
        _in_queue). The sidecar entry is preserved so that the block's
        priority survives a reuse cycle and is restored on next free."""
        self._in_queue.discard(block.block_id)

    def pop_lowest(self) -> KVCacheBlock | None:
        """Pop the lowest-priority block from the heap. Stale entries
        (block_id no longer in _in_queue) and expired entries are skipped.
        Returns None when the queue is empty."""
        now = time.monotonic()
        while self._heap:
            _, _, block_id, block = heapq.heappop(self._heap)
            if block_id not in self._in_queue:
                continue
            meta = self._meta.get(block_id)
            if meta is not None and meta.expiry is not None and meta.expiry <= now:
                # Expired — discard sidecar and skip (caller treats as
                # unprioritized; will be served by the LRU path).
                self._in_queue.discard(block_id)
                self._meta.pop(block_id, None)
                continue
            self._in_queue.discard(block_id)
            self._meta.pop(block_id, None)
            return block
        return None

    def apply_directives(
        self,
        blocks: list[KVCacheBlock],
        directives: list[dict],
        scope: str | None,
        block_size: int,
    ) -> None:
        """For each full block, find the highest-priority directive whose
        token range overlaps the block's range and update the sidecar
        entry. Blocks with no matching directive are left untouched at
        this stage (ownership/clear semantics added in Task 7)."""
        if not directives:
            return
        now = time.monotonic()
        for idx, block in enumerate(blocks):
            if block.is_null:
                continue
            token_start = idx * block_size
            token_end = token_start + block_size
            best_priority = -1
            best_duration: float | None = None
            for d in directives:
                d_start = d.get("start", 0)
                d_end = d.get("end")
                if d_end is not None and d_end <= token_start:
                    continue
                if d_start >= token_end:
                    continue
                p = d.get("priority", 0)
                if p > best_priority:
                    best_priority = p
                    best_duration = d.get("duration")
            if best_priority < 0:
                continue
            expiry = now + best_duration if best_duration is not None else None
            self._meta[block.block_id] = RetentionMeta(
                priority=best_priority,
                expiry=expiry,
                scope=scope,
                last_freed_time=0.0,
            )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Priority-based eviction queue with sidecar storage of per-block
retention metadata."""

import heapq
import os
import time
from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import KVCacheBlock

# Priority threshold for entering the priority eviction queue. Entries
# whose sidecar priority falls below this value get routed to the LRU
# free list instead. The LRU has to stay populated; otherwise every
# get_new_blocks call has to pull from the priority queue, and the
# protected entries never get the breathing room to satisfy
# prefix-cache hits (1B Slack-regime regression). Continuum's
# priorities are 90 (system_prompt) / 70 (history) / 50 (tail); the
# default 60 routes tail to LRU and keeps the rest protected.
_PRIORITY_THRESHOLD = int(os.environ.get("VLLM_RETENTION_PRIORITY_THRESHOLD", "60"))


@dataclass(slots=True)
class RetentionMeta:
    priority: int
    expiry: float | None
    scope: str | None
    last_freed_time: float


class PriorityEvictionQueue:
    def __init__(self) -> None:
        self._meta: dict[int, RetentionMeta] = {}
        self._heap: list[tuple[int, float, int, int, KVCacheBlock]] = []
        self._in_queue: set[int] = set()
        # Per-block monotonic generation counter. try_insert bumps the
        # block's generation and stamps it on the pushed heap tuple;
        # pop_lowest skips any tuple whose generation is not the block's
        # current one. This closes the lazy-delete hazard: remove() leaves a
        # tuple in the heap and a later re-insert pushes a second tuple for
        # the same block_id, so without the generation stamp pop_lowest could
        # evict using the stale tuple's (priority, last_freed_time) and invert
        # the eviction order. _gen is monotonic per block_id (never reset on
        # remove/pop, only on clear) and bounded by the physical block count.
        self._gen: dict[int, int] = {}

    @property
    def num_blocks(self) -> int:
        return len(self._in_queue)

    def __contains__(self, block: KVCacheBlock) -> bool:
        return block.block_id in self._in_queue

    def try_insert(
        self,
        block: KVCacheBlock,
        last_freed_time: float | None = None,
    ) -> bool:
        """If the block has an unexpired sidecar entry, insert into the heap
        and return True. Otherwise return False (caller routes elsewhere).

        If the sidecar entry exists but is already expired, drop the sidecar
        and return False so the caller routes the block to the LRU free
        list. Inserting an expired entry would leave the block in the
        priority queue indefinitely (or, on subsequent pop_lowest, drop it
        into the limbo state where it belongs to no queue at all).

        last_freed_time, when provided, overrides the value stored in the
        sidecar entry. This is used by free_blocks() to stamp the current
        monotonic time on freshly-freed blocks so the heap tiebreak
        reflects this most-recent free."""
        meta = self._meta.get(block.block_id)
        if meta is None:
            return False
        if meta.expiry is not None and meta.expiry <= time.monotonic():
            # Expired-on-free: drop sidecar, route to LRU free list.
            self._meta.pop(block.block_id, None)
            return False
        if meta.priority < _PRIORITY_THRESHOLD:
            # Below-threshold: drop sidecar, route to LRU. Keeps the
            # LRU populated with low-priority cached blocks so
            # get_new_blocks can satisfy demand without draining
            # protected (high-priority) entries from the priority
            # queue. See the priority-threshold spec for rationale.
            self._meta.pop(block.block_id, None)
            return False
        if last_freed_time is not None:
            meta.last_freed_time = last_freed_time
        gen = self._gen.get(block.block_id, 0) + 1
        self._gen[block.block_id] = gen
        heapq.heappush(
            self._heap,
            (meta.priority, meta.last_freed_time, gen, block.block_id, block),
        )
        self._in_queue.add(block.block_id)
        return True

    def remove(self, block: KVCacheBlock) -> None:
        """Remove the block from the heap (lazy delete: just drops from
        _in_queue). The sidecar entry is preserved so that the block's
        priority survives a reuse cycle and is restored on next free."""
        self._in_queue.discard(block.block_id)

    def pop_lowest(self) -> KVCacheBlock | None:
        """Pop the lowest-priority block from the heap.

        A heap tuple is stale and skipped when either (a) its block_id is no
        longer in _in_queue (removed via touch/drain), or (b) its generation
        is not the block's current generation — i.e. a later try_insert
        pushed a newer tuple for the same block_id after a remove()+re-free
        reuse cycle. Skipping (b) keeps eviction ordered by the block's
        CURRENT (priority, last_freed_time) instead of a stale tuple's.

        Expired entries are NOT special-cased here: callers must invoke
        drain_expired() first to demote expired entries to the LRU free list.

        Returns None only when the heap has no live entries left."""
        while self._heap:
            _, _, gen, block_id, block = heapq.heappop(self._heap)
            if block_id not in self._in_queue:
                continue
            if gen != self._gen.get(block_id):
                # Superseded by a newer insert for the same block_id.
                continue
            self._in_queue.discard(block_id)
            self._meta.pop(block_id, None)
            return block
        return None

    def drain_expired(self) -> list[int]:
        """Pop all currently-expired entries off the priority queue and
        return their block_ids. The caller is expected to route the
        corresponding blocks into the LRU free list — expiry means
        "protection released", not "evict immediately".

        Drains lazily: walks _in_queue, checks each entry's sidecar
        expiry, and discards from _in_queue + _meta where expired. Stale
        heap entries are cleaned up by the next pop_lowest as usual.
        """
        now = time.monotonic()
        drained: list[int] = []
        for block_id in list(self._in_queue):
            meta = self._meta.get(block_id)
            if meta is not None and meta.expiry is not None and meta.expiry <= now:
                self._in_queue.discard(block_id)
                self._meta.pop(block_id, None)
                drained.append(block_id)
        return drained

    def clear_priority(self, block_id: int) -> None:
        """Drop the sidecar entry for a block (called when its hash is
        reset or it is permanently evicted from prefix cache)."""
        self._meta.pop(block_id, None)
        self._in_queue.discard(block_id)

    def clear(self) -> None:
        """Drop all sidecar entries and heap state."""
        self._meta.clear()
        self._heap.clear()
        self._in_queue.clear()
        self._gen.clear()

    def apply_directives(
        self,
        blocks: list[KVCacheBlock],
        directives: list[dict],
        scope: str | None,
        block_size: int,
    ) -> None:
        """For each full block, find the highest-priority overlapping
        directive and update the sidecar entry under these rules:

        - Escalation (new > current priority): any caller may raise priority
          and takes ownership of the block.
        - Downgrade or refresh (new <= current priority): only the current
          owner may do this.
        - No matching directive: if the caller has scope ownership of this
          block, the sidecar entry is cleared.
        """
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

            current = self._meta.get(block.block_id)

            if best_priority < 0:
                # No matching directive. Owner-initiated clear only.
                if scope is not None and current is not None and current.scope == scope:
                    self._meta.pop(block.block_id, None)
                continue

            expiry = now + best_duration if best_duration is not None else None
            current_priority = current.priority if current is not None else -1
            if best_priority > current_priority:
                # Escalation: any caller may raise priority and takes ownership.
                self._meta[block.block_id] = RetentionMeta(
                    priority=best_priority,
                    expiry=expiry,
                    scope=scope,
                    last_freed_time=current.last_freed_time if current else 0.0,
                )
            elif current is not None and scope is not None and current.scope == scope:
                # Same scope: owner may downgrade or refresh.
                self._meta[block.block_id] = RetentionMeta(
                    priority=best_priority,
                    expiry=expiry,
                    scope=scope,
                    last_freed_time=current.last_freed_time,
                )
            # Non-owner downgrade: silently ignored.

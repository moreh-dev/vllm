# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Priority-based eviction queue with sidecar storage of per-block
retention metadata."""

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

    def pop_lowest(self) -> KVCacheBlock | None:
        if not self._in_queue:
            return None
        # Implementation deferred to Task 3
        raise NotImplementedError

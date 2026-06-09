# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest


@pytest.fixture(autouse=True)
def _retention_priority_threshold_zero(monkeypatch):
    """Lower `_PRIORITY_THRESHOLD` to 0 inside every test in this dir.

    The retention priority threshold is a production safety knob (default
    60) that routes sub-threshold sidecar entries to the LRU rather than
    the priority queue. The unit tests in this directory use priority=50
    throughout as a convention to mean "any prioritized entry"; we lower
    the threshold inside tests so the convention keeps working without
    rewriting every test fixture. Tests that explicitly exercise the
    threshold behavior re-set it via monkeypatch in-body.
    """
    import vllm.v1.core.priority_eviction_queue as pq_mod

    monkeypatch.setattr(pq_mod, "_PRIORITY_THRESHOLD", 0)

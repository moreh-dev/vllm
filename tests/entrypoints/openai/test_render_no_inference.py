# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for --disable-inference mode with AsyncMPClientNoInference.

Verifies that the /render endpoints work without GPU inference,
and that actual inference requests do not complete.
"""

import httpx
import pytest
import pytest_asyncio

from ...utils import RemoteOpenAIServer

MODEL_NAME = "openai/gpt-oss-20b"


@pytest.fixture(scope="module")
def server():
    args = ["--disable-inference"]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with httpx.AsyncClient(
        base_url=server.url_for(""), timeout=30.0
    ) as http_client:
        yield http_client


@pytest.mark.asyncio
async def test_health_check(client):
    """Health endpoint should work in no-inference mode."""
    response = await client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_chat_completion_render_works_without_inference(client):
    """Chat completion render should work without inference engine."""
    response = await client.post(
        "/v1/chat/completions/render",
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) == 2

    conversation, engine_prompts = data

    # Verify conversation contains messages
    assert len(conversation) > 0

    # Find the user message (Harmony models use 'author' dict instead of 'role')
    user_msg = None
    for msg in conversation:
        # Check for direct 'role' key (standard format)
        if msg.get("role") == "user":
            user_msg = msg
            break
        # Check for nested 'author' dict (Harmony format)
        author = msg.get("author")
        if isinstance(author, dict) and author.get("role") == "user":
            user_msg = msg
            break

    assert user_msg is not None, (
        f"User message not found in conversation: {conversation}"
    )

    # Verify tokenization occurred
    assert len(engine_prompts) > 0
    assert "prompt_token_ids" in engine_prompts[0]
    assert len(engine_prompts[0]["prompt_token_ids"]) > 0


@pytest.mark.asyncio
async def test_completion_render_works_without_inference(client):
    """Completion render should work without inference engine."""
    response = await client.post(
        "/v1/completions/render",
        json={
            "model": MODEL_NAME,
            "prompt": "Hello, how are you?",
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) > 0
    assert "prompt_token_ids" in data[0]
    assert "prompt" in data[0]
    assert len(data[0]["prompt_token_ids"]) > 0


@pytest.mark.asyncio
async def test_inference_disabled(client):
    """Actual inference should fail in no-inference mode.

    AsyncMPClientNoInference does not start the engine core,
    so inference requests should return an internal server error.
    """
    response = await client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
            ],
            "max_tokens": 5,
        },
    )

    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == 500

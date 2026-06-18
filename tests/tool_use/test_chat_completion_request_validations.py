# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest


def test_chat_completion_request_with_no_tools():
    # tools key is not present
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "facebook/opt-125m",
        }
    )
    assert request.tool_choice == "none"

    # tools key is None
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "facebook/opt-125m",
            "tools": None,
        }
    )
    assert request.tool_choice == "none"

    # tools key present but empty -- should be rejected
    with pytest.raises(ValueError, match="must not be an empty array"):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "tools": [],
            }
        )


@pytest.mark.parametrize("tool_choice", ["auto", "required"])
def test_chat_completion_request_with_tool_choice_but_no_tools(tool_choice):
    with pytest.raises(
        ValueError, match="When using `tool_choice`, `tools` must be set."
    ):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "tool_choice": tool_choice,
            }
        )

    with pytest.raises(
        ValueError, match="When using `tool_choice`, `tools` must be set."
    ):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "tool_choice": tool_choice,
                "tools": None,
            }
        )


def test_reasoning_content_normalized_to_reasoning():
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "content": "4",
                    "reasoning_content": "2+2 equals 4",
                },
                {"role": "user", "content": "Are you sure?"},
            ],
            "model": "facebook/opt-125m",
        }
    )
    assistant_msg = request.messages[1]
    assert assistant_msg.get("reasoning") == "2+2 equals 4"
    assert "reasoning_content" not in assistant_msg


def test_reasoning_takes_precedence_over_reasoning_content():
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "content": "4",
                    "reasoning": "from reasoning field",
                    "reasoning_content": "from reasoning_content field",
                },
            ],
            "model": "facebook/opt-125m",
        }
    )
    assistant_msg = request.messages[1]
    assert assistant_msg.get("reasoning") == "from reasoning field"
    assert "reasoning_content" not in assistant_msg


def test_no_reasoning_fields_unchanged():
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            "model": "facebook/opt-125m",
        }
    )
    assistant_msg = request.messages[1]
    assert assistant_msg.get("reasoning") is None
    assert "reasoning_content" not in assistant_msg


SAMPLE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
        },
    },
}


def test_structured_outputs_with_named_tool_choice_rejected():
    """structured_outputs cannot be combined with a named tool_choice."""
    with pytest.raises(
        ValueError,
        match="structured outputs or tools, not both",
    ):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "tools": [SAMPLE_TOOL],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "get_weather"},
                },
                "structured_outputs": {"json": {"type": "object"}},
            }
        )


def test_structured_outputs_with_auto_tool_choice_allowed():
    """structured_outputs with tool_choice 'auto' should be allowed."""
    request = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "facebook/opt-125m",
            "tools": [SAMPLE_TOOL],
            "tool_choice": "auto",
            "structured_outputs": {"json": {"type": "object"}},
        }
    )
    assert request.tool_choice == "auto"


def test_multiple_structured_outputs_rejected():
    """Only one kind of structured output constraint is allowed."""
    with pytest.raises(
        ValueError,
        match="You can only use one kind of constraints",
    ):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "facebook/opt-125m",
                "structured_outputs": {
                    "json": {"type": "object"},
                    "regex": ".*",
                },
            }
        )


def test_retention_directives_field_round_trips():
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )

    req = ChatCompletionRequest(
        model="dummy",
        messages=[{"role": "user", "content": "hi"}],
        retention_directives=[{"start": 0, "end": 16, "priority": 80}],
        retention_scope="alice",
    )
    sp = req.to_sampling_params(
        max_tokens=16,
        default_sampling_params={},
    )
    assert sp.extra_args["retention_directives"] == [
        {"start": 0, "end": 16, "priority": 80}
    ]
    assert sp.extra_args["retention_scope"] == "alice"


def _build_request_with_directives(directives):
    return ChatCompletionRequest(
        model="dummy",
        messages=[{"role": "user", "content": "hi"}],
        retention_directives=directives,
    )


def test_retention_directives_monotonic_priorities_valid():
    # Strictly decreasing priorities across rising token positions: valid.
    _build_request_with_directives(
        [
            {"start": 0, "end": 100, "priority": 90},
            {"start": 100, "end": 200, "priority": 60},
            {"start": 200, "end": 300, "priority": 30},
        ]
    )


def test_retention_directives_monotonic_priorities_equal_is_valid():
    # Non-increasing (equal) priorities: valid.
    _build_request_with_directives(
        [
            {"start": 0, "end": 100, "priority": 50},
            {"start": 100, "end": 200, "priority": 50},
        ]
    )


def test_retention_directives_empty_or_none_is_valid():
    _build_request_with_directives(None)
    _build_request_with_directives([])


def test_retention_directives_single_directive_is_valid():
    _build_request_with_directives([{"start": 0, "end": 16, "priority": 80}])


def test_retention_directives_increasing_priority_rejected():
    with pytest.raises(ValueError, match="non-increasing"):
        _build_request_with_directives(
            [
                {"start": 0, "end": 100, "priority": 30},
                {"start": 100, "end": 200, "priority": 80},  # increases — invalid
            ]
        )


def test_retention_directives_unsorted_input_still_validated():
    # Input order is unsorted, but the validator should sort by start
    # before checking monotonicity.
    with pytest.raises(ValueError, match="non-increasing"):
        _build_request_with_directives(
            [
                {"start": 100, "end": 200, "priority": 80},
                {
                    "start": 0,
                    "end": 100,
                    "priority": 30,
                },  # sorted: 30 then 80 → invalid
            ]
        )

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Any

from openai import Stream


@dataclass
class TextDelta:
    content: str

    def __str__(self) -> str:
        return self.content


class StreamEventType(str, Enum):
    TEXT_DELTA = "text_delta"
    MESSAGE_COMPLETE = "message_complete"
    ERROR = "error"

    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_COMPLETE = "tool_call_complete"


@dataclass
class TokenUsage:
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    def __add__(self, other: TokenUsage):
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


@dataclass
class StreamEvent:
    type: StreamEventType
    text: TextDelta | None = None
    error: str | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    tool_calls: list[ToolCall] | None = None

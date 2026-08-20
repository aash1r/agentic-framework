from dataclasses import dataclass
from typing import Any
from prompts.system_prompt import get_system_prompt
from utils.text import count_tokens


@dataclass
class MessageItem:
    role: str
    content: str
    token_usage: int | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"role": self.role}

        if self.content:
            result["content"] = self.content

        if self.tool_calls:
            result["tool_calls"] = self.tool_calls

        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id

        return result


class ContextManager:
    def __init__(self):
        self._system_prompt = get_system_prompt()
        self._model_name = "cohere/north-mini-code:free"
        self._messages: list[MessageItem] = []

    def add_user_message(self, content):
        item = MessageItem(
            role="user",
            content=content or "",
            token_usage=count_tokens(text=content or "", model=self._model_name),
        )
        self._messages.append(item)

    def add_assistant_message(
        self, content, tool_calls: list[dict[str, Any]] | None = None
    ):
        item = MessageItem(
            role="assistant",
            content=content or "",
            token_usage=count_tokens(text=content or "", model=self._model_name),
            tool_calls=tool_calls,
        )
        self._messages.append(item)

    def add_tool_message(self, tool_call_id, content):
        item = MessageItem(
            role="tool",
            content=content,
            token_usage=count_tokens(text=content or "", model=self._model_name),
            tool_call_id=tool_call_id,
        )
        self._messages.append(item)

    def get_messages(self) -> list[dict[str, Any]]:
        messages = []

        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        for item in self._messages:
            messages.append(item.to_dict())

        return messages

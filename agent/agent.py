from __future__ import annotations
from typing import AsyncGenerator
from agent.events import AgentEvent, AgentEventType
from client.llm_client import LLMClient
from client.response import StreamEventType


class Agent:
    def __init__(self):
        self.client = LLMClient()

    async def run(self, messages: str):
        yield AgentEvent.agent_start(message=messages)

        final_response: str | None = None
        async for event in self._agentic_loop(message=messages):
            yield event

            if event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content")

        yield AgentEvent.agent_end(final_response)

    async def _agentic_loop(self, message) -> AsyncGenerator[AgentEvent, None]:
        messages = [{"role": "user", "content": message}]
        response_text = ""

        async for event in self.client.chat_completion(messages=messages, stream=True):
            if event.type == StreamEventType.TEXT_DELTA:
                content = event.text.content
                response_text += content
                yield AgentEvent.text_delta(content)
            if event.type == StreamEventType.ERROR:
                yield AgentEvent.agent_error(
                    error=event.error or "Unknown error occured"
                )

        if response_text:
            yield AgentEvent.text_complete(response_text)

    async def __aenter__(self) -> Agent:
        return self

    async def __aexit__(self, exc_type, exc_val, tb):
        if self.client:
            await self.client.close()
            self.client = None

from email import message
from typing import Any, AsyncGenerator
from agent.events import AgentEventType
from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError
import asyncio
import os
from dotenv import load_dotenv

from client.response import (
    StreamEvent,
    StreamEventType,
    TextDelta,
    TokenUsage,
    ToolCall,
)

# EventType,

load_dotenv()


class LLMClient:
    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None
        self._max_retries: int = 3

    def get_client(self) -> AsyncOpenAI:
        if self._client is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY not set in environment")
            self._client = AsyncOpenAI(
                api_key=api_key, base_url="https://openrouter.ai/api/v1"
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
    ) -> AsyncGenerator[StreamEvent, None]:
        client = self.get_client()

        kwargs = {
            "model": "cohere/north-mini-code:free",
            "messages": messages,
            "stream": stream,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        for attempt in range(self._max_retries + 1):
            try:
                if stream:
                    async for event in self._stream_response(
                        client=client, kwargs=kwargs
                    ):
                        yield event
                else:
                    event = await self._non_stream_response(
                        client=client, kwargs=kwargs
                    )
                    yield event
                return
            except RateLimitError as e:
                if attempt < self._max_retries:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                    continue
                yield StreamEvent(
                    type=StreamEventType.ERROR, error=f"Rate Limit Exceeded :{e}"
                )
                return
            except APIConnectionError as e:
                if attempt < self._max_retries:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                    continue
                yield StreamEvent(
                    type=StreamEventType.ERROR, error=f"Connection Error :{e}"
                )
                return
            except APIError as e:
                yield StreamEvent(type=StreamEventType.ERROR, error=f"API Error :{e}")
                return

    async def _stream_response(
        self, client: AsyncOpenAI, kwargs: dict[str, Any]
    ) -> AsyncGenerator[StreamEvent, None]:
        response = await client.chat.completions.create(**kwargs)

        finish_reason: str | None = None
        usage: TokenUsage | None = None
        in_progress: dict[int, ToolCall] = {}

        async for chunk in response:
            if hasattr(chunk, "usage") and chunk.usage:
                usage = TokenUsage(
                    completion_tokens=chunk.usage.completion_tokens,
                    prompt_tokens=chunk.usage.prompt_tokens,
                    total_tokens=chunk.usage.total_tokens,
                    cached_tokens=chunk.usage.prompt_tokens_details.cached_tokens,
                )

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if choice.finish_reason:
                finish_reason = choice.finish_reason

            if delta.content:
                yield StreamEvent(
                    type=StreamEventType.TEXT_DELTA, text=TextDelta(delta.content)
                )

            if delta.tool_calls is not None:
                for tc in delta.tool_calls:
                    if tc.function.name is not None:
                        idx = ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=tc.function.arguments,
                        )
                        in_progress[tc.index] = idx
                    else:
                        in_progress[tc.index].arguments += tc.function.arguments

        yield StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            finish_reason=finish_reason,
            usage=usage,
            tool_calls=list(in_progress.values()),
        )

    async def _non_stream_response(
        self, client: AsyncOpenAI, kwargs: dict[str, Any]
    ) -> StreamEvent:
        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        text_delta = None
        if message.content:
            text_delta = TextDelta(message.content)

        usage = None
        if response.usage:
            usage = TokenUsage(
                completion_tokens=response.usage.completion_tokens,
                prompt_tokens=response.usage.prompt_tokens,
                total_tokens=response.usage.total_tokens,
                cached_tokens=response.usage.prompt_tokens_details.cached_tokens,
            )

        return StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            text=text_delta,
            finish_reason=choice.finish_reason,
            usage=usage,
        )

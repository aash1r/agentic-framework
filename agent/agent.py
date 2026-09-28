from __future__ import annotations
from typing import AsyncGenerator
from agent.events import AgentEvent, AgentEventType
from client.response import StreamEventType, ToolCall
import json
from pathlib import Path
from agent.session import Session


class Agent:
    def __init__(self):
        self.session: Session | None = Session()

    async def run(self, messages: str):
        yield AgentEvent.agent_start(message=messages)
        self.session.context_manager.add_user_message(messages)

        final_response: str | None = None
        async for event in self._agentic_loop():
            yield event

            if event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content")

        yield AgentEvent.agent_end(final_response)

    async def _agentic_loop(self) -> AsyncGenerator[AgentEvent, None]:
        max_turns = 100

        for turns in range(max_turns):
            response_text = ""
            tool_schemas = self.session.tool_registry.get_schemas()
            tool_calls: list[ToolCall] = []

            async for event in self.session.client.chat_completion(
                messages=self.session.context_manager.get_messages(),
                tools=tool_schemas if tool_schemas else None,
                stream=True,
            ):
                # print(event)
                if event.type == StreamEventType.TEXT_DELTA:
                    content = event.text.content
                    response_text += content
                    yield AgentEvent.text_delta(content)

                elif event.type == StreamEventType.MESSAGE_COMPLETE:
                    if event.tool_calls:
                        tool_calls.extend(event.tool_calls)
                elif event.type == StreamEventType.ERROR:
                    yield AgentEvent.agent_error(
                        error=event.error or "Unknown error occured"
                    )

            self.session.context_manager.add_assistant_message(
                response_text or None,
                (
                    [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": str(tc.arguments),
                            },
                        }
                        for tc in tool_calls
                    ]
                    if tool_calls
                    else None
                ),
            )

            if response_text:
                yield AgentEvent.text_complete(response_text)

            if not tool_calls:
                return

            for tc in tool_calls:
                args = json.loads(tc.arguments)
                yield AgentEvent.tool_call_start(
                    call_id=tc.id, name=tc.name, arguments=args
                )

                result = await self.session.tool_registry.invoke(
                    name=tc.name, params=args, cwd=Path.cwd()
                )

                yield AgentEvent.tool_call_complete(
                    call_id=tc.id, name=tc.name, result=result
                )

                self.session.context_manager.add_tool_message(
                    tool_call_id=tc.id,
                    content=result.output
                    or f"Error: {result.error}\n\nOutput:\n{result.output}",
                )

    async def __aenter__(self) -> Agent:
        return self

    async def __aexit__(self, exc_type, exc_val, tb):
        if self.session and self.session.client:
            await self.session.client.close()
            self.session = None

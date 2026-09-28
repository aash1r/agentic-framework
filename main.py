import asyncio
from pathlib import Path
import sys
import click
from asyncio import run
from agent.agent import Agent
from agent.events import AgentEventType
from ui.renderer import Renderer, get_console

console = get_console()


class CLI:
    def __init__(self) -> None:
        self.agent: Agent | None = None
        self.renderer = Renderer(console=console)

    async def run_single(self, message: str) -> str | None:
        async with Agent() as agent:
            self.agent = agent
            return await self._process_message(message)

    async def run_interactive(self) -> str | None:
        self.renderer.print_welcome(
            title="PlsCode",
            lines=[
                f"model: cohere/north-mini-code:free",
                f"cwd: {Path.cwd()}",
            ],
        )
        async with Agent() as agent:
            self.agent = agent
            while True:
                try:
                    user_input = console.input("\n[user]>[/user] ").strip()
                    if not user_input:
                        continue
                    await self._process_message(user_input)
                except KeyboardInterrupt:
                    console.print("\n[dim]Use /exit to quit[/dim]")
                except EOFError:
                    break

    def get_tool_kind(self, tool_name: str):
        tool_kind = None
        tool = self.agent.session.tool_registry.get(tool_name)
        if not tool:
            tool_kind = None
        tool_kind = tool.kind.value
        return tool_kind

    async def _process_message(self, message):
        if not self.agent:
            return None

        assistant_streaming = False
        final_response: str | None = None

        async for event in self.agent.run(message):
            # print(event)
            if event.type == AgentEventType.TEXT_DELTA:
                content = event.data.get("content", "")
                if not assistant_streaming:
                    self.renderer.begin_assistant()
                    assistant_streaming = True
                self.renderer.stream_assistant_delta(content=content)
            elif event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content", "")
                if assistant_streaming:
                    self.renderer.end_assistant()
                    assistant_streaming = False
            elif event.type == AgentEventType.AGENT_ERROR:
                error = event.data.get("error", "Unknow Error!")
                console.print(f"\n[error]Error: {error}[/error]")
            elif event.type == AgentEventType.TOOL_CALL_START:
                tool_name = event.data.get("name", "unknown")
                tool_kind = self.get_tool_kind(tool_name=tool_name)
                self.renderer.tool_call_start(
                    call_id=event.data.get("call_id", ""),
                    name=tool_name,
                    tool_kind=tool_kind,
                    arguments=event.data.get("arguments", {}),
                )
            elif event.type == AgentEventType.TOOL_CALL_COMPLETE:
                tool_name = event.data.get("name", "unknown")
                tool_kind = self.get_tool_kind(tool_name=tool_name)
                self.renderer.tool_call_complete(
                    call_id=event.data.get("call_id", ""),
                    name=tool_name,
                    tool_kind=tool_kind,
                    success=event.data.get("success", False),
                    output=event.data.get("output", ""),
                    error=event.data.get("error"),
                    metadata=event.data.get("metadata"),
                    truncated=event.data.get("truncated", False),
                )

        return final_response


@click.command()
@click.argument("prompt", required=False)
def main(prompt: str):
    cli = CLI()
    if prompt:
        result = asyncio.run(cli.run_single(prompt))
        if result is None:
            sys.exit(1)
    else:
        asyncio.run(cli.run_interactive())


if __name__ == "__main__":
    main()

# import asyncio
# from client.llm_client import LLMClient


# async def main():
#     client = LLMClient()
#     messages = [{"role": "user", "content": "What's Up?"}]
#     async for event in client.chat_completion(messages, True):
#         print(event)
#     print("done")


# if __name__ == "__main__":
#     asyncio.run(main())

import asyncio
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

    async def _process_message(self, message):
        if not self.agent:
            return None

        assistant_streaming = False

        async for event in self.agent.run(message):
            if event.type == AgentEventType.TEXT_DELTA:
                content = event.data.get("content", "")
                if not assistant_streaming:
                    self.renderer.begin_assistant()
                    assistant_streaming = True
                self.renderer.stream_assistant_delta(content=content)


@click.command()
@click.argument("prompt", required=False)
def main(prompt: str):
    cli = CLI()
    if prompt:
        result = asyncio.run(cli.run_single(prompt))
        if result is None:
            sys.exit(1)


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

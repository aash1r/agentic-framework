import asyncio
import click

from agent.agent import Agent
from agent.events import AgentEventType

class CLI:
    def __init__(self) -> None:
        self.agent : Agent | None = None
    
    async def run_single(self, message:str) -> str | None:
        async with Agent() as agent:
            self.agent = agent
            self._process_message(message)

    async def _process_message(self, message):    
        if not self.agent:
            return None
        
        async for event in self.agent.run(message):
            if event.type == AgentEventType.TEXT_DELTA:
                content = event.data.get("content", "")



@click.command()
@click.argument("prompt",required=False)
def main(prompt:str):
    asyncio.run(run(prompt))
    print("done")

if __name__ == "__main__":
    main()
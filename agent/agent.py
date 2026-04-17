from agent.events import AgentEvent, AgentEventType
from client.response import StreamEventType


class Agent:
    def __init__(self):
        self.client = LLMClient()

    async def run(self, messages:str):
        yield AgentEvent.agent_start(message=messages)
        
        async for event in self._agentic_loop():
            yield event
            
            if event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content")
        
        yield AgentEvent.agent_end(final_response)

    async def _agentic_loop(self) -> AsyncGenerator[AgentEvent,None]:
        messages= [{"role":"user","content":"Hello, how are you?"}]
        response_text = ""

        async for event in self.client.chat_completion(messages=messages,stream=True):
            if event.type == StreamEventType.TEXT_DELTA:
                content = event.text_delta.content
                response_text += content
                yield AgentEvent.text_delta(content)
            if event.type == StreamEventType.ERROR:
                yield AgentEvent.agent_error(error=event.error or "Unknown error occured")
        
        if response_text:
            yield AgentEvent.text_complete(response_text)
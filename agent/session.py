from datetime import datetime
from uuid import uuid4
from client.llm_client import LLMClient
from context.context_manager import ContextManager
from tools.registry import create_default_registry

class Session:
    def __init__(self):
        self.client: LLMClient = LLMClient()
        self.context_manager: ContextManager = ContextManager()
        self.tool_registry = create_default_registry()
        self.session_id = str(uuid4())
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.turn_count = 0
    
    def increment_turn_count(self):
        self.turn_count += 1
        self.updated_at = datetime.now()
        
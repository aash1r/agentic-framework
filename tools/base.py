from abc import ABC
import abc
from enum import Enum
from typing import Any
from dataclasses import dataclass, field
from pydantic import BaseModel, ValidationError


class ToolKind(str, Enum):
    READ = "read"
    WRITE = "write"
    NETWORK = "network"
    MCP = "mcp"
    SHELL = "shell"
    MEMORY = "memory"


@dataclass
class ToolInvocation:
    params: dict[str, Any]
    cwd: str


@dataclass
class ToolConfirmation:
    tool_name: str
    params: str
    description: str


@dataclass
class ToolResult:
    success: bool
    output: str
    error: str | None = None
    metadata: dict[str, Any] = field(default=dict)


class Tool(ABC):
    name: str = "Base tool"
    description: str = "base tool desc"
    kind: ToolKind = ToolKind.READ

    @property
    def schema(self) -> dict[str, Any] | type["BaseModel"]:
        raise NotImplementedError("You must define schema property or class attribute")

    @abc.abstractmethod
    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        pass

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        schema = self.schema

        if isinstance(schema, type) and issubclass(schema, BaseModel):
            try:
                schema.model_validate(params)
                return []
            except ValidationError as e:
                return [str(e)]

        return []

    def is_mutating(self, params: dict[str, Any]) -> bool:
        return self.kind in {
            ToolKind.WRITE,
            ToolKind.SHELL,
            ToolKind.NETWORK,
            ToolKind.MEMORY,
        }

    def get_confirmation(self, invocation: ToolInvocation):
        if not self.is_mutating(invocation.params):
            return None

        return ToolConfirmation(
            tool_name=self.name, params=invocation.params, description=self.description
        )

from pathlib import Path
from typing import Any

from tools.base import Tool, ToolInvocation, ToolResult
from tools.builtin.read_file import ReadTool
from tools.builtin.write_file import WriteFileTool


class ToolRegistry:
    # create empty dictionary
    def __init__(self):
        self.tools: dict[str, Tool] = {}

    # create a register function which takes a tool and checks if already present then override it else add it
    def register(self, tool: Tool):
        if tool.name in self.tools:
            print(f"Overriding exisiting tool {tool.name}")

        self.tools[tool.name] = tool
        print("Registered Tool")

    def unregister(self, name: str):
        if name in self.tools:
            del self.tools[name]
            return True

        return False

    def get(self, name):
        if name in self.tools:
            return self.tools[name]
        else:
            return None

    def get_tools(self):
        tools: list[Tool] = []

        for tool in self.tools.values():
            tools.append(tool)

        return tools

    def get_schemas(self):
        return [tool.to_open_ai() for tool in self.get_tools()]

    async def invoke(self, name: str, params: dict[str, Any], cwd: Path):
        tool = self.get(name=name)
        if tool is None:
            return ToolResult.error_result(
                error=f"Unknown tool: {tool}",
                metadata={"tool_name": tool},
            )
        validation_errors = tool.validate_params(params=params)
        if validation_errors:
            return ToolResult.error_result(
                error=f"Invalid parameters: {';'.join(validation_errors)}",
                metadata={"tool_name": name},
            )

        invocation = ToolInvocation(params=params, cwd=cwd)
        try:
            return await tool.execute(invocation=invocation)
        except Exception as e:
            return ToolResult.error_result(
                error=f"Internal error: {str(e)}", metadata={"tool_name": name}
            )


def create_default_registry():
    registry = ToolRegistry()

    BUILT_IN_TOOLS = [ReadTool(), WriteFileTool()]

    for tool in BUILT_IN_TOOLS:
        registry.register(tool)

    return registry

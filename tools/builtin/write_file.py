from pydantic import BaseModel, Field

from tools.base import Tool, ToolInvocation, ToolKind, ToolResult
from utils.paths import resolve_path


class WriteFileParams(BaseModel):
    path: str = Field(description= "Path to the file to write")
    content: str = Field(description= "Content to write to the file")


class WriteFileTool(Tool):
    name = "write_file"
    description = ("Write content to a file. Creates the file if it doesn't exist,"
                  "or overwrites if it does. Parent directories are automatically created,"
                  "Use this for creating new files or completely replacing file contents."
                  "For partial modifications, use edit file tool instead.")
    kind = ToolKind.WRITE
    schema = WriteFileParams

    async def execute(self, invocation: ToolInvocation):
        params = WriteFileParams(**invocation.params)
        path = resolve_path(invocation.cwd, params.path)

        if path.exists() and not path.is_file():
            return ToolResult.error_result(
                error=f"Path exists and is a directory, not a file: {path}"
            )

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(params.content, encoding="utf-8")
            return ToolResult(success=True, output="Success")
        except Exception as e:
            return ToolResult.error_result(f"Failed to write file: {e}")

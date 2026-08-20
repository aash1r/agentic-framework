from pydantic import BaseModel, Field
from tools.base import Tool, ToolInvocation, ToolKind, ToolResult
from utils.paths import is_binary_file, resolve_path
from utils.text import count_tokens


class ReadFileParams(BaseModel):
    path: str = Field("Path to the file to read")
    offset: int = Field(
        1, ge=1, description="Line number to start reading the file. Defaults to 1"
    )
    limit: int | None = Field(
        None,
        ge=1,
        description="Maximum number of lines to read, if none then reads entire file.",
    )


class ReadTool(Tool):
    name: str = "read_file"
    description = (
        "Read the contents of a text file. Returns the file content with line numbers."
        "For large files, use offset and limit to read specific portions. "
        "Cannot read binary files (images, executables, etc.)."
    )
    kind = ToolKind.READ
    schema = ReadFileParams

    MAX_FILE_SIZE = 1024 * 1024 * 10

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = ReadFileParams(**invocation.params)
        path = resolve_path(invocation.cwd, params.path)

        if not path.exists():
            return ToolResult.error_result(f"File not found: {path}")

        if not path.is_file():
            return ToolResult.error_result(f"Path is not a file: {path}")

        file_size = path.stat().st_size

        if file_size > self.MAX_FILE_SIZE:
            return ToolResult.error_result(f"File too large: {file_size}")

        if is_binary_file(path=path):
            return ToolResult.error_result("Cannot read binary file")

        try:
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeEncodeError:
                content = path.read_text(encoding="latin-1")

            lines = content.splitlines()
            total_lines = len(lines)

            if total_lines == 0:
                ToolResult.success_result(output="File is Empty", metadata={"lines": 0})

            start_idx = max(0, params.offset - 1)

            if params.limit is not None:
                end_idx = min(start_idx + params.limit, total_lines)
            else:
                end_idx = total_lines

            selected_lines = lines[start_idx:end_idx]
            formatted_lines = []

            for i, line in enumerate(selected_lines, start=start_idx + 1):
                formatted_lines.append(f"{i:6}|{line}")

            output = "\n".join(formatted_lines)
            token_count = count_tokens(output)

            return ToolResult.success_result(output=output)

        except Exception as e:
            return ToolResult.error_result(f"Failed to read file {e}")

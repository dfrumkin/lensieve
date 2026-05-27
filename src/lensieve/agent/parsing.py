import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from lensieve.tools.enums import Tool

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

logger = logging.getLogger(__name__)


class ToolCallPayload(BaseModel):
    name: Tool
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    tool: Tool
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.tool.value,
            "arguments": self.arguments,
        }

    def to_tagged_text(self) -> str:
        return f"<tool_call>\n{json.dumps(self.to_dict(), ensure_ascii=False)}\n</tool_call>"


def extract_tool_call(text: str) -> ToolCall | None:
    tool_call_match = TOOL_CALL_RE.search(text)  # Sometimes the model may include additional explanation.
    if tool_call_match is None:
        return None

    if text.strip() != tool_call_match.group(0).strip():
        logger.warning("Model emitted text around tool call: %r", text)

    try:
        payload = ToolCallPayload.model_validate_json(tool_call_match.group(1))
    except ValidationError:
        return None

    return ToolCall(
        tool=payload.name,
        arguments=payload.arguments,
    )


def serialize_tool_result(result: Any) -> str:
    if isinstance(result, str):
        return result

    if isinstance(result, BaseModel):
        return result.model_dump_json(indent=2, exclude_none=True)

    try:
        return json.dumps(result, default=str, indent=2)
    except TypeError:
        return str(result)

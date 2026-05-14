import json
import re
from typing import Any

from pydantic import BaseModel

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def extract_tool_call(text: str) -> dict[str, Any] | None:
    match = TOOL_CALL_RE.search(text)
    if match is None:
        return None

    try:
        tool_call = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None

    if not isinstance(tool_call, dict):
        return None

    if "name" not in tool_call or "arguments" not in tool_call:
        return None

    return tool_call


def serialize_tool_result(result: Any) -> str:
    if isinstance(result, str):
        return result

    if isinstance(result, BaseModel):
        return result.model_dump_json(indent=2)

    try:
        return json.dumps(result, default=str, indent=2)
    except TypeError:
        return str(result)

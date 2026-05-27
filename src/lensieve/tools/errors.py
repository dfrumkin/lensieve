from pydantic import BaseModel


class ToolError(BaseModel):
    error_type: str
    message: str

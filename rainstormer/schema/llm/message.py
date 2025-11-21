import enum
from datetime import datetime, UTC

from pydantic import BaseModel, ConfigDict, Field


class ChatRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    DEVELOPER = "developer"
    TOOL = "tool"


class Function(BaseModel):
    """Represents a function call within a tool call."""

    model_config = ConfigDict(from_attributes=True)

    name: str = Field(description="Name of the function to call")
    arguments: str = Field(description="JSON string of function arguments")


class ToolCall(BaseModel):
    """Represents a tool call in the OpenAI format."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(description="Unique identifier for the tool call")
    type: str = Field(default="function", description="Type of tool call")
    function: Function = Field(description="Function call details")


class ChatMessage(BaseModel):
    """Represents a single message in a chat."""

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)

    role: ChatRole = Field(description="Role of the message")
    content: str | None = Field(
        default=None,
        description="Content of the message (optional when tool_calls present)",
    )
    name: str | None = Field(default=None, description="Name of the tool, if any")
    tool_calls: list[ToolCall] | None = Field(
        default=None, description="List of tool calls"
    )
    tool_call_id: str | None = Field(default=None, description="ID of the tool call")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of the message",
    )

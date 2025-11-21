import enum
from collections.abc import Awaitable, Callable
from copy import deepcopy
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


def _default_parameters_schema() -> dict[str, object]:
    return {"type": "object", "properties": {}, "required": []}


class ChatRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    DEVELOPER = "developer"
    TOOL = "tool"


class ToolCallFunction(BaseModel):
    """Represents a function call within a tool call."""

    model_config = ConfigDict(from_attributes=True)

    name: str = Field(description="Name of the function to call")
    arguments: str = Field(description="JSON string of function arguments")


class ToolCall(BaseModel):
    """Represents a tool call in the OpenAI format."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(description="Unique identifier for the tool call")
    type: str = Field(default="function", description="Type of tool call")
    function: ToolCallFunction = Field(description="Function call details")


class ToolDefinitionFunction(BaseModel):
    """Definition of a callable function tool."""

    model_config = ConfigDict(from_attributes=True)

    name: str = Field(description="Name of the tool function")
    description: str = Field(description="Description of what the tool does")
    parameters: dict[str, object] = Field(
        default_factory=_default_parameters_schema,
        description="JSON schema describing tool arguments",
    )


class ToolDefinition(BaseModel):
    """Serializable representation of a tool definition."""

    model_config = ConfigDict(from_attributes=True)

    type: str = Field(default="function")
    function: ToolDefinitionFunction


class Tool(BaseModel):
    """
    Represents a callable tool with metadata and an execution handler.

    Handlers may be synchronous or asynchronous and should return either
    a string or JSON-serializable object that can be passed back to the model.
    """

    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)

    name: str = Field(description="Name of the tool")
    description: str = Field(description="Description of the tool behaviour")
    parameters: dict[str, object] = Field(
        default_factory=_default_parameters_schema,
        description="JSON schema for tool arguments",
    )
    handler: Callable[..., Awaitable[object] | object] = Field(
        exclude=True, description="Callable used to execute the tool"
    )

    @property
    def definition(self) -> ToolDefinition:
        """Return the OpenAI-compliant tool definition."""
        return ToolDefinition(
            function=ToolDefinitionFunction(
                name=self.name,
                description=self.description,
                parameters=deepcopy(self.parameters),
            )
        )

    def to_openai_param(self) -> dict[str, object]:
        """Return the tool definition dict for the OpenAI client."""
        return self.definition.model_dump(exclude_none=True)


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

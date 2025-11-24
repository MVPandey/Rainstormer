"""Contains the chat model service class."""

import asyncio
import inspect
import json

from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel, Field, SecretStr

from ..schema.llm.message import ChatMessage, ChatRole, Tool, ToolCall, ToolCallFunction
from ..utils.config import Config
from ..utils.exceptions import ChatModelError
from ..utils.logger import logger


class ChatModelHyperparams(BaseModel):
    """Hyperparameters for the chat model."""

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1, le=8192)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    presence_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    n: int = Field(default=1, ge=1, le=10)
    stop: list[str] = Field(default=[], description="Stop tokens")
    stream: bool = Field(default=False, description="Stream the response")
    logprobs: int | None = Field(default=None, ge=0, le=10)


class ModelConfig(BaseModel):
    """Configuration for multi-model strategy."""

    primary_model: str = Field(
        default="google/gemini-3-pro-preview",
        description="Primary model for high-quality generation",
    )

    auxiliary_model: str = Field(
        default="openai/o3-mini",
        description="Cheaper model for judges, simulated users, etc.",
    )

    @property
    def model_for_generation(self) -> str:
        """Model for main brainstorming generation."""
        return self.primary_model

    @property
    def model_for_simulated_user(self) -> str:
        """Model for simulated user responses."""
        return self.auxiliary_model

    @property
    def model_for_micro_judge(self) -> str:
        """Model for per-turn micro-scoring."""
        return self.auxiliary_model

    @property
    def model_for_final_judge(self) -> str:
        """Model for final conversation evaluation."""
        return self.primary_model

    @property
    def model_for_novelty_judge(self) -> str:
        """Model for novelty comparison."""
        return self.auxiliary_model


class ChatModelService:
    """Service class for the chat model."""

    # ============================================================================
    # Initialization
    # ============================================================================

    def __init__(
        self,
        api_key: str | SecretStr | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        config: Config | None = None,
        max_tool_iterations: int = 50,
    ):
        """
        Initialize the chat model service.

        Args:
            api_key: API key for the LLM service.
            base_url: Base URL for the LLM service.
            model_name: Name of the model to use.
            config: Optional Config object to fall back to.
            max_tool_iterations: Safety cap for recursive tool executions.
        """
        self.llm_api_key = api_key or (config.llm_api_key if config else None)
        self.llm_base_url = (
            base_url
            or (config.llm_base_url if config else None)
            or "https://openrouter.ai/api/v1"
        )
        self.llm_name = (
            model_name or (config.llm_name if config else None) or "openai/o3-mini"
        )
        self.max_tool_iterations = max(1, max_tool_iterations)

        if not self.llm_api_key:
            logger.warning("No API key provided for ChatModelService.")

        api_key_str = (
            self.llm_api_key.get_secret_value()
            if isinstance(self.llm_api_key, SecretStr)
            else self.llm_api_key
        )

        self._client: AsyncOpenAI = AsyncOpenAI(
            base_url=self.llm_base_url, api_key=api_key_str
        )

        logger.info(
            f"Initialized ChatModelService with model={self.llm_name}, base_url={self.llm_base_url}"
        )

    # ============================================================================
    # Public Interface
    # ============================================================================

    async def _chat(
        self,
        messages: list[ChatMessage] | ChatMessage | str,
        hyperparams: ChatModelHyperparams | None = None,
        tools: list[Tool] | None = None,
        model_override: str | None = None,
    ) -> ChatCompletion:
        """
        Execute a chat completion, optionally handling tool calls until resolution.

        Args:
            messages: Messages to send to the model.
            hyperparams: Optional hyperparameters for the completion.
            tools: Optional tools available for the model to call.
            model_override: Optional model name to use instead of the default.
                           Useful for multi-model strategies (e.g., cheaper models
                           for judges or simulated users).
        """
        try:
            normalized_messages = self._normalize_messages(messages)
            tool_defs, tool_map = self._prepare_tools(tools)

            completion = await self._create_completion(
                normalized_messages, hyperparams, tool_defs, model_override
            )

            iteration = 0
            while (
                tool_map
                and completion.choices
                and completion.choices[0].message.tool_calls
            ):
                if iteration >= self.max_tool_iterations:
                    raise ChatModelError(
                        "Maximum tool iterations exceeded. Possible tool loop detected."
                    )

                assistant_message = self._from_completion_message(
                    completion.choices[0].message
                )
                normalized_messages.append(assistant_message)
                tool_responses = await self._run_tool_calls(
                    assistant_message.tool_calls, tool_map
                )
                normalized_messages.extend(tool_responses)

                completion = await self._create_completion(
                    normalized_messages, hyperparams, tool_defs, model_override
                )
                iteration += 1

            logger.debug("Chat completion received")
            return completion

        except ChatModelError:
            raise
        except Exception as exc:
            logger.error(f"Chat completion failed: {exc}")
            raise ChatModelError(f"Error chatting with model: {exc}") from exc

    # ============================================================================
    # Message Processing
    # ============================================================================

    async def _create_completion(
        self,
        messages: list[ChatMessage],
        hyperparams: ChatModelHyperparams | None,
        tool_defs: list[dict[str, object]] | None,
        model_override: str | None = None,
    ) -> ChatCompletion:
        """
        Invoke the OpenAI completion endpoint.

        Args:
            messages: Normalized messages to send.
            hyperparams: Optional hyperparameters.
            tool_defs: Optional tool definitions.
            model_override: Optional model name to use instead of default.
        """
        # normalize messages for OpenAI API (e.g., DEVELOPER -> SYSTEM)
        normalized_for_api: list[ChatMessage] = []
        for m in messages:
            if m.role == ChatRole.DEVELOPER:
                # OpenAI doesn't support DEVELOPER role, map to SYSTEM
                m = ChatMessage(**{**m.model_dump(), "role": ChatRole.SYSTEM})
            normalized_for_api.append(m)

        payload_messages = [
            message.model_dump(
                exclude_none=True,
                mode="json",
                exclude={"created_at"},
            )
            for message in normalized_for_api
        ]

        # Use model_override if provided, otherwise default to self.llm_name
        model_name = model_override or self.llm_name

        request_kwargs: dict[str, object] = {
            "model": model_name,
            "messages": payload_messages,
        }

        if model_override:
            logger.debug(f"Using model override: {model_override}")

        if hyperparams:
            request_kwargs.update(hyperparams.model_dump(exclude_none=True))
        if tool_defs:
            request_kwargs["tools"] = tool_defs

        logger.debug(
            "Dispatching chat completion request",
            extra={"has_tools": bool(tool_defs), "message_count": len(messages)},
        )

        requested_n = hyperparams.n if hyperparams else 1

        if requested_n > 1:
            return await self._parallel_completions(requested_n, request_kwargs)

        return await self._client.chat.completions.create(**request_kwargs)

    async def _parallel_completions(
        self,
        n: int,
        request_kwargs: dict[str, object],
    ) -> ChatCompletion:
        """Make n parallel completion calls and combine results."""
        single_request_kwargs = {**request_kwargs, "n": 1}

        tasks = [
            self._client.chat.completions.create(**single_request_kwargs)
            for _ in range(n)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_choices = []
        base_completion = None
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Parallel completion call failed: {res}")
                continue
            if base_completion is None:
                base_completion = res
            if res.choices:
                choice = res.choices[0]
                choice.index = len(all_choices)
                all_choices.append(choice)

        if base_completion is None:
            raise ChatModelError("All parallel completion calls failed")

        base_completion.choices = all_choices
        return base_completion

    @staticmethod
    def _normalize_messages(
        messages: list[ChatMessage] | ChatMessage | str,
    ) -> list[ChatMessage]:
        """Ensure all inputs are ChatMessage instances."""
        if isinstance(messages, str):
            return [ChatMessage(role=ChatRole.USER, content=messages)]

        if isinstance(messages, ChatMessage):
            return [messages]

        normalized: list[ChatMessage] = []
        for message in messages:
            if isinstance(message, ChatMessage):
                normalized.append(message)
            elif isinstance(message, str):
                normalized.append(ChatMessage(role=ChatRole.USER, content=message))
            elif isinstance(message, dict):
                normalized.append(ChatMessage(**message))
            else:
                raise ChatModelError(
                    f"Unsupported message type: {type(message).__name__}"
                )
        return normalized

    @staticmethod
    def _from_completion_message(openai_message) -> ChatMessage:
        """Convert an OpenAI completion message into a ChatMessage."""
        content = openai_message.content
        if isinstance(content, list):
            content = "".join(
                fragment.get("text", "")
                if isinstance(fragment, dict)
                else str(fragment)
                for fragment in content
            )

        tool_calls = None
        if openai_message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tool_call.id,
                    type=tool_call.type or "function",
                    function=ToolCallFunction(
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments or "{}",
                    ),
                )
                for tool_call in openai_message.tool_calls
            ]

        return ChatMessage(
            role=ChatRole(openai_message.role),
            content=content,
            tool_calls=tool_calls,
        )

    # ============================================================================
    # Tool Handling
    # ============================================================================

    def _prepare_tools(
        self, tools: list[Tool] | None
    ) -> tuple[list[dict[str, object]] | None, dict[str, Tool]]:
        """Prepare tool definitions and lookup map for execution."""
        if not tools:
            return None, {}

        tool_map: dict[str, Tool] = {}
        tool_defs: list[dict[str, object]] = []
        for tool in tools:
            if tool.name in tool_map:
                raise ChatModelError(f"Duplicate tool name detected: {tool.name}")
            tool_map[tool.name] = tool
            tool_defs.append(tool.to_openai_param())

        return tool_defs, tool_map

    async def _run_tool_calls(
        self, tool_calls: list[ToolCall] | None, tool_map: dict[str, Tool]
    ) -> list[ChatMessage]:
        """Execute the requested tool calls and return tool response messages."""
        if not tool_calls:
            return []

        responses: list[ChatMessage] = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool = tool_map.get(tool_name)
            if not tool:
                raise ChatModelError(f"Received unknown tool call: {tool_name}")

            arguments = self._parse_tool_arguments(tool_call.function.arguments)
            result = await self._invoke_tool(tool, arguments)

            responses.append(
                ChatMessage(
                    role=ChatRole.TOOL,
                    name=tool_name,
                    content=result,
                    tool_call_id=tool_call.id,
                )
            )

        return responses

    async def _invoke_tool(self, tool: Tool, arguments: dict[str, object]) -> str:
        """Invoke tool handler and normalize the result."""
        sig = inspect.signature(tool.handler)
        params = list(sig.parameters.values())

        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)

        if accepts_kwargs or len(params) != 1:
            try:
                result = tool.handler(**arguments)
            except TypeError:
                result = tool.handler(arguments)
        else:
            result = tool.handler(arguments)

        if inspect.isawaitable(result):
            result = await result

        if isinstance(result, (dict, list)):
            return json.dumps(result)

        if not isinstance(result, str):
            return str(result)

        return result

    @staticmethod
    def _parse_tool_arguments(arguments: str | None) -> dict[str, object]:
        """Parse the JSON arguments string provided by the model."""
        if not arguments:
            return {}

        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ChatModelError(
                f"Invalid JSON in tool arguments: {arguments}"
            ) from exc

        if not isinstance(parsed, dict):
            raise ChatModelError("Tool arguments must decode into a JSON object.")

        return parsed

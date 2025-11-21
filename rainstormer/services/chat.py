"""Contains the chat model service class."""

from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel, Field, SecretStr

from ..schema.llm.message import ChatMessage
from ..utils.config import Config
from ..utils.exceptions import ChatModelError
from ..utils.logger import logger


class ChatModelHyperparams(BaseModel):
    """Hyperparameters for the chat model."""

    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(default=1000, ge=1, le=8192)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    presence_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    n: int = Field(default=1, ge=1, le=10)
    stop: list[str] = Field(default=[], description="Stop tokens")
    stream: bool = Field(default=False, description="Stream the response")
    logprobs: int | None = Field(default=None, ge=0, le=10)


class ChatModelService:
    """Service class for the chat model."""

    def __init__(
        self,
        api_key: str | SecretStr | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        hyperparams: ChatModelHyperparams | None = None,
        config: Config | None = None,
    ):
        """
        Initialize the chat model service.

        Args:
            api_key: API key for the LLM service.
            base_url: Base URL for the LLM service.
            model_name: Name of the model to use.
            hyperparams: Hyperparameters for the model.
            config: Optional Config object to fall back to.
        """
        self.hyperparams = hyperparams
        self.llm_api_key = api_key or (config.llm_api_key if config else None)
        self.llm_base_url = (
            base_url
            or (config.llm_base_url if config else None)
            or "https://openrouter.ai/api/v1"
        )
        self.llm_name = (
            model_name or (config.llm_name if config else None) or "openai/gpt-4o"
        )

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

    async def _chat(
        self,
        messages: list[ChatMessage] | ChatMessage,
    ) -> ChatCompletion:
        """Chat with the model."""
        logger.debug(f"Initiating chat completion with {len(messages)} messages")

        try:
            if not isinstance(messages, list):
                messages = [messages]
            completion = await self._client.chat.completions.create(
                model=self.llm_name,
                messages=messages,
                **(self.hyperparams.model_dump() if self.hyperparams else {}),
            )

            logger.debug("Chat completion received")
            return completion

        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise ChatModelError(f"Error chatting with model: {e}") from e

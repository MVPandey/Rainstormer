from typing import Any

from ..schema.llm.message import ChatMessage, ChatRole
from ..services.chat import ChatModelService
from ..services.mcts import MCTSService
from ..utils.config import Config


class Rainstormer:
    """
    Main client for Rainstormer.

    Usage:
        client = Rainstormer(idea="I want to build a flying car")
        tree = await client.run()
    """

    def __init__(
        self,
        idea: str | None = None,
        history: list[dict[str, str]] | list[ChatMessage] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
    ):
        """
        Initialize the Rainstormer client.

        Args:
            idea: Initial idea string.
            history: Conversation history.
            api_key: LLM API key.
            base_url: LLM base URL.
            model_name: LLM model name.
        """
        self.config = Config()
        if api_key:
            self.config.llm_api_key = api_key
        if base_url:
            self.config.llm_base_url = base_url
        if model_name:
            self.config.llm_name = model_name

        self.llm_service = ChatModelService(config=self.config)
        self.mcts_service = MCTSService(llm_service=self.llm_service)

        self.messages: list[ChatMessage] = []
        if history:
            for msg in history:
                if isinstance(msg, dict):
                    self.messages.append(ChatMessage(**msg))
                elif isinstance(msg, ChatMessage):
                    self.messages.append(msg)

        if idea:
            self.messages.append(ChatMessage(role=ChatRole.USER, content=idea))

    async def run(self, iterations: int = 5, depth: int = 3) -> dict[str, Any]:
        """
        Run the brainstorming session.

        Args:
            iterations: Number of MCTS iterations.
            depth: Max depth of the tree.

        Returns:
            The MCTS tree as a dictionary.
        """
        if not self.messages:
            raise ValueError("No initial idea or history provided.")

        return await self.mcts_service.run(
            initial_messages=self.messages, iterations=iterations, depth=depth
        )

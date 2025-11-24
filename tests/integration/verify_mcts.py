import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from rainstormer import Rainstormer
from rainstormer.schema.llm.message import ChatMessage, ChatRole


async def run_verification():
    print("Starting MCTS Verification with Rainstormer Client...")

    # Mock responses
    expansion_response = MagicMock()
    expansion_response.choices = [
        MagicMock(message=MagicMock(content="Idea A")),
        MagicMock(message=MagicMock(content="Idea B")),
    ]

    judge_response_a = MagicMock()
    judge_response_a.choices = [
        MagicMock(message=MagicMock(content='{"winner_index": 0}'))
    ]

    sim_user_response = MagicMock()
    sim_user_response.choices = [
        MagicMock(message=MagicMock(content="That sounds interesting."))
    ]

    sim_asst_response = MagicMock()
    sim_asst_response.choices = [
        MagicMock(message=MagicMock(content="Here is more detail."))
    ]

    async def chat_side_effect(messages, hyperparams=None, tools=None):
        content = messages[-1].content if messages else ""
        if "Compare these two brainstorming responses" in str(content):
            return judge_response_a
        if "You are simulating a User" in str(messages[0].content):
            return sim_user_response
        if "You are a creative brainstorming assistant" in str(messages[0].content):
            if hyperparams and getattr(hyperparams, "n", 1) == 2:
                return expansion_response
            else:
                return sim_asst_response
        return expansion_response

    # Patch ChatModelService inside Rainstormer
    with patch(
        "rainstormer.services.chat.ChatModelService._chat", new_callable=AsyncMock
    ) as mock_chat:
        mock_chat.side_effect = chat_side_effect

        # Initialize Client
        client = Rainstormer(idea="I want to invent a new toy.")

        # Run MCTS
        tree_data = await client.run(iterations=2, depth=2)

        print("MCTS Run Completed.")
        print(json.dumps(tree_data, indent=2))

        # Verify structure
        assert tree_data["message"]["content"] == "I want to invent a new toy."
        assert len(tree_data["children"]) > 0
        print("Verification Successful: Tree structure created.")


if __name__ == "__main__":
    asyncio.run(run_verification())

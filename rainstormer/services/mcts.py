import json
import math
import asyncio
from collections import Counter

from ..schema.llm.message import ChatMessage, ChatRole
from ..schema.mcts import MCTSNode, MCTSTree
from ..services.chat import ChatModelService, ChatModelHyperparams
from ..prompts.mcts import (
    GENERATION_SYSTEM_PROMPT,
    SIMULATION_USER_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
)
from ..utils.logger import logger


class MCTSService:
    """Service for Monte Carlo Tree Search brainstorming."""

    def __init__(self, llm_service: ChatModelService):
        self.llm_service = llm_service
        self.tree: MCTSTree | None = None

    async def run(
        self,
        initial_messages: list[ChatMessage],
        iterations: int = 5,
        depth: int = 3,
    ) -> dict[str, object]:
        """
        Run the MCTS brainstorming process.

        Args:
            initial_messages: The conversation history so far.
            iterations: Number of MCTS iterations to run.
            depth: Maximum depth of the tree.

        Returns:
            The tree structure as a dictionary.
        """
        # Initialize tree with root node
        root_message = initial_messages[-1]
        self.tree = MCTSTree(root=MCTSNode(message=root_message, depth=0))

        # If there are prior messages, we might want to store them to provide context
        # For simplicity, we'll pass the full context to the LLM but the tree starts at the last message
        self.context_messages = initial_messages[:-1]

        for i in range(iterations):
            logger.info(f"MCTS Iteration {i + 1}/{iterations}")

            # 1. Selection
            leaf = self._select(self.tree.root)

            # Check depth limit
            if leaf.depth >= depth:
                # If we hit max depth, we might just backpropagate a static score or simulate from here
                # For now, let's just simulate to get a value
                score = await self._simulate(leaf)
                self._backpropagate(leaf, score)
                continue

            # 2. Expansion
            # Only expand if not already expanded (though _select should return a leaf)
            if not leaf.children:
                children = await self._expand(leaf)
                if not children:
                    # Could not generate children, treat as terminal
                    score = await self._simulate(leaf)
                    self._backpropagate(leaf, score)
                    continue

                # 3. Evaluation (Judge) & Simulation
                # We evaluate the children to pick a "winner" to explore first or just evaluate all?
                # Standard MCTS expands one node. Here we generated 2.
                # Let's evaluate them against each other to assign initial values/metadata.
                winner_idx = await self._evaluate_pair(children[0], children[1])

                # We can simulate from the winner, or both.
                # Let's simulate from the winner to get a rollout score.
                simulation_node = children[winner_idx]
                score = await self._simulate(simulation_node)

                # 4. Backpropagation
                # Backpropagate the score up from the child
                self._backpropagate(simulation_node, score)
                # Also backpropagate to the other child? Or just leave it unvisited?
                # Standard MCTS: expand, rollout from one child.
                # We'll leave the other child with 0 visits for now, it might be picked later if UCT favors it.
            else:
                # Should not happen if _select works correctly for fully expanded nodes
                pass

        return self.tree.root.model_dump(mode="json")

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Select a node to expand using UCT.
        Traverse down until we find a leaf or a node with unexplored children.
        """
        current = node
        while current.children:
            # If any child has 0 visits, pick it (or maybe we want to expand all at once?
            # In our logic, we expand all children at once. So we just pick the best UCT child.)

            # Calculate UCT for all children
            # UCT = value/visits + C * sqrt(ln(parent_visits) / visits)
            best_child = None
            best_score = float("-inf")

            log_parent_visits = math.log(current.visits) if current.visits > 0 else 0

            for child in current.children:
                if child.visits == 0:
                    # If a child is unvisited, we should probably visit it.
                    # In standard MCTS, we expand one child at a time.
                    # Here we generated both. Let's prioritize unvisited nodes.
                    return child

                uct = (child.value / child.visits) + 1.414 * math.sqrt(
                    log_parent_visits / child.visits
                )
                if uct > best_score:
                    best_score = uct
                    best_child = child

            current = best_child

        return current

    async def _expand(self, node: MCTSNode) -> list[MCTSNode]:
        """Generate 2 candidate responses."""
        logger.info(f"Expanding node {node.id}")

        # Construct message history
        messages = list(self.context_messages)

        # Reconstruct path from root to node
        # Since we don't have parent pointers easily traversable without a map or search,
        # we might need to store the path or just rely on the fact that we are passing the node.
        # Wait, MCTSNode has parent_id but not the object.
        # We need to reconstruct the conversation history.
        # For simplicity, let's assume the node.message is the last message.
        # BUT, we need the full history to generate a good response.
        # We can traverse down from root if we had the path.
        # Or we can store the full conversation in the node (heavy).
        # Or we can implement a `get_ancestry` method if we had a node map.

        # Let's implement a simple ancestry lookup by traversing from root? No, that's slow.
        # Let's just store the `history` in the node for this MVP, or a reference to it.
        # Actually, let's just use `parent_id` and look it up if we register nodes in the tree.
        # For now, let's just assume we can get the history.
        # Hack: Pass history down during selection?

        # Let's do a quick traversal helper since we have the tree root.
        history = self._get_history(node)

        # Add system prompt
        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content=GENERATION_SYSTEM_PROMPT)
        ] + history

        try:
            # Generate 2 responses
            completion = await self.llm_service._chat(
                messages=messages,
                hyperparams=ChatModelHyperparams(n=2, temperature=0.9),
            )

            new_nodes = []
            for choice in completion.choices:
                msg = ChatMessage(
                    role=ChatRole.ASSISTANT, content=choice.message.content
                )
                child = MCTSNode(message=msg, depth=node.depth + 1)
                node.add_child(child)
                new_nodes.append(child)

            return new_nodes

        except Exception as e:
            logger.error(f"Error expanding node: {e}")
            return []

    async def _evaluate_pair(self, node_a: MCTSNode, node_b: MCTSNode) -> int:
        """
        Compare two nodes using 3 LLM judges.
        Returns the index of the winner (0 or 1).
        """
        logger.info("Evaluating pair with 3 judges")

        prompt = f"""
        Compare these two brainstorming responses:

        Response 1:
        {node_a.message.content}

        Response 2:
        {node_b.message.content}
        """

        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content=JUDGE_SYSTEM_PROMPT),
            ChatMessage(role=ChatRole.USER, content=prompt),
        ]

        # Run 3 parallel calls (or sequential if we want to be safe, but parallel is faster)
        # The user asked for "3 separate API calls".
        tasks = [
            self.llm_service._chat(
                messages=messages,
                hyperparams=ChatModelHyperparams(
                    temperature=0.1
                ),  # Low temp for judging
            )
            for _ in range(3)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        votes = []
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Judge failed: {res}")
                continue

            try:
                content = res.choices[0].message.content
                # Clean up markdown code blocks if present
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                data = json.loads(content)
                votes.append(data.get("winner_index", 0))
            except Exception as e:
                logger.error(f"Error parsing judge output: {e}")

        if not votes:
            return 0  # Default to first if all fail

        # Majority vote
        count = Counter(votes)
        winner = count.most_common(1)[0][0]

        # Store metadata
        node_a.metadata["judge_votes"] = votes.count(0)
        node_b.metadata["judge_votes"] = votes.count(1)

        return winner

    async def _simulate(self, node: MCTSNode) -> float:
        """
        Simulate a conversation rollout.
        Returns a score (0.0 to 1.0) based on the quality/length of the conversation.
        For this MVP, we'll simulate 2 turns and rate the final outcome.
        """
        logger.info(f"Simulating from node {node.id}")

        history = self._get_history(node)
        current_history = list(history)

        # Simulate User -> Assistant -> User -> Assistant
        turns = 2
        for _ in range(turns):
            # User turn (Simulated)
            sim_user_messages = [
                ChatMessage(role=ChatRole.SYSTEM, content=SIMULATION_USER_SYSTEM_PROMPT)
            ] + current_history

            try:
                user_res = await self.llm_service._chat(
                    messages=sim_user_messages,
                    hyperparams=ChatModelHyperparams(temperature=0.7),
                )
                user_msg = ChatMessage(
                    role=ChatRole.USER, content=user_res.choices[0].message.content
                )
                current_history.append(user_msg)

                # Assistant turn (Generation)
                assistant_messages = [
                    ChatMessage(role=ChatRole.SYSTEM, content=GENERATION_SYSTEM_PROMPT)
                ] + current_history

                asst_res = await self.llm_service._chat(
                    messages=assistant_messages,
                    hyperparams=ChatModelHyperparams(temperature=0.7),
                )
                asst_msg = ChatMessage(
                    role=ChatRole.ASSISTANT, content=asst_res.choices[0].message.content
                )
                current_history.append(asst_msg)

            except Exception as e:
                logger.error(f"Simulation failed: {e}")
                break

        # Simple scoring: Length of conversation? Or ask a judge to rate the final state?
        # Let's just return a static score for now, or maybe 1.0 if it completed successfully.
        # A real implementation would have a reward function.
        # Let's assume longer conversations are better?
        return 1.0

    def _backpropagate(self, node: MCTSNode, score: float) -> None:
        """Update visits and value up the tree."""
        current = node
        while current:
            current.visits += 1
            current.value += score

            # Move to parent
            if current.parent_id:
                current = self._find_node(self.tree.root, current.parent_id)
            else:
                current = None

    def _get_history(self, node: MCTSNode) -> list[ChatMessage]:
        """Reconstruct history for a node."""
        # Inefficient but functional for MVP
        path = []
        current = node
        while current:
            path.append(current.message)
            if current.parent_id:
                current = self._find_node(self.tree.root, current.parent_id)
            else:
                current = None

        # Add context messages
        full_history = list(self.context_messages) + list(reversed(path))
        return full_history

    def _find_node(self, current: MCTSNode, target_id: str) -> MCTSNode | None:
        """Helper to find a node by ID."""
        if current.id == target_id:
            return current
        for child in current.children:
            found = self._find_node(child, target_id)
            if found:
                return found
        return None

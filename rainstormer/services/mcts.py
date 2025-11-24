import json
import math
import asyncio
from collections import Counter

from ..schema.llm.message import ChatMessage, ChatRole
from ..schema.mcts import MCTSNode, MCTSTree, TurnSummary
from ..schema.scoring import TurnScore, FinalScore, SimulationResult, EarlyTermination
from ..schema.structured_output import (
    PCBResponse,
    TurnRole,
    TURN_CONFIGS,
    MicroJudgeVerdict,
)
from ..services.chat import ChatModelService, ChatModelHyperparams, ModelConfig
from ..prompts.mcts import (
    GENERATION_SYSTEM_PROMPT,
    SIMULATION_USER_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    MICRO_JUDGE_SYSTEM_PROMPT,
    MICRO_JUDGE_USER_TEMPLATE,
    FINAL_JUDGE_SYSTEM_PROMPT,
    FINAL_JUDGE_USER_TEMPLATE,
    NOVELTY_JUDGE_SYSTEM_PROMPT,
    NOVELTY_JUDGE_USER_TEMPLATE,
)
from ..prompts.structured import (
    get_role_prompt,
    get_simulated_user_prompt,
    build_simulation_context,
    PCB_SCHEMA_INSTRUCTION,
)
from ..utils.logger import logger


class MCTSService:
    """Service for Monte Carlo Tree Search brainstorming."""

    def __init__(
        self,
        llm_service: ChatModelService,
        model_config: ModelConfig | None = None,
    ):
        self.llm_service = llm_service
        self.model_config = model_config or ModelConfig()
        self.tree: MCTSTree | None = None
        self.context_messages: list[ChatMessage] = []
        self._original_idea: str = ""

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
        self._original_idea = root_message.content or ""

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
                if len(children) >= 2:
                    winner_idx = await self._evaluate_pair(children[0], children[1])
                    simulation_node = children[winner_idx]
                else:
                    # Only one child was generated (e.g., provider doesn't support n>1)
                    simulation_node = children[0]
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
        Simulate a 6-turn conversation rollout with role choreography.

        Returns a score (0.0 to 1.0) based on multi-dimensional evaluation:
        - Per-turn micro-scores (structure, novelty, depth, engagement, progress)
        - Final judge evaluation (convergence, coverage, coherence, outcome)
        - Trajectory bonus for improvement over turns
        """
        logger.info(f"Simulating 6-turn exchange from node {node.id}")

        history = self._get_history(node)
        current_history = list(history)
        simulation_messages: list[ChatMessage] = []

        # Track scoring
        turn_scores: list[TurnScore] = []
        early_termination = EarlyTermination.NONE

        # Build compressed context for prompts
        compressed_context = self._build_compressed_context(node)

        # Run 6-turn simulation with role choreography
        for turn_config in TURN_CONFIGS:
            turn_num = turn_config.turn_number
            role = turn_config.role
            temperature = turn_config.temperature

            logger.debug(f"Simulation turn {turn_num}: {role.value}")

            try:
                # === Simulated User Turn ===
                user_prompt = get_simulated_user_prompt(turn_num)
                sim_user_messages = [
                    ChatMessage(role=ChatRole.SYSTEM, content=user_prompt)
                ] + current_history

                user_res = await self.llm_service._chat(
                    messages=sim_user_messages,
                    hyperparams=ChatModelHyperparams(temperature=0.7, max_tokens=150),
                    model_override=self.model_config.model_for_simulated_user,
                )
                user_msg = ChatMessage(
                    role=ChatRole.USER,
                    content=user_res.choices[0].message.content,
                )
                current_history.append(user_msg)
                simulation_messages.append(user_msg)

                # === Assistant Turn (with role) ===
                role_prompt = get_role_prompt(role)
                context_prompt = build_simulation_context(
                    original_idea=self._original_idea,
                    compressed_history=compressed_context,
                    last_message=user_msg.content or "",
                    role=role,
                )

                assistant_messages = [
                    ChatMessage(role=ChatRole.SYSTEM, content=role_prompt),
                    ChatMessage(role=ChatRole.USER, content=context_prompt),
                ]

                asst_res = await self.llm_service._chat(
                    messages=assistant_messages,
                    hyperparams=ChatModelHyperparams(
                        temperature=temperature,
                        max_tokens=250,
                    ),
                    model_override=self.model_config.model_for_generation,
                )
                asst_content = asst_res.choices[0].message.content or ""
                asst_msg = ChatMessage(role=ChatRole.ASSISTANT, content=asst_content)
                current_history.append(asst_msg)
                simulation_messages.append(asst_msg)

                # === Parse PCB Response ===
                pcb_response = self._parse_pcb_response(asst_content)

                # === Micro-Judge Scoring ===
                turn_score = await self._score_turn(
                    history=current_history[:-1],
                    current_turn=asst_content,
                    turn_number=turn_num,
                )
                turn_scores.append(turn_score)

                # === Update Node Context ===
                if pcb_response:
                    turn_summary = TurnSummary(
                        turn_number=turn_num,
                        role=role.value,
                        key_proposal=(
                            pcb_response.proposal[:200]
                            if pcb_response.proposal
                            else None
                        ),
                        cons_identified=pcb_response.cons[:2],
                        benefits_claimed=pcb_response.benefits[:2],
                        token_count=len(asst_content.split()),
                    )
                    node.context.turn_summaries.append(turn_summary)

                # === Check Early Termination ===
                early_termination = self._check_early_termination(turn_scores)
                if early_termination != EarlyTermination.NONE:
                    logger.info(
                        f"Early termination at turn {turn_num}: {early_termination.value}"
                    )
                    break

                # Update compressed context for next turn
                compressed_context = self._build_compressed_context(node)

            except Exception as e:
                logger.error(f"Simulation turn {turn_num} failed: {e}")
                early_termination = EarlyTermination.ERROR
                break

        # === Final Judge Evaluation ===
        final_score = None
        if len(turn_scores) >= 3:
            final_score = await self._score_final(
                original_idea=self._original_idea,
                conversation=simulation_messages,
            )

        # === Build Simulation Result ===
        result = SimulationResult(
            turn_scores=turn_scores,
            final_score=final_score,
            early_termination=early_termination,
            exchanges_completed=len(turn_scores),
        )

        # Store simulation data in node metadata
        node.metadata["simulation_result"] = result.model_dump()

        logger.info(
            f"Simulation complete: {len(turn_scores)} turns, "
            f"reward={result.final_reward:.3f}, "
            f"termination={early_termination.value}"
        )

        return result.final_reward

    def _parse_pcb_response(self, content: str) -> PCBResponse | None:
        """Parse and validate PCB-format response."""
        try:
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            data = json.loads(content)
            return PCBResponse(**data)
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"Failed to parse PCB response: {e}")
            return None

    async def _score_turn(
        self,
        history: list[ChatMessage],
        current_turn: str,
        turn_number: int,
    ) -> TurnScore:
        """Score a single turn using LLM micro-judge."""
        # Format history for prompt
        history_str = "\n".join(
            f"{m.role.value.upper()}: {m.content}" for m in history[-6:]
        )

        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content=MICRO_JUDGE_SYSTEM_PROMPT),
            ChatMessage(
                role=ChatRole.USER,
                content=MICRO_JUDGE_USER_TEMPLATE.format(
                    history=history_str,
                    current_turn=current_turn,
                ),
            ),
        ]

        try:
            res = await self.llm_service._chat(
                messages=messages,
                hyperparams=ChatModelHyperparams(temperature=0.1, max_tokens=200),
                model_override=self.model_config.model_for_micro_judge,
            )

            content = res.choices[0].message.content or "{}"
            # Clean markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            data = json.loads(content)
            return TurnScore(
                turn_number=turn_number,
                structure=data.get("structure", 0.5),
                novelty=data.get("novelty", 0.5),
                depth=data.get("depth", 0.5),
                engagement=data.get("engagement", 0.5),
                progress=data.get("progress", 0.5),
            )
        except Exception as e:
            logger.warning(f"Micro-judge failed, using neutral scores: {e}")
            return TurnScore(
                turn_number=turn_number,
                structure=0.5,
                novelty=0.5,
                depth=0.5,
                engagement=0.5,
                progress=0.5,
            )

    async def _score_final(
        self,
        original_idea: str,
        conversation: list[ChatMessage],
    ) -> FinalScore:
        """Score the complete conversation using final judge."""
        conversation_str = "\n\n".join(
            f"[{m.role.value.upper()}]: {m.content}" for m in conversation
        )

        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content=FINAL_JUDGE_SYSTEM_PROMPT),
            ChatMessage(
                role=ChatRole.USER,
                content=FINAL_JUDGE_USER_TEMPLATE.format(
                    original_idea=original_idea,
                    conversation=conversation_str,
                ),
            ),
        ]

        try:
            res = await self.llm_service._chat(
                messages=messages,
                hyperparams=ChatModelHyperparams(temperature=0.1, max_tokens=300),
                model_override=self.model_config.model_for_final_judge,
            )

            content = res.choices[0].message.content or "{}"
            # Clean markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            data = json.loads(content)
            return FinalScore(
                convergence=data.get("convergence", 0.5),
                coverage=data.get("coverage", 0.5),
                coherence=data.get("coherence", 0.5),
                outcome_quality=data.get("outcome_quality", 0.5),
                reasoning=data.get("reasoning", ""),
            )
        except Exception as e:
            logger.warning(f"Final judge failed, using neutral scores: {e}")
            return FinalScore(
                convergence=0.5,
                coverage=0.5,
                coherence=0.5,
                outcome_quality=0.5,
                reasoning="Judge evaluation failed",
            )

    def _check_early_termination(
        self,
        turn_scores: list[TurnScore],
    ) -> EarlyTermination:
        """Check if simulation should terminate early."""
        if len(turn_scores) < 3:
            return EarlyTermination.NONE

        recent = turn_scores[-3:]

        # Low quality: 3 consecutive turns below 0.3
        if all(ts.weighted_total < 0.3 for ts in recent):
            return EarlyTermination.LOW_QUALITY

        # Plateau: no meaningful improvement for 3 turns
        if len(turn_scores) >= 3:
            deltas = [
                turn_scores[i].weighted_total - turn_scores[i - 1].weighted_total
                for i in range(len(turn_scores) - 2, len(turn_scores))
            ]
            if all(abs(d) < 0.05 for d in deltas):
                return EarlyTermination.PLATEAU

        return EarlyTermination.NONE

    def _build_compressed_context(self, node: MCTSNode) -> str:
        """Build compressed context string from node's turn summaries."""
        if not node.context.turn_summaries:
            return node.context.accumulated_summary or "No prior context."

        parts = []
        # Only use last 4 turn summaries to limit context
        for ts in node.context.turn_summaries[-4:]:
            parts.append(f"[Turn {ts.turn_number} - {ts.role}]")
            if ts.key_proposal:
                parts.append(f"  Proposal: {ts.key_proposal}")
            if ts.cons_identified:
                parts.append(f"  Cons: {', '.join(ts.cons_identified)}")
            if ts.benefits_claimed:
                parts.append(f"  Benefits: {', '.join(ts.benefits_claimed)}")

        return "\n".join(parts) if parts else "No prior context."

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

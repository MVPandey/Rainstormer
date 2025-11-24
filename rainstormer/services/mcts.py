import json
import math
import asyncio
from collections import Counter

from ..schema.llm.message import ChatMessage, ChatRole
from ..schema.mcts import MCTSNode, MCTSTree, TurnSummary
from ..schema.scoring import (
    TurnScore,
    FinalScore,
    NoveltyScore,
    SimulationResult,
    EarlyTermination,
)
from ..schema.structured_output import (
    PCBResponse,
    TURN_CONFIGS,
)
from ..services.chat import ChatModelService, ChatModelHyperparams, ModelConfig
from ..scoring.reward import RewardService
from ..prompts.mcts import (
    GENERATION_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
)
from ..prompts.structured import (
    get_role_prompt,
    get_simulated_user_prompt,
    build_simulation_context,
)
from ..tools import BRAINSTORMING_TOOLS, execute_tool, summarize_tool_result
from .context import ContextService, get_exploration_coefficient
from ..utils.logger import logger

# Maximum tool calls per simulation to limit costs
MAX_TOOL_CALLS_PER_SIMULATION = 4


class MCTSService:
    """Service for Monte Carlo Tree Search brainstorming."""

    def __init__(
        self,
        llm_service: ChatModelService,
        model_config: ModelConfig | None = None,
    ):
        self.llm_service = llm_service
        self.model_config = model_config or ModelConfig()
        self.reward_service = RewardService(llm_service, self.model_config)
        self.context_service = ContextService(llm_service, self.model_config)
        self.tree: MCTSTree | None = None
        self.context_messages: list[ChatMessage] = []
        self._original_idea: str = ""
        self._current_iteration: int = 0
        self._total_iterations: int = 0

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
        self._total_iterations = iterations

        for i in range(iterations):
            self._current_iteration = i
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
        Select a node to expand using UCT with adaptive exploration.

        The exploration coefficient varies based on search progress:
        - Early iterations (< 30%): C = 2.0 (explore widely)
        - Middle iterations (30-70%): C = 1.414 (balanced)
        - Late iterations (> 70%): C = 0.8 (exploit best paths)

        Traverse down until we find a leaf or a node with unexplored children.
        """
        # Get adaptive exploration coefficient
        exploration_c = get_exploration_coefficient(
            self._current_iteration,
            self._total_iterations,
        )

        current = node
        while current.children:
            # Calculate UCT for all children
            # UCT = value/visits + C * sqrt(ln(parent_visits) / visits)
            best_child = None
            best_score = float("-inf")

            log_parent_visits = math.log(current.visits) if current.visits > 0 else 0

            for child in current.children:
                if child.visits == 0:
                    # Prioritize unvisited nodes
                    return child

                uct = (child.value / child.visits) + exploration_c * math.sqrt(
                    log_parent_visits / child.visits
                )
                if uct > best_score:
                    best_score = uct
                    best_child = child

            current = best_child

        return current

    async def _expand(self, node: MCTSNode) -> list[MCTSNode]:
        """Generate 2 candidate responses with novelty scoring."""
        logger.info(f"Expanding node {node.id}")

        # Get history for context
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

            # Score novelty for each new node against existing siblings
            await self._score_novelty_for_children(node, new_nodes)

            return new_nodes

        except Exception as e:
            logger.error(f"Error expanding node: {e}")
            return []

    async def _score_novelty_for_children(
        self,
        parent: MCTSNode,
        new_children: list[MCTSNode],
    ) -> None:
        """
        Score novelty of new children against actual sibling nodes.

        Compares each new child against:
        1. Existing siblings from prior expansions of this parent
        2. Other children generated in the same expansion batch
        """
        # Get proposals from NEW children being added
        new_proposals = [(child.message.content or "")[:500] for child in new_children]

        # Get proposals from EXISTING siblings (children already attached to parent)
        existing_siblings = [c for c in parent.children if c not in new_children]
        existing_proposals = [
            (c.message.content or "")[:500] for c in existing_siblings
        ]

        for i, child in enumerate(new_children):
            # Compare against: existing siblings + other new children (not self)
            other_new = [p for j, p in enumerate(new_proposals) if j != i]
            sibling_summaries = existing_proposals + other_new

            novelty_score = await self.reward_service.score_novelty(
                new_proposal=new_proposals[i],
                sibling_summaries=sibling_summaries,
            )

            # Store novelty in child metadata
            child.metadata["novelty_score"] = novelty_score.novelty_score
            child.metadata["novelty_reasoning"] = novelty_score.reasoning

            logger.debug(
                f"Novelty for child {child.id}: {novelty_score.novelty_score:.2f} "
                f"(compared against {len(sibling_summaries)} siblings)"
            )

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

        # Track scoring and tool usage
        turn_scores: list[TurnScore] = []
        early_termination = EarlyTermination.NONE
        tool_calls_made = 0
        tool_results_context: list[str] = []

        # Build compressed context for prompts
        compressed_context = self._build_compressed_context(node)

        # Run 6-turn simulation with role choreography
        for turn_config in TURN_CONFIGS:
            turn_num = turn_config.turn_number
            role = turn_config.role
            temperature = turn_config.temperature
            allow_tools = turn_config.allow_tools

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

                # === Assistant Turn (with role and optional tools) ===
                role_prompt = get_role_prompt(role)

                # Include tool results in context if any
                tool_context = ""
                if tool_results_context:
                    tool_context = "\n\nTOOL RESULTS:\n" + "\n".join(
                        tool_results_context
                    )

                context_prompt = build_simulation_context(
                    original_idea=self._original_idea,
                    compressed_history=compressed_context + tool_context,
                    last_message=user_msg.content or "",
                    role=role,
                )

                assistant_messages = [
                    ChatMessage(role=ChatRole.SYSTEM, content=role_prompt),
                    ChatMessage(role=ChatRole.USER, content=context_prompt),
                ]

                # Determine if tools should be offered this turn
                tools_for_turn = None
                if allow_tools and tool_calls_made < MAX_TOOL_CALLS_PER_SIMULATION:
                    tools_for_turn = BRAINSTORMING_TOOLS

                asst_res = await self.llm_service._chat(
                    messages=assistant_messages,
                    hyperparams=ChatModelHyperparams(
                        temperature=temperature,
                        max_tokens=250,
                    ),
                    model_override=self.model_config.model_for_generation,
                    tools=tools_for_turn,
                )

                # Handle tool calls if any
                choice = asst_res.choices[0]
                if (
                    choice.message.tool_calls
                    and tool_calls_made < MAX_TOOL_CALLS_PER_SIMULATION
                ):
                    for tool_call in choice.message.tool_calls:
                        if tool_calls_made >= MAX_TOOL_CALLS_PER_SIMULATION:
                            break

                        tool_name = tool_call.function.name
                        try:
                            args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            args = {}

                        logger.debug(f"Executing tool: {tool_name} with args: {args}")
                        result = await execute_tool(tool_name, args)
                        summary = summarize_tool_result(tool_name, result)
                        tool_results_context.append(summary)
                        tool_calls_made += 1

                        # Store in node context
                        node.context.tool_results[tool_name] = summary

                asst_content = choice.message.content or ""
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

        # === Get Novelty Score from node metadata (set during expansion) ===
        novelty_score = None
        if "novelty_score" in node.metadata:
            novelty_score = NoveltyScore(
                novelty_score=node.metadata.get("novelty_score", 0.5),
                reasoning=node.metadata.get("novelty_reasoning", ""),
            )

        # === Build Simulation Result ===
        result = SimulationResult(
            turn_scores=turn_scores,
            final_score=final_score,
            novelty_score=novelty_score,
            early_termination=early_termination,
            exchanges_completed=len(turn_scores),
            tool_calls_made=tool_calls_made,
        )

        # Store simulation data in node metadata
        node.metadata["simulation_result"] = result.model_dump()

        logger.info(
            f"Simulation complete: {len(turn_scores)} turns, "
            f"reward={result.final_reward:.3f}, "
            f"novelty={novelty_score.novelty_score if novelty_score else 'N/A'}, "
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
        """Delegate turn scoring to RewardService."""
        return await self.reward_service.score_turn(
            history=history,
            current_turn=current_turn,
            turn_number=turn_number,
        )

    async def _score_final(
        self,
        original_idea: str,
        conversation: list[ChatMessage],
    ) -> FinalScore:
        """Delegate final scoring to RewardService."""
        return await self.reward_service.score_final(
            original_idea=original_idea,
            conversation=conversation,
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

"""Reward service for computing multi-dimensional MCTS rewards."""

import json

from ..schema.llm.message import ChatMessage, ChatRole
from ..schema.scoring import (
    TurnScore,
    FinalScore,
    NoveltyScore,
    EarlyTermination,
)
from ..schema.mcts import MCTSNode
from ..services.chat import ChatModelService, ChatModelHyperparams, ModelConfig
from ..prompts.mcts import (
    MICRO_JUDGE_SYSTEM_PROMPT,
    MICRO_JUDGE_USER_TEMPLATE,
    FINAL_JUDGE_SYSTEM_PROMPT,
    FINAL_JUDGE_USER_TEMPLATE,
    NOVELTY_JUDGE_SYSTEM_PROMPT,
    NOVELTY_JUDGE_USER_TEMPLATE,
)
from ..utils.logger import logger


class RewardService:
    """
    Service for computing multi-dimensional rewards for MCTS simulations.

    Handles:
    - Per-turn micro-scoring (structure, novelty, depth, engagement, progress)
    - Final conversation evaluation (convergence, coverage, coherence, outcome)
    - Novelty assessment vs. sibling branches
    - Reward aggregation with trajectory bonuses
    """

    def __init__(
        self,
        llm_service: ChatModelService,
        model_config: ModelConfig | None = None,
    ):
        self.llm_service = llm_service
        self.model_config = model_config or ModelConfig()

    async def score_turn(
        self,
        history: list[ChatMessage],
        current_turn: str,
        turn_number: int,
    ) -> TurnScore:
        """
        Score a single simulation turn using LLM micro-judge.

        Args:
            history: Conversation history up to (but not including) current turn.
            current_turn: The assistant response to score.
            turn_number: Which turn this is (1-6).

        Returns:
            TurnScore with 5 dimensions scored 0-1.
        """
        # Format history for prompt (last 6 messages for context)
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
            content = self._clean_json(content)

            data = json.loads(content)
            return TurnScore(
                turn_number=turn_number,
                structure=self._clamp(data.get("structure", 0.5)),
                novelty=self._clamp(data.get("novelty", 0.5)),
                depth=self._clamp(data.get("depth", 0.5)),
                engagement=self._clamp(data.get("engagement", 0.5)),
                progress=self._clamp(data.get("progress", 0.5)),
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

    async def score_final(
        self,
        original_idea: str,
        conversation: list[ChatMessage],
    ) -> FinalScore:
        """
        Evaluate the complete conversation using final judge.

        Args:
            original_idea: The brainstorming topic.
            conversation: Full list of simulation messages.

        Returns:
            FinalScore with 4 dimensions scored 0-1.
        """
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
            content = self._clean_json(content)

            data = json.loads(content)
            return FinalScore(
                convergence=self._clamp(data.get("convergence", 0.5)),
                coverage=self._clamp(data.get("coverage", 0.5)),
                coherence=self._clamp(data.get("coherence", 0.5)),
                outcome_quality=self._clamp(data.get("outcome_quality", 0.5)),
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

    async def score_novelty(
        self,
        new_proposal: str,
        sibling_summaries: list[str],
    ) -> NoveltyScore:
        """
        Compare a new proposal against existing sibling proposals.

        Args:
            new_proposal: The proposal to evaluate.
            sibling_summaries: Summaries of proposals from sibling nodes.

        Returns:
            NoveltyScore with novelty rating 0-1.
        """
        if not sibling_summaries:
            # No siblings to compare against - maximum novelty
            return NoveltyScore(
                novelty_score=1.0,
                reasoning="First proposal in this branch - inherently novel.",
            )

        # Format sibling summaries
        sibling_str = "\n".join(
            f"{i + 1}. {summary}" for i, summary in enumerate(sibling_summaries)
        )

        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content=NOVELTY_JUDGE_SYSTEM_PROMPT),
            ChatMessage(
                role=ChatRole.USER,
                content=NOVELTY_JUDGE_USER_TEMPLATE.format(
                    sibling_summaries=sibling_str,
                    new_proposal=new_proposal,
                ),
            ),
        ]

        try:
            res = await self.llm_service._chat(
                messages=messages,
                hyperparams=ChatModelHyperparams(temperature=0.1, max_tokens=150),
                model_override=self.model_config.model_for_novelty_judge,
            )

            content = res.choices[0].message.content or "{}"
            content = self._clean_json(content)

            data = json.loads(content)
            return NoveltyScore(
                novelty_score=self._clamp(data.get("novelty_score", 0.5)),
                reasoning=data.get("reasoning", ""),
            )
        except Exception as e:
            logger.warning(f"Novelty judge failed, using neutral score: {e}")
            return NoveltyScore(
                novelty_score=0.5,
                reasoning="Novelty evaluation failed",
            )

    def aggregate_reward(
        self,
        turn_scores: list[TurnScore],
        final_score: FinalScore | None,
        novelty_score: NoveltyScore | None = None,
        early_termination: EarlyTermination = EarlyTermination.NONE,
    ) -> float:
        """
        Combine all scoring signals into a final [0,1] bounded reward.

        Formula:
        R = w_micro * mean(turn_scores) +
            w_trajectory * trajectory_bonus +
            w_final * final_judge +
            w_novelty * novelty_score +
            termination_adjustment

        Args:
            turn_scores: Per-turn micro-scores.
            final_score: Terminal evaluation (optional).
            novelty_score: Novelty vs. siblings (optional).
            early_termination: Reason for early termination.

        Returns:
            Reward bounded to [0, 1].
        """
        # Early termination adjustments
        termination_adjustments = {
            EarlyTermination.NONE: 0.0,
            EarlyTermination.LOW_QUALITY: -0.3,
            EarlyTermination.PLATEAU: -0.1,
            EarlyTermination.REPETITION: -0.3,
            EarlyTermination.CONVERGENCE: 0.1,  # Bonus for natural conclusion
            EarlyTermination.ERROR: -0.5,
        }

        # Calculate micro-score component (weight: 0.35)
        if turn_scores:
            micro_avg = sum(ts.weighted_total for ts in turn_scores) / len(turn_scores)
        else:
            micro_avg = 0.0

        # Trajectory bonus (weight: 0.15)
        trajectory = self._compute_trajectory_bonus(turn_scores)

        # Final judge score (weight: 0.35)
        final_judge = final_score.score if final_score else 0.5

        # Novelty component (weight: 0.15)
        novelty = novelty_score.novelty_score if novelty_score else 0.5

        # Weighted combination
        raw_reward = (
            0.35 * micro_avg + 0.15 * trajectory + 0.35 * final_judge + 0.15 * novelty
        )

        # Apply termination adjustment
        adjustment = termination_adjustments.get(early_termination, 0.0)
        raw_reward += adjustment

        # Clamp to [0, 1]
        return max(0.0, min(1.0, raw_reward))

    def _compute_trajectory_bonus(self, turn_scores: list[TurnScore]) -> float:
        """Calculate trajectory bonus based on improvement over turns."""
        if len(turn_scores) < 2:
            return 0.5  # Neutral if not enough data

        deltas = [
            turn_scores[i + 1].weighted_total - turn_scores[i].weighted_total
            for i in range(len(turn_scores) - 1)
        ]
        positive_deltas = [d for d in deltas if d > 0]

        if not positive_deltas:
            return 0.3  # Some penalty for no improvement

        # Average positive improvement, scaled to [0, 1]
        avg_improvement = sum(positive_deltas) / len(positive_deltas)
        # Scale: 0.1 improvement = full bonus
        return min(1.0, 0.5 + avg_improvement * 5)

    def get_sibling_summaries(self, node: MCTSNode) -> list[str]:
        """
        Get proposal summaries from sibling nodes for novelty comparison.

        Args:
            node: The node to find siblings for.

        Returns:
            List of proposal summaries from siblings.
        """
        # This would need access to the tree to find siblings
        # For now, return proposals from node's context if available
        summaries = []

        # Check if node has simulation result with proposals
        if "simulation_result" in node.metadata:
            result = node.metadata["simulation_result"]
            if isinstance(result, dict) and "turn_scores" in result:
                # We don't store proposals in turn_scores, so check context
                pass

        # Get from context turn summaries
        for ts in node.context.turn_summaries:
            if ts.key_proposal:
                summaries.append(ts.key_proposal)

        return summaries

    @staticmethod
    def _clean_json(content: str) -> str:
        """Clean markdown code blocks from JSON content."""
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return content

    @staticmethod
    def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Clamp a value to [min_val, max_val]."""
        try:
            return max(min_val, min(max_val, float(value)))
        except (TypeError, ValueError):
            return 0.5  # Default neutral value

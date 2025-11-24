"""Context compression and management service for MCTS."""

from __future__ import annotations

import json

from ..schema.llm.message import ChatMessage, ChatRole
from ..schema.mcts import MCTSNode, MCTSNodeContext, TurnSummary
from ..services.chat import ChatModelService, ChatModelHyperparams, ModelConfig
from ..utils.logger import logger

# Token budget constants
MAX_ACCUMULATED_SUMMARY_TOKENS = 300
MAX_TURN_SUMMARIES_IN_CONTEXT = 4
MAX_TURN_SUMMARY_TOKENS = 60

DISTILLATION_PROMPT = """Summarize this turn into a structured JSON format:

TURN CONTENT:
{turn_content}

Output ONLY valid JSON with this structure:
{{
    "key_proposal": "Main idea in 1 sentence (max 30 tokens)",
    "cons": ["Top concern 1", "Top concern 2"],
    "benefits": ["Top benefit 1", "Top benefit 2"],
    "unresolved": "Any open question (or null)"
}}"""


class ContextService:
    """
    Service for managing context compression and token budgets.

    Provides O(1) context retrieval instead of O(depth) history reconstruction.
    """

    def __init__(
        self,
        llm_service: ChatModelService,
        model_config: ModelConfig | None = None,
    ):
        self.llm_service = llm_service
        self.model_config = model_config or ModelConfig()

    async def distill_turn(
        self,
        turn_content: str,
        turn_number: int,
        role: str,
    ) -> TurnSummary:
        """
        Distill a turn's content into a compressed summary.

        Uses LLM to extract key information in a token-efficient format.

        Args:
            turn_content: Full content of the assistant's turn.
            turn_number: Turn number (1-6).
            role: Role that generated this turn.

        Returns:
            TurnSummary with extracted key information.
        """
        prompt = DISTILLATION_PROMPT.format(turn_content=turn_content[:1000])

        messages = [
            ChatMessage(
                role=ChatRole.SYSTEM,
                content="You are a precise summarizer. Output ONLY valid JSON.",
            ),
            ChatMessage(role=ChatRole.USER, content=prompt),
        ]

        try:
            response = await self.llm_service._chat(
                messages=messages,
                hyperparams=ChatModelHyperparams(temperature=0.1, max_tokens=150),
                model_override=self.model_config.model_for_judge,
            )

            content = response.choices[0].message.content or "{}"

            # Parse JSON response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            data = json.loads(content)

            return TurnSummary(
                turn_number=turn_number,
                role=role,
                key_proposal=data.get("key_proposal", "")[:200],
                cons_identified=data.get("cons", [])[:3],
                benefits_claimed=data.get("benefits", [])[:3],
                token_count=len(turn_content.split()),
            )

        except Exception as e:
            logger.warning(f"Turn distillation failed: {e}")
            # Fallback: simple truncation
            return TurnSummary(
                turn_number=turn_number,
                role=role,
                key_proposal=turn_content[:200],
                cons_identified=[],
                benefits_claimed=[],
                token_count=len(turn_content.split()),
            )

    def build_compressed_context(
        self,
        node: MCTSNode,
        include_tool_results: bool = True,
    ) -> str:
        """
        Build compressed context string from node's context.

        Uses pre-computed turn summaries instead of full history.

        Args:
            node: The MCTS node to build context for.
            include_tool_results: Whether to include tool result summaries.

        Returns:
            Compressed context string suitable for prompts.
        """
        context = node.context
        parts = []

        # Add accumulated summary if present
        if context.accumulated_summary:
            parts.append(f"[PRIOR CONTEXT]\n{context.accumulated_summary}")

        # Add recent turn summaries (last N only)
        summaries = context.turn_summaries[-MAX_TURN_SUMMARIES_IN_CONTEXT:]
        if summaries:
            parts.append("\n[RECENT TURNS]")
            for ts in summaries:
                turn_part = [f"Turn {ts.turn_number} ({ts.role}):"]
                if ts.key_proposal:
                    turn_part.append(f"  Proposal: {ts.key_proposal}")
                if ts.cons_identified:
                    turn_part.append(f"  Cons: {', '.join(ts.cons_identified)}")
                if ts.benefits_claimed:
                    turn_part.append(f"  Benefits: {', '.join(ts.benefits_claimed)}")
                parts.append("\n".join(turn_part))

        # Add tool results if requested
        if include_tool_results and context.tool_results:
            parts.append("\n[TOOL RESULTS]")
            for tool_name, result in list(context.tool_results.items())[:4]:
                parts.append(f"{tool_name}: {result[:200]}")

        return "\n".join(parts) if parts else "No prior context."

    def update_accumulated_summary(
        self,
        context: MCTSNodeContext,
        new_turn_summary: TurnSummary,
    ) -> None:
        """
        Update the rolling accumulated summary with a new turn.

        Maintains a fixed token budget by truncating older content.

        Args:
            context: Node context to update.
            new_turn_summary: New turn summary to incorporate.
        """
        # Simple strategy: append key proposal to accumulated summary
        # and truncate if too long
        if new_turn_summary.key_proposal:
            addition = (
                f"T{new_turn_summary.turn_number}: {new_turn_summary.key_proposal}. "
            )
            context.accumulated_summary += addition

            # Rough token estimate (4 chars per token)
            if len(context.accumulated_summary) > MAX_ACCUMULATED_SUMMARY_TOKENS * 4:
                # Keep last N characters
                context.accumulated_summary = context.accumulated_summary[
                    -(MAX_ACCUMULATED_SUMMARY_TOKENS * 4) :
                ]

    def get_context_token_estimate(self, context: MCTSNodeContext) -> int:
        """
        Estimate total tokens used by context.

        Args:
            context: Node context to estimate.

        Returns:
            Estimated token count.
        """
        total = 0

        # Accumulated summary
        total += len(context.accumulated_summary.split())

        # Turn summaries
        for ts in context.turn_summaries[-MAX_TURN_SUMMARIES_IN_CONTEXT:]:
            total += ts.token_count // 4  # Compressed form

        # Tool results
        for result in context.tool_results.values():
            total += len(result.split())

        return total

    def is_within_budget(self, context: MCTSNodeContext) -> bool:
        """
        Check if context is within token budget.

        Args:
            context: Node context to check.

        Returns:
            True if within budget, False otherwise.
        """
        estimate = self.get_context_token_estimate(context)
        return estimate <= context.budget_remaining


# =============================================================================
# UCB1 Exploration Coefficient Scheduling
# =============================================================================


def get_exploration_coefficient(
    current_iteration: int,
    total_iterations: int,
) -> float:
    """
    Get adaptive exploration coefficient based on search progress.

    Early iterations favor exploration (high coefficient).
    Late iterations favor exploitation (low coefficient).

    Args:
        current_iteration: Current MCTS iteration (0-indexed).
        total_iterations: Total planned iterations.

    Returns:
        Exploration coefficient for UCB1 calculation.
    """
    if total_iterations <= 0:
        return 1.414  # Default

    progress = current_iteration / total_iterations

    if progress < 0.3:
        return 2.0  # Explore widely early
    elif progress < 0.7:
        return 1.414  # Standard UCB1
    else:
        return 0.8  # Exploit late


def get_exploration_coefficient_linear(
    current_iteration: int,
    total_iterations: int,
    c_start: float = 2.0,
    c_end: float = 0.8,
) -> float:
    """
    Get exploration coefficient with linear decay.

    Provides smoother transition than step function.

    Args:
        current_iteration: Current MCTS iteration (0-indexed).
        total_iterations: Total planned iterations.
        c_start: Starting coefficient (high exploration).
        c_end: Ending coefficient (high exploitation).

    Returns:
        Exploration coefficient for UCB1 calculation.
    """
    if total_iterations <= 1:
        return (c_start + c_end) / 2

    progress = current_iteration / (total_iterations - 1)
    return c_start + progress * (c_end - c_start)

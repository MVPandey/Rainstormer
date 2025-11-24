from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .llm.message import ChatMessage


class TurnSummary(BaseModel):
    """Distilled representation of a single simulation turn."""

    turn_number: int = Field(ge=1, le=6, description="Turn number (1-6)")
    role: str = Field(description="Role that generated this turn")
    key_proposal: str | None = Field(
        default=None, max_length=200, description="Core proposal (max 50 tokens)"
    )
    cons_identified: list[str] = Field(
        default_factory=list, description="Top cons mentioned (max 3)"
    )
    benefits_claimed: list[str] = Field(
        default_factory=list, description="Top benefits mentioned (max 3)"
    )
    token_count: int = Field(default=0, description="Estimated token count")
    tool_used: str | None = Field(default=None, description="Tool called, if any")


class DiversityTag(BaseModel):
    """Semantic tag for novelty tracking across branches."""

    category: str = Field(
        description="Category: technical, market, user, risk, implementation, etc."
    )
    concept: str = Field(description="The specific idea or angle explored")
    novelty_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="How different from siblings (0-1)"
    )


class MCTSNodeContext(BaseModel):
    """
    Rich context stored per-node for efficient history access.

    Replaces O(depth) history reconstruction with compressed summaries.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Running summary (replaces full history reconstruction)
    accumulated_summary: str = Field(
        default="", max_length=1200, description="Rolling summary (max 300 tokens)"
    )
    turn_summaries: list[TurnSummary] = Field(
        default_factory=list, description="Per-turn distilled summaries"
    )

    # Diversity tracking for novelty
    diversity_tags: list[DiversityTag] = Field(
        default_factory=list, description="Semantic tags for this branch"
    )
    explored_angles: list[str] = Field(
        default_factory=list, description="Concepts explored in this branch"
    )

    # Token budget tracking
    total_context_tokens: int = Field(
        default=0, description="Total tokens used in context"
    )
    budget_remaining: int = Field(
        default=2000, description="Remaining token budget for this node"
    )

    # Tool results cache (store separately, retrieve on demand)
    tool_results: dict[str, str] = Field(
        default_factory=dict, description="Tool call ID -> summarized result"
    )


class MCTSNode(BaseModel):
    """Represents a node in the MCTS tree."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str | None = None
    children: list[MCTSNode] = Field(default_factory=list)
    message: ChatMessage
    visits: int = 0
    value: float = 0.0
    depth: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Rich context for efficient history (optional, for enhanced nodes)
    context: MCTSNodeContext = Field(default_factory=MCTSNodeContext)

    def add_child(self, child: MCTSNode) -> None:
        """Add a child node."""
        child.parent_id = self.id
        self.children.append(child)

    def uct_score(self, exploration_weight: float = 1.414) -> float:
        """Calculate UCT score for selection."""
        if self.visits == 0:
            return float("inf")

        # Note: This needs the parent's visits, which we don't strictly store on the node
        # without a back-reference object.
        # In a simple implementation, we can pass parent_visits as an arg,
        # or just use a simplified selection logic in the service.
        # For now, we'll leave the calculation to the service or update this signature later.
        return self.value / self.visits


class MCTSTree(BaseModel):
    """Represents the MCTS tree."""

    root: MCTSNode

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from .llm.message import ChatMessage


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

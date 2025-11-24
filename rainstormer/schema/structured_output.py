"""Structured output schemas for MCTS simulation."""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field


class TurnRole(str, enum.Enum):
    """Roles for each turn in 6-exchange simulation."""

    RESEARCHER = "researcher"
    SKEPTIC = "skeptic"
    BUILDER = "builder"
    SYNTHESIZER = "synthesizer"
    CRITIC = "critic"
    FINALIZER = "finalizer"


class PCBResponse(BaseModel):
    """
    Structured response format: Proposal-Cons-Benefits.

    Every simulation turn should produce this format for
    consistent evaluation and information density.
    """

    proposal: str = Field(
        max_length=500,
        description="Main idea or response (target: 100 tokens max)",
    )
    cons: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Up to 3 potential downsides or risks",
    )
    benefits: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Up to 3 advantages or strengths",
    )
    follow_up_question: str | None = Field(
        default=None,
        max_length=200,
        description="Optional question to explore further",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Self-assessed confidence in this proposal (0-1)",
    )

    @property
    def is_well_formed(self) -> bool:
        """Check if response has minimum required structure."""
        return bool(self.proposal) and (len(self.cons) > 0 or len(self.benefits) > 0)


class JudgeVerdict(BaseModel):
    """Structured output for pairwise evaluation."""

    reasoning: str = Field(
        max_length=300,
        description="Brief explanation of choice (target: 50 tokens)",
    )
    winner_index: int = Field(
        ge=0, le=1, description="0 for first response, 1 for second"
    )
    creativity_score_a: float = Field(
        ge=0.0, le=1.0, description="Creativity score for response A"
    )
    creativity_score_b: float = Field(
        ge=0.0, le=1.0, description="Creativity score for response B"
    )
    actionability_score_a: float = Field(
        ge=0.0, le=1.0, description="Actionability score for response A"
    )
    actionability_score_b: float = Field(
        ge=0.0, le=1.0, description="Actionability score for response B"
    )


class MicroJudgeVerdict(BaseModel):
    """Per-turn micro-scoring output from judge."""

    structure: float = Field(ge=0.0, le=1.0, description="PCB format adherence (0-1)")
    novelty: float = Field(
        ge=0.0, le=1.0, description="New concepts vs. repetition (0-1)"
    )
    depth: float = Field(
        ge=0.0, le=1.0, description="Specificity and actionability (0-1)"
    )
    engagement: float = Field(
        ge=0.0, le=1.0, description="User engagement quality (0-1)"
    )
    progress: float = Field(
        ge=0.0, le=1.0, description="Forward movement in brainstorm (0-1)"
    )
    reasoning: str = Field(default="", max_length=200, description="Brief explanation")


class SimulationTurnConfig(BaseModel):
    """Configuration for a single turn in simulation."""

    turn_number: int = Field(ge=1, le=6, description="Turn number (1-6)")
    role: TurnRole = Field(description="Role for this turn")
    temperature: float = Field(
        ge=0.0, le=2.0, description="LLM temperature for this turn"
    )
    max_tokens: int = Field(
        default=200, ge=50, le=500, description="Max tokens for response"
    )
    allow_tools: bool = Field(
        default=True, description="Whether tools are allowed this turn"
    )


TURN_CONFIGS: list[SimulationTurnConfig] = [
    SimulationTurnConfig(
        turn_number=1, role=TurnRole.RESEARCHER, temperature=0.9, allow_tools=True
    ),
    SimulationTurnConfig(
        turn_number=2, role=TurnRole.SKEPTIC, temperature=0.5, allow_tools=False
    ),
    SimulationTurnConfig(
        turn_number=3, role=TurnRole.BUILDER, temperature=0.7, allow_tools=True
    ),
    SimulationTurnConfig(
        turn_number=4, role=TurnRole.SYNTHESIZER, temperature=0.6, allow_tools=True
    ),
    SimulationTurnConfig(
        turn_number=5, role=TurnRole.CRITIC, temperature=0.4, allow_tools=False
    ),
    SimulationTurnConfig(
        turn_number=6, role=TurnRole.FINALIZER, temperature=0.3, allow_tools=False
    ),
]


class ToolResultSummary(BaseModel):
    """Summarized tool result for context inclusion."""

    tool_name: str = Field(description="Name of the tool called")
    query: str = Field(description="What was asked/computed")
    summary: str = Field(
        max_length=400, description="Summarized result (target: 200 tokens max)"
    )
    token_count: int = Field(default=0, description="Estimated token count")

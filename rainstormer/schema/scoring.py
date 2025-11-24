"""Scoring schemas for MCTS reward computation."""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field


class EarlyTermination(str, enum.Enum):
    """Reasons for early simulation termination."""

    NONE = "none"
    LOW_QUALITY = "low_quality"
    PLATEAU = "plateau"
    REPETITION = "repetition"
    CONVERGENCE = "convergence"
    ERROR = "error"


class TurnScore(BaseModel):
    """Per-turn micro-score for simulation evaluation."""

    turn_number: int = Field(ge=1, le=6, description="Turn number (1-6)")
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

    @property
    def weighted_total(self) -> float:
        """Calculate weighted total score using defined weights."""
        return (
            self.structure * 0.15
            + self.novelty * 0.15
            + self.depth * 0.25
            + self.engagement * 0.20
            + self.progress * 0.25
        )

    @property
    def average(self) -> float:
        """Simple average of all dimensions."""
        return (
            self.structure + self.novelty + self.depth + self.engagement + self.progress
        ) / 5.0


class FinalScore(BaseModel):
    """Terminal evaluation of complete simulation."""

    convergence: float = Field(
        ge=0.0, le=1.0, description="Did brainstorm reach actionable conclusions? (0-1)"
    )
    coverage: float = Field(
        ge=0.0, le=1.0, description="Were multiple angles explored? (0-1)"
    )
    coherence: float = Field(
        ge=0.0, le=1.0, description="Is the conversation logically consistent? (0-1)"
    )
    outcome_quality: float = Field(
        ge=0.0, le=1.0, description="How valuable is the final set of ideas? (0-1)"
    )
    reasoning: str = Field(default="", description="Brief explanation of scores")

    @property
    def score(self) -> float:
        """Calculate average final score."""
        return (
            self.convergence + self.coverage + self.coherence + self.outcome_quality
        ) / 4.0


class NoveltyScore(BaseModel):
    """Score for novelty comparison against siblings."""

    novelty_score: float = Field(
        ge=0.0, le=1.0, description="How novel is this proposal vs. siblings (0-1)"
    )
    reasoning: str = Field(default="", description="Brief explanation")


class SimulationResult(BaseModel):
    """Complete simulation result with all scoring data."""

    turn_scores: list[TurnScore] = Field(
        default_factory=list, description="Per-turn micro-scores"
    )
    final_score: FinalScore | None = Field(
        default=None, description="Terminal evaluation"
    )
    novelty_score: NoveltyScore | None = Field(
        default=None, description="Novelty vs. siblings"
    )
    early_termination: EarlyTermination = Field(
        default=EarlyTermination.NONE, description="Reason for early termination"
    )
    exchanges_completed: int = Field(
        default=0, ge=0, le=6, description="Number of exchanges completed"
    )
    tool_calls_made: int = Field(
        default=0, ge=0, description="Number of tool calls during simulation"
    )

    @property
    def trajectory_bonus(self) -> float:
        """Calculate trajectory bonus based on improvement over turns."""
        if len(self.turn_scores) < 2:
            return 0.0

        deltas = [
            self.turn_scores[i + 1].weighted_total - self.turn_scores[i].weighted_total
            for i in range(len(self.turn_scores) - 1)
        ]
        positive_deltas = [d for d in deltas if d > 0]

        if not positive_deltas:
            return 0.0

        return min(0.2, sum(positive_deltas) / len(positive_deltas))

    @property
    def final_reward(self) -> float:
        """
        Compute final [0,1] bounded reward.

        Formula:
        R = w_micro * mean(turn_scores) + w_trajectory * trajectory_bonus + w_final * final_judge
        """
        # Early termination penalties/bonuses
        termination_adjustments = {
            EarlyTermination.NONE: 0.0,
            EarlyTermination.LOW_QUALITY: -0.3,
            EarlyTermination.PLATEAU: -0.1,
            EarlyTermination.REPETITION: -0.3,
            EarlyTermination.CONVERGENCE: 0.1,
            EarlyTermination.ERROR: -0.5,
        }

        if self.turn_scores:
            micro_avg = sum(ts.weighted_total for ts in self.turn_scores) / len(
                self.turn_scores
            )
        else:
            micro_avg = 0.0

        trajectory = self.trajectory_bonus

        final_judge = self.final_score.score if self.final_score else 0.5

        raw_reward = 0.4 * micro_avg + 0.2 * trajectory + 0.4 * final_judge

        adjustment = termination_adjustments.get(self.early_termination, 0.0)
        raw_reward += adjustment

        return max(0.0, min(1.0, raw_reward))

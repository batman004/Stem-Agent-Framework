"""
Stem Agent State Models

The code (genome) is fixed. The expressed configuration (phenotype) evolves.
Self-modification means updating configuration state, NOT rewriting source code.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, TypedDict

from pydantic import BaseModel, Field


class AgentPhase(str, Enum):
    """Lifecycle phases of the stem agent."""
    PROBING      = "probing"
    ARCHITECTING = "architecting"
    EXECUTING    = "executing"
    EVALUATING   = "evaluating"
    BRANCHING    = "branching"
    COMPLETE     = "complete"


class TaskRecord(BaseModel):
    """Record of a single task execution."""
    task_id: str = ""
    task: dict[str, Any] = {}
    output: str = ""
    score: float = 0.0
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    iteration: int = 0


class SubProblemState(BaseModel):
    """State for a single sub-problem the stem agent has identified."""
    name: str
    description: str
    expert_workflow: list[str] = []
    required_capabilities: list[str] = []
    quality_rubric: str = ""

    tools_acquired: list[str] = []
    prompt_templates: dict[str, str] = {}
    agent_pattern: str = "react"

    competence_score: float = 0.0
    task_history: list[TaskRecord] = []
    iterations: int = 0

    specialist_id: str | None = None

    @property
    def is_branched(self) -> bool:
        return self.specialist_id is not None


class StemAgentState(BaseModel):
    """Complete state of the stem agent. Serializable for checkpointing and rollback."""

    task_class: str = ""
    task_class_description: str = ""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    phase: AgentPhase = AgentPhase.PROBING
    iteration: int = 0
    max_iterations: int = 20

    domain_model: dict[str, Any] = {}
    sub_problems: dict[str, SubProblemState] = {}

    branch_threshold: float = 0.75
    stop_threshold: float = 0.85

    checkpoints: list[dict[str, Any]] = []
    max_checkpoints: int = 5

    specialists: dict[str, str] = {}

    errors: list[str] = []
    warnings: list[str] = []
    log: list[str] = []

    def add_checkpoint(self) -> None:
        """Snapshot current state for rollback. Keeps last N checkpoints."""
        snapshot = self.model_dump(exclude={"checkpoints"})
        snapshot["_checkpoint_time"] = datetime.now(timezone.utc).isoformat()
        self.checkpoints.append(snapshot)
        if len(self.checkpoints) > self.max_checkpoints:
            self.checkpoints = self.checkpoints[-self.max_checkpoints:]

    def add_log(self, message: str) -> None:
        """Append a timestamped log entry."""
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {message}")

    @property
    def all_branched(self) -> bool:
        """True if every sub-problem has been branched to a specialist."""
        if not self.sub_problems:
            return False
        return all(sp.is_branched for sp in self.sub_problems.values())

    @property
    def any_ready_to_branch(self) -> bool:
        """True if any un-branched sub-problem has reached the branch threshold."""
        return any(
            sp.competence_score >= self.branch_threshold and not sp.is_branched
            for sp in self.sub_problems.values()
        )


class GraphState(TypedDict):
    """LangGraph requires TypedDict for state. Wraps our Pydantic model."""
    agent: StemAgentState

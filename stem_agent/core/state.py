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


class UserResource(BaseModel):
    """A user-provided resource for the stem agent to inspect autonomously.

    The user provides connection info (URL, credentials), NOT full schemas.
    The stem agent discovers the structure itself via inspection tools.
    """
    type: str = ""              # database, github_repo, github_pr, api, url, file_path
    url: str = ""
    label: str = ""
    credentials: dict[str, str] = {}   # Optional auth (tokens, passwords)
    discovered_schema: str = ""        # Populated by environment_probe inspection


class TaskRecord(BaseModel):
    """Record of a single task execution."""
    task_id: str = ""
    task: dict[str, Any] = {}
    output: str = ""
    score: float = 0.0
    eval_reasoning: str = ""
    tools_used: list[str] = []
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    iteration: int = 0


class BenchmarkTask(BaseModel):
    """A concrete, executable benchmark task for evaluating agent competence."""
    task_id: str = ""
    instruction: str = ""
    expected_approach: str = ""
    difficulty: str = "medium"  # easy, medium, hard
    requires_tools: list[str] = []
    success_criteria: str = ""


class SubProblemState(BaseModel):
    """State for a single sub-problem the stem agent has identified."""
    name: str
    description: str
    expert_workflow: list[str] = []
    required_capabilities: list[str] = []
    quality_rubric: str = ""

    tools_acquired: list[str] = []
    custom_tools: dict[str, str] = {}
    prompt_templates: dict[str, str] = {}
    agent_pattern: str = "react"

    benchmark_tasks: list[BenchmarkTask] = []
    competence_score: float = 0.0
    task_history: list[TaskRecord] = []
    eval_feedback: list[str] = []  # Accumulated feedback from competence tracker
    iterations: int = 0

    specialist_id: str | None = None

    @property
    def is_branched(self) -> bool:
        return self.specialist_id is not None

    def export_specialist_artifact(self, output_dir: str, grounding_resources: list[dict[str, Any]] | None = None) -> str:
        """Serialize the matured sub-problem state into a LangChain Agent JSON file."""
        import json
        from pathlib import Path
        
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"specialist_{self.name}_{self.specialist_id[:8] if self.specialist_id else 'dev'}.json"
        filepath = out_path / filename
        
        # Format as a LangChain Agent configurable JSON
        agent_config = {
            "name": self.name,
            "description": self.description,
            "agent_type": "zero-shot-react-description",
            "prompt_templates": self.prompt_templates,
            "tools": self.tools_acquired,
            "custom_tool_code": self.custom_tools,
            "metadata": {
                "competence_score": self.competence_score,
                "iterations_trained": self.iterations,
                "rubric": self.quality_rubric,
                "grounding_resources": grounding_resources or []
            }
        }
        
        with open(filepath, "w") as f:
            json.dump(agent_config, f, indent=4)
            
        return str(filepath)


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

    # Before/after evaluation
    baseline_scores: dict[str, float] = {}  # sub_problem_name → baseline score

    user_resources: list[UserResource] = []       # User-provided DB URLs, repo links, etc.
    resource_context: str = ""                    # Discovered schemas/structures, fed to LLM
    pending_clarifications: list[str] = []        # Questions to surface to the user

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

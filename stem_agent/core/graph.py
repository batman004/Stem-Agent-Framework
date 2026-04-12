"""
Stem Agent LangGraph Definition

State machine: PROBING → ARCHITECTING → EXECUTING → EVALUATING → BRANCHING → COMPLETE
Stub nodes advance the phase so the graph can run end-to-end.
"""

from __future__ import annotations

import sys
from typing import Literal

from loguru import logger
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from stem_agent.core.state import (
    AgentPhase,
    GraphState,
    StemAgentState,
    SubProblemState,
)


def environment_probe(state: GraphState) -> GraphState:
    """SP-1: Domain Comprehension. Stub: creates dummy domain model."""
    agent: StemAgentState = state["agent"]
    agent.add_log("🔬 environment_probe: Probing domain...")

    agent.domain_model = {
        "sub_problems": [
            {
                "name": "sub_problem_1",
                "description": "First discovered sub-problem",
                "expert_workflow": ["step_a", "step_b"],
                "required_tool_capabilities": ["search", "analyze"],
                "quality_rubric": "Output addresses the core question completely.",
            },
            {
                "name": "sub_problem_2",
                "description": "Second discovered sub-problem",
                "expert_workflow": ["step_x", "step_y", "step_z"],
                "required_tool_capabilities": ["search", "compute"],
                "quality_rubric": "Output is numerically accurate and well-structured.",
            },
        ],
        "overall_eval_criteria": "Completeness, accuracy, and actionability.",
    }

    agent.phase = AgentPhase.ARCHITECTING
    agent.add_log("🔬 environment_probe: Done. Found 2 sub-problems.")
    logger.info("environment_probe complete → ARCHITECTING")
    return {"agent": agent}


def architect_planner(state: GraphState) -> GraphState:
    """SP-2: Self-Architecture Design. Stub: converts domain model into SubProblemStates."""
    agent: StemAgentState = state["agent"]
    agent.add_log("📐 architect_planner: Designing architecture...")

    for sp_data in agent.domain_model.get("sub_problems", []):
        name = sp_data["name"]
        agent.sub_problems[name] = SubProblemState(
            name=name,
            description=sp_data["description"],
            expert_workflow=sp_data.get("expert_workflow", []),
            required_capabilities=sp_data.get("required_tool_capabilities", []),
            quality_rubric=sp_data.get("quality_rubric", ""),
            tools_acquired=["web_search", "python_repl"],
            prompt_templates={"base": f"You are an expert at: {name}"},
            agent_pattern="react",
        )

    agent.phase = AgentPhase.EXECUTING
    agent.add_log(
        f"📐 architect_planner: Done. Created {len(agent.sub_problems)} sub-problems."
    )
    logger.info("architect_planner complete → EXECUTING")
    return {"agent": agent}


def execution_loop(state: GraphState) -> GraphState:
    """Run tasks, observe, reflect, adapt. Stub: bumps competence scores toward threshold."""
    agent: StemAgentState = state["agent"]
    agent.add_log(f"🔄 execution_loop: Iteration {agent.iteration + 1}")

    for sp in agent.sub_problems.values():
        if not sp.is_branched:
            sp.competence_score = min(0.95, 0.3 + (agent.iteration * 0.1))
            sp.iterations += 1

    agent.iteration += 1
    agent.add_checkpoint()
    agent.phase = AgentPhase.EVALUATING
    agent.add_log(
        f"🔄 execution_loop: Scores = "
        f"{', '.join(f'{n}: {sp.competence_score:.2f}' for n, sp in agent.sub_problems.items())}"
    )
    logger.info(f"execution_loop iteration {agent.iteration} complete → EVALUATING")
    return {"agent": agent}


def competence_tracker(state: GraphState) -> GraphState:
    """SP-4: Evaluate competence and route to branching or back to execution."""
    agent: StemAgentState = state["agent"]
    agent.add_log("📊 competence_tracker: Evaluating competence...")

    if agent.iteration >= agent.max_iterations:
        agent.add_log("📊 competence_tracker: Max iterations reached.")
        agent.phase = AgentPhase.BRANCHING
    elif agent.any_ready_to_branch:
        agent.phase = AgentPhase.BRANCHING
        ready = [
            n for n, sp in agent.sub_problems.items()
            if sp.competence_score >= agent.branch_threshold and not sp.is_branched
        ]
        agent.add_log(f"📊 competence_tracker: Ready to branch: {ready}")
    else:
        agent.phase = AgentPhase.EXECUTING
        agent.add_log("📊 competence_tracker: Not ready yet. Continuing execution.")

    logger.info(f"competence_tracker complete → {agent.phase.value}")
    return {"agent": agent}


def branching_mechanism(state: GraphState) -> GraphState:
    """SP-5: Spawn specialists. Force-branches all remaining if max_iterations reached."""
    agent: StemAgentState = state["agent"]
    agent.add_log("🌿 branching_mechanism: Checking for branching...")

    force_branch = agent.iteration >= agent.max_iterations

    for name, sp in agent.sub_problems.items():
        if sp.is_branched:
            continue

        should_branch = sp.competence_score >= agent.branch_threshold or force_branch

        if should_branch:
            specialist_id = f"specialist_{name}_{agent.session_id[:8]}"
            sp.specialist_id = specialist_id
            agent.specialists[name] = specialist_id
            label = "⚠️ Force-branched" if force_branch else "✅ Branched"
            agent.add_log(
                f"🌿 branching_mechanism: {label} '{name}' → {specialist_id} "
                f"(score: {sp.competence_score:.2f})"
            )

    if agent.all_branched:
        agent.phase = AgentPhase.COMPLETE
        agent.add_log("🌿 branching_mechanism: All sub-problems branched! → COMPLETE")
    else:
        agent.phase = AgentPhase.EXECUTING
        remaining = [n for n, sp in agent.sub_problems.items() if not sp.is_branched]
        agent.add_log(
            f"🌿 branching_mechanism: Remaining un-branched: {remaining}"
        )

    logger.info(f"branching_mechanism complete → {agent.phase.value}")
    return {"agent": agent}


def complete(state: GraphState) -> GraphState:
    """Terminal node. Logs summary and returns final state."""
    agent: StemAgentState = state["agent"]
    agent.add_log("🏁 complete: Stem agent lifecycle finished.")
    agent.add_log(
        f"🏁 Summary: {len(agent.specialists)} specialists spawned "
        f"in {agent.iteration} iterations."
    )
    for name, sid in agent.specialists.items():
        score = agent.sub_problems[name].competence_score
        agent.add_log(f"   └─ {name}: {sid} (score: {score:.2f})")

    logger.info("Stem agent lifecycle COMPLETE")
    return {"agent": agent}


def route_after_competence(
    state: GraphState,
) -> Literal["branching_mechanism", "execution_loop"]:
    """After competence check: branch if ready, else keep executing."""
    agent = state["agent"]
    if agent.phase == AgentPhase.BRANCHING:
        return "branching_mechanism"
    return "execution_loop"


def route_after_branching(
    state: GraphState,
) -> Literal["complete", "execution_loop"]:
    """After branching: complete if all done, else keep evolving."""
    agent = state["agent"]
    if agent.phase == AgentPhase.COMPLETE:
        return "complete"
    return "execution_loop"


def build_graph(checkpointer=None) -> StateGraph:
    """Build and compile the stem agent LangGraph state machine."""
    graph = StateGraph(GraphState)

    graph.add_node("environment_probe", environment_probe)
    graph.add_node("architect_planner", architect_planner)
    graph.add_node("execution_loop", execution_loop)
    graph.add_node("competence_tracker", competence_tracker)
    graph.add_node("branching_mechanism", branching_mechanism)
    graph.add_node("complete", complete)

    graph.set_entry_point("environment_probe")
    graph.add_edge("environment_probe", "architect_planner")
    graph.add_edge("architect_planner", "execution_loop")
    graph.add_edge("execution_loop", "competence_tracker")

    graph.add_conditional_edges(
        "competence_tracker",
        route_after_competence,
        {
            "branching_mechanism": "branching_mechanism",
            "execution_loop": "execution_loop",
        },
    )
    graph.add_conditional_edges(
        "branching_mechanism",
        route_after_branching,
        {
            "complete": "complete",
            "execution_loop": "execution_loop",
        },
    )

    graph.add_edge("complete", END)

    if checkpointer is None:
        checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)


def main():
    """Run the stem agent graph with dummy state for smoke testing."""
    from rich.console import Console
    from rich.panel import Panel

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {message}")

    console = Console()

    console.print(Panel(
        "🧬 [bold]Stem Agent[/bold] — Smoke Test\n"
        "Running graph with stub nodes to verify structure.",
        border_style="bright_cyan",
    ))

    app = build_graph()

    initial_state: GraphState = {
        "agent": StemAgentState(
            task_class="test_domain",
            task_class_description="A test domain for smoke testing the graph structure.",
            max_iterations=10,
        )
    }

    console.print("\n[dim]Running graph...[/dim]\n")

    final_state = app.invoke(
        initial_state,
        config={"configurable": {"thread_id": "smoke-test-1"}},
    )
    agent = final_state["agent"]

    console.print(Panel(
        "\n".join(agent.log),
        title="📋 Execution Log",
        border_style="green",
    ))

    console.print(f"\n[bold green]✅ Graph completed successfully![/bold green]")
    console.print(f"  Phase: {agent.phase.value}")
    console.print(f"  Iterations: {agent.iteration}")
    console.print(f"  Sub-problems: {len(agent.sub_problems)}")
    console.print(f"  Specialists: {len(agent.specialists)}")
    console.print(f"  Checkpoints: {len(agent.checkpoints)}")
    console.print(f"  Errors: {len(agent.errors)}")
    console.print()


if __name__ == "__main__":
    main()

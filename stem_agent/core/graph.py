"""
Stem Agent LangGraph Definition

State machine: PROBING → ARCHITECTING → EXECUTING → EVALUATING → BRANCHING → COMPLETE
Stub nodes advance the phase so the graph can run end-to-end.
"""

from __future__ import annotations

import sys
import json
import yaml
from pathlib import Path
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
from stem_agent.core.config import config, get_openai_client
from stem_agent.core.skill_store import SkillStore
from stem_agent.tools.primitives import web_search

DOMAIN_ANALYSIS_PROMPT = """\
You are a domain analysis expert. Given web search results about a task domain, 
identify the key sub-problems that an AI agent would need to solve to be useful 
in this domain.

Task domain: {task_class}
Description: {description}

Web search results:
{search_results}

Analyze this domain and identify 3-5 distinct sub-problems. For each, provide:
1. name: a snake_case identifier
2. description: what this sub-problem involves
3. expert_workflow: the steps a human expert would follow (list of strings)
4. required_tool_capabilities: what tools/abilities are needed (list of strings)
5. quality_rubric: how to judge if the output is good

Also provide:
- overall_eval_criteria: how to judge the complete solution

Return ONLY valid JSON in this exact format, with no markdown fences or other text:
{{
    "sub_problems": [
        {{
            "name": "example_name",
            "description": "What this involves",
            "expert_workflow": ["step1", "step2"],
            "required_tool_capabilities": ["search", "compute"],
            "quality_rubric": "How to judge quality"
        }}
    ],
    "overall_eval_criteria": "How to judge the full solution"
}}
"""



def environment_probe(state: GraphState) -> GraphState:
    """SP-1: Domain Comprehension using SkillStore cache, web_search, and LLM."""
    agent: StemAgentState = state["agent"]
    agent.add_log(f"environment_probe: Probing domain for '{agent.task_class}'...")

    store = SkillStore()
    cached = store.get_domain(agent.task_class)

    if cached:
        agent.add_log("environment_probe: Retrieved domain model from SkillStore cache.")
        agent.domain_model = cached
    else:
        # Step 1: Web Search for context
        agent.add_log("environment_probe: Cache miss. Searching the web for domain knowledge...")
        queries = [
            f"{agent.task_class.replace('_', ' ')} key challenges and best practices",
            f"{agent.task_class.replace('_', ' ')} workflow automation opportunities"
        ]
        
        all_results = []
        for q in queries:
            res = web_search.invoke(q)
            all_results.append(f"--- Query: {q} ---\n{res}")
        
        combined_results = "\n\n".join(all_results)[:6000]

        # Step 2: Extract domain schema with LLM
        agent.add_log("environment_probe: Analyzing search results with LLM...")
        client = get_openai_client()
        prompt = DOMAIN_ANALYSIS_PROMPT.format(
            task_class=agent.task_class,
            description=agent.task_class_description,
            search_results=combined_results,
        )

        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": "You are a domain analysis expert. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=2000,
            )

            if hasattr(response, "usage") and response.usage:
                u = response.usage
                agent.add_log(f"environment_probe tokens: {u.prompt_tokens} prompt / {u.completion_tokens} completion")

            raw = response.choices[0].message.content.strip()
            if "```" in raw:
                start = raw.find("{")
                end = raw.rfind("}") + 1
                if start != -1 and end > start:
                    raw = raw[start:end]

            agent.domain_model = json.loads(raw)
            store.store_domain(agent.task_class, agent.task_class_description, agent.domain_model)
            agent.add_log("environment_probe: Successfully analyzed and cached domain model.")
        
        except Exception as e:
            agent.add_log(f"environment_probe: Failed to construct domain model. Error: {e}")
            agent.errors.append(f"Probe failure: {e}")
            agent.phase = AgentPhase.COMPLETE
            return {"agent": agent}

    num_subs = len(agent.domain_model.get("sub_problems", []))
    agent.phase = AgentPhase.ARCHITECTING
    agent.add_log(f"environment_probe: Done. Found {num_subs} sub-problems.")
    logger.info("environment_probe complete → ARCHITECTING")
    return {"agent": agent}


def architect_planner(state: GraphState) -> GraphState:
    """SP-2: Self-Architecture Design. Uses registry and composer to equip agents."""
    from stem_agent.tools.registry import DynamicToolRegistry
    from stem_agent.tools.composer import ToolComposer

    agent: StemAgentState = state["agent"]
    agent.add_log("architect_planner: Designing architecture...")

    registry = DynamicToolRegistry()
    composer = ToolComposer()

    for sp_data in agent.domain_model.get("sub_problems", []):
        name = sp_data["name"]
        
        required_caps = sp_data.get("required_tool_capabilities", [])
        acquired_tools = []
        custom_tools = {}
        
        for cap in required_caps:
            matched = registry.query([cap])
            if matched:
                acquired_tools.append(matched[0].name)
            else:
                agent.add_log(f"architect_planner: Capability '{cap}' not found. Composing new tool...")
                # Try to compose it. (For safety we might pass description if we had one per tool, but cap works)
                new_tool = composer.compose(capability=cap)
                if new_tool:
                    registry.register(new_tool, capabilities=[cap])
                    acquired_tools.append(new_tool.name)
                    if hasattr(new_tool, "metadata") and "source_code" in new_tool.metadata:
                        custom_tools[new_tool.name] = new_tool.metadata["source_code"]
                else:
                    agent.add_log(f"Failed to compose tool for capability: {cap}")
                    
        # Ensure a basic primitive is available as fallback
        if "web_search" not in acquired_tools:
            acquired_tools.append("web_search")

        expert_workflow = sp_data.get("expert_workflow", [])
        quality_rubric = sp_data.get("quality_rubric", "")
        desc = sp_data.get("description", "")
        
        system_prompt = (
            f"You are a specialized agent designed to handle: {name}.\n"
            f"Description: {desc}\n\n"
            f"Expert Workflow to follow:\n"
             + "\n".join([f"- {s}" for s in expert_workflow]) + "\n\n"
            f"Quality Rubric to maintain:\n{quality_rubric}"
        )

        agent.sub_problems[name] = SubProblemState(
            name=name,
            description=desc,
            expert_workflow=expert_workflow,
            required_capabilities=required_caps,
            quality_rubric=quality_rubric,
            tools_acquired=list(set(acquired_tools)),
            custom_tools=custom_tools,
            prompt_templates={"system": system_prompt},
            agent_pattern="react",
        )

    agent.phase = AgentPhase.EXECUTING
    agent.add_log(
        f"architect_planner: Done. Created {len(agent.sub_problems)} sub-problems."
    )
    logger.info("architect_planner complete → EXECUTING")
    return {"agent": agent}


def execution_loop(state: GraphState) -> GraphState:
    """Run tasks, observe, reflect, adapt. Supplies a synthetic test task to sub-problem agents."""
    from stem_agent.core.config import config, get_openai_client
    from stem_agent.core.state import TaskRecord

    agent: StemAgentState = state["agent"]
    agent.add_log(f"execution_loop: Iteration {agent.iteration + 1}")
    client = get_openai_client()

    for name, sp in agent.sub_problems.items():
        if not sp.is_branched:
            task_description = f"Outline a functional plan addressing '{sp.description}' using your workflow."
            system_prompt = sp.prompt_templates.get("system", "")
            
            try:
                response = client.chat.completions.create(
                    model=config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": task_description}
                    ],
                    max_completion_tokens=800
                )
                
                if hasattr(response, "usage") and response.usage:
                    u = response.usage
                    agent.add_log(f"execution_loop [{name}] tokens: {u.prompt_tokens} prompt / {u.completion_tokens} completion")
                    
                output = response.choices[0].message.content.strip()
            except Exception as e:
                output = f"Error during execution: {e}"

            sp.task_history.append(TaskRecord(
                task_id=f"{name}_iter_{agent.iteration}",
                task={"instruction": task_description},
                output=output,
                score=0.0,
                iteration=agent.iteration
            ))
            sp.iterations += 1

    agent.iteration += 1
    agent.add_checkpoint()
    agent.phase = AgentPhase.EVALUATING
    agent.add_log("execution_loop: Tasks executed. Moving to EVALUATING.")
    logger.info(f"execution_loop iteration {agent.iteration} complete → EVALUATING")
    return {"agent": agent}


EVALUATION_PROMPT = """\
You are an objective judge. Evaluate the following output based on the quality rubric.
Sub-problem: {name}
Rubric:
{rubric}

Task: {task}
Output:
{output}

Grade the response from 0.0 to 1.0, where 1.0 means it perfectly fulfills the rubric.
Respond with ONLY a valid JSON object in this exact format:
{{
    "score": 0.8,
    "reasoning": "short explanation"
}}
"""

def competence_tracker(state: GraphState) -> GraphState:
    """SP-4: Evaluate competence using LLM-as-judge and route to branching or back to execution."""
    from stem_agent.core.config import config, get_openai_client
    import json
    
    agent: StemAgentState = state["agent"]
    agent.add_log("competence_tracker: Evaluating execution outputs via LLM judge...")
    client = get_openai_client()

    for name, sp in agent.sub_problems.items():
        if not sp.is_branched and sp.task_history:
            latest_task = sp.task_history[-1]
            eval_prompt = EVALUATION_PROMPT.format(
                name=sp.name,
                rubric=sp.quality_rubric,
                task=latest_task.task.get("instruction", ""),
                output=latest_task.output
            )
            
            try:
                response = client.chat.completions.create(
                    model=config.model,
                    messages=[
                        {"role": "system", "content": "You are an objective judge. Output only JSON."},
                        {"role": "user", "content": eval_prompt}
                    ],
                    max_completion_tokens=200
                )
                
                if hasattr(response, "usage") and response.usage:
                    u = response.usage
                    agent.add_log(f"competence_tracker [{name}] tokens: {u.prompt_tokens} prompt / {u.completion_tokens} completion")
                
                raw = response.choices[0].message.content.strip()
                if "```" in raw:
                    raw = raw[raw.find("{"):raw.rfind("}")+1]
                
                eval_data = json.loads(raw)
                score = float(eval_data.get("score", 0.0))
                
                # Take max to reflect best capability so far, prevents regression blocking completion
                sp.competence_score = max(sp.competence_score, score)
                latest_task.score = score
                
            except Exception as e:
                agent.add_log(f"competence_tracker: Eval failed for {name}: {e}")
                sp.competence_score = min(0.95, sp.competence_score + 0.1)

    if agent.iteration >= agent.max_iterations:
        agent.add_log("competence_tracker: Max iterations reached.")
        agent.phase = AgentPhase.BRANCHING
    elif agent.any_ready_to_branch:
        agent.phase = AgentPhase.BRANCHING
        ready = [
            n for n, sp in agent.sub_problems.items()
            if sp.competence_score >= agent.branch_threshold and not sp.is_branched
        ]
        agent.add_log(f"competence_tracker: Ready to branch: {ready}")
    else:
        agent.phase = AgentPhase.EXECUTING
        agent.add_log("competence_tracker: Not ready yet. Continuing execution.")

    logger.info(f"competence_tracker complete → {agent.phase.value}")
    return {"agent": agent}


def branching_mechanism(state: GraphState) -> GraphState:
    """SP-5: Spawn specialists. Force-branches all remaining if max_iterations reached."""
    agent: StemAgentState = state["agent"]
    agent.add_log("branching_mechanism: Checking for branching...")

    force_branch = agent.iteration >= agent.max_iterations

    for name, sp in agent.sub_problems.items():
        if sp.is_branched:
            continue

        should_branch = sp.competence_score >= agent.branch_threshold or force_branch

        if should_branch:
            specialist_id = f"specialist_{name}_{agent.session_id[:8]}"
            sp.specialist_id = specialist_id
            
            try:
                out_path = sp.export_specialist_artifact("specialists")
                agent.add_log(f"Persistent agent artifact saved to: {out_path}")
            except Exception as e:
                agent.add_log(f"Failed to save agent artifact {name}: {e}")
                
            agent.specialists[name] = specialist_id
            label = "Force-branched" if force_branch else "Branched"
            agent.add_log(
                f"branching_mechanism: {label} '{name}' → {specialist_id} "
                f"(score: {sp.competence_score:.2f})"
            )

    if agent.all_branched:
        agent.phase = AgentPhase.COMPLETE
        agent.add_log("branching_mechanism: All sub-problems branched! → COMPLETE")
    else:
        agent.phase = AgentPhase.EXECUTING
        remaining = [n for n, sp in agent.sub_problems.items() if not sp.is_branched]
        agent.add_log(
            f"branching_mechanism: Remaining un-branched: {remaining}"
        )

    logger.info(f"branching_mechanism complete → {agent.phase.value}")
    return {"agent": agent}


def complete(state: GraphState) -> GraphState:
    """Terminal node. Logs summary and returns final state."""
    agent: StemAgentState = state["agent"]
    agent.add_log("complete: Stem agent lifecycle finished.")
    agent.add_log(
        f"Summary: {len(agent.specialists)} specialists spawned "
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
        "[bold]Stem Agent[/bold] — Smoke Test\n"
        "Running graph with stub nodes to verify structure.",
        border_style="bright_cyan",
    ))

    app = build_graph()

    initial_state: GraphState = {
        "agent": StemAgentState(
            task_class="restaurant_ops",
            task_class_description="Operational management for restaurants including inventory, scheduling, menu optimization, and cost control.",
            max_iterations=10,
        )
    }

    # Try loading from the yaml file if present
    yaml_path = Path("domains/restaurant_ops.yaml")
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
            initial_state["agent"].task_class = data.get("task_class", "restaurant_ops")
            initial_state["agent"].task_class_description = data.get("description", "")

    console.print("\n[dim]Running graph...[/dim]\n")

    final_state = app.invoke(
        initial_state,
        config={"configurable": {"thread_id": "smoke-test-1"}},
    )
    agent = final_state["agent"]

    console.print(Panel(
        "\n".join(agent.log),
        title="Execution Log",
        border_style="green",
    ))

    console.print(f"\n[bold green]Graph completed successfully![/bold green]")
    console.print(f"  Phase: {agent.phase.value}")
    console.print(f"  Iterations: {agent.iteration}")
    console.print(f"  Sub-problems: {len(agent.sub_problems)}")
    console.print(f"  Specialists: {len(agent.specialists)}")
    console.print(f"  Checkpoints: {len(agent.checkpoints)}")
    console.print(f"  Errors: {len(agent.errors)}")
    console.print()


if __name__ == "__main__":
    main()

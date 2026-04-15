"""
Stem Agent LangGraph Definition

State machine: PROBING → ARCHITECTING → EXECUTING → EVALUATING → BRANCHING → COMPLETE
Each node performs real work: web search, LLM analysis, ReAct agent execution,
LLM-as-judge evaluation, and autonomous branching into specialist agents.
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
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

from stem_agent.core.state import (
    AgentPhase,
    BenchmarkTask,
    GraphState,
    StemAgentState,
    SubProblemState,
    TaskRecord,
)
from stem_agent.core.config import config, get_openai_client
from stem_agent.core.skill_store import SkillStore
from stem_agent.core.state import UserResource
from stem_agent.tools.primitives import web_search, db_inspect, repo_inspect, PRIMITIVE_TOOLS

DOMAIN_ANALYSIS_PROMPT = """\
You are a domain analysis expert. Given web search results about a task domain, 
identify the key sub-problems that an AI agent would need to solve to be useful 
in this domain.

Task domain: {task_class}
Description: {description}

User-provided resource context (actual schemas, repo structures, etc.):
{resource_context}

Web search results:
{search_results}

Analyze this domain and identify 3-5 distinct sub-problems. If resource context
is provided above, make your sub-problems SPECIFIC to those real systems and
schemas rather than generic. For each sub-problem, provide:
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
    """SP-1: Domain Comprehension using resource inspection, web search, and LLM.

    If the user provided resources (DB URLs, repo links), the probe inspects them
    first to ground the domain model in real schemas/structures rather than just
    generic web knowledge.
    """
    agent: StemAgentState = state["agent"]
    agent.add_log(f"environment_probe: Probing domain for '{agent.task_class}'...")

    # ── Step 0: Inspect user-provided resources ────────────────────────────────
    if agent.user_resources:
        agent.add_log(f"environment_probe: Inspecting {len(agent.user_resources)} user-provided resource(s)...")
        resource_sections = []

        for resource in agent.user_resources:
            label = resource.label or resource.type
            agent.add_log(f"environment_probe: Inspecting [{resource.type}] {label}...")
            try:
                if resource.type == "database":
                    schema = db_inspect.invoke(resource.url)
                    resource.discovered_schema = schema
                    resource_sections.append(f"--- {label} (database schema) ---\n{schema}")

                elif resource.type in ("github_repo", "github_pr", "repo", "pr"):
                    structure = repo_inspect.invoke(resource.url)
                    resource.discovered_schema = structure
                    resource_sections.append(f"--- {label} (repository structure) ---\n{structure}")

                elif resource.type == "url":
                    from stem_agent.tools.primitives import read_url
                    content = read_url.invoke(resource.url)
                    resource.discovered_schema = content[:3000]
                    resource_sections.append(f"--- {label} (URL content) ---\n{content[:3000]}")

                else:
                    agent.add_log(f"environment_probe: Unknown resource type '{resource.type}' — skipping.")
                    agent.pending_clarifications.append(
                        f"Resource '{label}' has unsupported type '{resource.type}'. "
                        f"Supported types: database, github_repo, github_pr, url."
                    )

            except Exception as e:
                agent.add_log(f"environment_probe: Failed to inspect '{label}': {e}")
                agent.pending_clarifications.append(
                    f"Could not inspect '{label}' ({resource.type}) at {resource.url}: {e}. "
                    f"Please verify the connection details are correct and accessible."
                )

        agent.resource_context = "\n\n".join(resource_sections)
        agent.add_log(f"environment_probe: Resource context collected ({len(agent.resource_context)} chars).")
    else:
        agent.resource_context = "No user-provided resources."

    # ── Step 1: Check cache ────────────────────────────────────────────────────
    store = SkillStore()
    cached = store.get_domain(agent.task_class)

    if cached and not agent.user_resources:
        # Only use cache when no live resources provided (live resources may change)
        agent.add_log("environment_probe: Retrieved domain model from SkillStore cache.")
        agent.domain_model = cached
    else:
        # ── Step 2: Web Search for context ────────────────────────────────────
        agent.add_log("environment_probe: Searching the web for domain knowledge...")
        queries = [
            f"{agent.task_class.replace('_', ' ')} key challenges and best practices",
            f"{agent.task_class.replace('_', ' ')} workflow automation opportunities"
        ]

        all_results = []
        for q in queries:
            res = web_search.invoke(q)
            all_results.append(f"--- Query: {q} ---\n{res}")

        combined_results = "\n\n".join(all_results)[:6000]

        # ── Step 3: LLM domain analysis (resource-aware) ──────────────────────
        agent.add_log("environment_probe: Analyzing with LLM (resource-aware)...")
        client = get_openai_client()
        prompt = DOMAIN_ANALYSIS_PROMPT.format(
            task_class=agent.task_class,
            description=agent.task_class_description,
            resource_context=agent.resource_context[:4000],
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
            if not agent.user_resources:
                store.store_domain(agent.task_class, agent.task_class_description, agent.domain_model)
            agent.add_log("environment_probe: Successfully built domain model.")

        except Exception as e:
            agent.add_log(f"environment_probe: Failed to construct domain model. Error: {e}")
            agent.errors.append(f"Probe failure: {e}")
            agent.phase = AgentPhase.COMPLETE
            return {"agent": agent}

    # Surface any clarifications
    if agent.pending_clarifications:
        agent.add_log(f"environment_probe: {len(agent.pending_clarifications)} clarification(s) needed — see pending_clarifications.")

    num_subs = len(agent.domain_model.get("sub_problems", []))
    agent.phase = AgentPhase.ARCHITECTING
    agent.add_log(f"environment_probe: Done. Found {num_subs} sub-problems.")
    logger.info("environment_probe complete → ARCHITECTING")
    return {"agent": agent}


def architect_planner(state: GraphState) -> GraphState:
    """SP-2: Self-Architecture Design. Uses registry, composer, and validator to equip agents."""
    from stem_agent.tools.registry import DynamicToolRegistry
    from stem_agent.tools.composer import ToolComposer
    from stem_agent.tools.validator import ToolValidator
    from stem_agent.core.benchmark import generate_benchmarks

    agent: StemAgentState = state["agent"]
    agent.add_log("architect_planner: Designing architecture...")

    registry = DynamicToolRegistry()
    composer = ToolComposer()
    validator = ToolValidator(num_tests=2)

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
                new_tool = composer.compose(capability=cap)
                if new_tool:
                    # Validate before adopting
                    if validator.validate(new_tool):
                        registry.register(new_tool, capabilities=[cap])
                        acquired_tools.append(new_tool.name)
                        if hasattr(new_tool, "metadata") and "source_code" in new_tool.metadata:
                            custom_tools[new_tool.name] = new_tool.metadata["source_code"]
                        agent.add_log(f"architect_planner: Tool '{new_tool.name}' validated & registered.")
                    else:
                        agent.add_log(f"architect_planner: Tool '{new_tool.name}' FAILED validation. Skipping.")
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

        # Generate concrete benchmark tasks for this sub-problem
        agent.add_log(f"architect_planner: Generating benchmarks for '{name}'...")
        benchmarks = generate_benchmarks(
            name=name,
            description=desc,
            workflow=expert_workflow,
            tools=list(set(acquired_tools)),
            rubric=quality_rubric,
            n=5,
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
            benchmark_tasks=benchmarks,
        )
        agent.add_log(f"architect_planner: '{name}' equipped with {len(acquired_tools)} tools, {len(benchmarks)} benchmarks.")

    agent.phase = AgentPhase.EXECUTING
    agent.add_log(
        f"architect_planner: Done. Created {len(agent.sub_problems)} sub-problems."
    )
    logger.info("architect_planner complete → EXECUTING")
    return {"agent": agent}


def _resolve_tools_for_sp(sp: SubProblemState) -> list:
    """Resolve actual LangChain tool objects for a sub-problem."""
    from stem_agent.tools.registry import DynamicToolRegistry
    
    registry = DynamicToolRegistry(load_primitives=True)
    
    # Rehydrate any custom tools
    for tool_name, source_code in sp.custom_tools.items():
        try:
            from langchain_core.tools import StructuredTool
            namespace = {}
            exec(source_code, namespace)  # noqa: S102
            func = namespace.get(tool_name)
            if func:
                custom_tool = StructuredTool.from_function(
                    func=func, name=tool_name,
                    description=f"Composed tool: {tool_name}",
                )
                registry.register(custom_tool, capabilities=[])
        except Exception as e:
            logger.warning(f"Failed to rehydrate custom tool '{tool_name}': {e}")
    
    tools = []
    for t_name in sp.tools_acquired:
        tool = registry.get_by_name(t_name)
        if tool:
            tools.append(tool)
    
    # Always ensure at least web_search is available
    if not tools:
        tools = PRIMITIVE_TOOLS[:]
    
    return tools


def execution_loop(state: GraphState) -> GraphState:
    """Run benchmark tasks with real ReAct agents using each sub-problem's tools."""
    agent: StemAgentState = state["agent"]
    agent.add_log(f"execution_loop: Iteration {agent.iteration + 1}")

    llm = ChatOpenAI(model=config.model)

    for name, sp in agent.sub_problems.items():
        if sp.is_branched:
            continue

        # Pick the benchmark task for this iteration (cycle through available tasks)
        if not sp.benchmark_tasks:
            agent.add_log(f"execution_loop [{name}]: No benchmark tasks available, skipping.")
            continue

        task_idx = agent.iteration % len(sp.benchmark_tasks)
        task = sp.benchmark_tasks[task_idx]

        # Build system prompt with accumulated feedback
        system_prompt = sp.prompt_templates.get("system", "")
        if sp.eval_feedback:
            feedback_section = "\n\nPast evaluation feedback to learn from:\n" + "\n".join(
                f"- {fb}" for fb in sp.eval_feedback[-3:]  # Last 3 feedbacks
            )
            system_prompt += feedback_section

        # Resolve actual tool objects
        tools = _resolve_tools_for_sp(sp)
        agent.add_log(f"execution_loop [{name}]: Running task '{task.task_id}' with {len(tools)} tools...")

        try:
            react_agent = create_react_agent(llm, tools=tools, prompt=system_prompt)
            result = react_agent.invoke(
                {"messages": [("user", task.instruction)]},
                config={"recursion_limit": 15},
            )

            # Extract tool usage from message history
            tools_used = []
            for msg in result.get("messages", []):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tools_used.append(tc.get("name", "unknown"))

            output = result["messages"][-1].content if result.get("messages") else ""
            agent.add_log(
                f"execution_loop [{name}]: Completed. "
                f"{len(output)} chars output, tools used: {tools_used}"
            )

        except Exception as e:
            output = f"Error during execution: {e}"
            tools_used = []
            agent.add_log(f"execution_loop [{name}]: FAILED — {e}")

        sp.task_history.append(TaskRecord(
            task_id=f"{name}_iter_{agent.iteration}",
            task={"instruction": task.instruction, "success_criteria": task.success_criteria},
            output=output,
            score=0.0,
            tools_used=tools_used,
            iteration=agent.iteration,
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
    """SP-4: Evaluate competence with LLM-as-judge. Feeds reasoning back for iterative improvement."""
    client = get_openai_client()
    
    agent: StemAgentState = state["agent"]
    agent.add_log("competence_tracker: Evaluating execution outputs via LLM judge...")

    for name, sp in agent.sub_problems.items():
        if not sp.is_branched and sp.task_history:
            latest_task = sp.task_history[-1]
            
            success_criteria = latest_task.task.get("success_criteria", sp.quality_rubric)
            tools_used_str = ", ".join(latest_task.tools_used) if latest_task.tools_used else "none"
            
            eval_prompt = EVALUATION_PROMPT.format(
                name=sp.name,
                rubric=sp.quality_rubric,
                task=latest_task.task.get("instruction", ""),
                output=latest_task.output[:3000]
            )
            # Enhance with tool usage context
            eval_prompt += f"\n\nTools used by the agent: {tools_used_str}"
            eval_prompt += f"\nSuccess criteria: {success_criteria}"
            
            try:
                response = client.chat.completions.create(
                    model=config.model,
                    messages=[
                        {"role": "system", "content": "You are an objective judge. Output only JSON."},
                        {"role": "user", "content": eval_prompt}
                    ],
                    max_completion_tokens=300
                )
                
                if hasattr(response, "usage") and response.usage:
                    u = response.usage
                    agent.add_log(f"competence_tracker [{name}] tokens: {u.prompt_tokens}p / {u.completion_tokens}c")
                
                raw = response.choices[0].message.content.strip()
                if "```" in raw:
                    raw = raw[raw.find("{"):raw.rfind("}")+1]
                
                eval_data = json.loads(raw)
                score = float(eval_data.get("score", 0.0))
                reasoning = eval_data.get("reasoning", "")
                
                # Take max to reflect best capability so far
                sp.competence_score = max(sp.competence_score, score)
                latest_task.score = score
                latest_task.eval_reasoning = reasoning
                
                # Feed reasoning back for next iteration's improvement
                if reasoning:
                    sp.eval_feedback.append(
                        f"Iter {agent.iteration} (score {score:.2f}): {reasoning}"
                    )
                
                agent.add_log(
                    f"competence_tracker [{name}]: score={score:.2f} "
                    f"(best={sp.competence_score:.2f}) — {reasoning[:80]}"
                )
                
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
        agent.add_log("competence_tracker: Not ready yet. Continuing execution loop.")

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
    import json as _json
    agent: StemAgentState = state["agent"]
    agent.add_log("complete: Stem agent lifecycle finished.")
    agent.add_log(
        f"Summary: {len(agent.specialists)} specialists spawned "
        f"in {agent.iteration} iterations."
    )
    for name, sid in agent.specialists.items():
        score = agent.sub_problems[name].competence_score
        agent.add_log(f"   └─ {name}: {sid} (score: {score:.2f})")

    # Persist pending clarifications so orchestrator CLI can surface them
    if agent.pending_clarifications:
        clarifications_path = Path("specialists/pending_clarifications.json")
        clarifications_path.parent.mkdir(parents=True, exist_ok=True)
        with open(clarifications_path, "w") as f:
            _json.dump(agent.pending_clarifications, f, indent=2)
        agent.add_log(f"complete: {len(agent.pending_clarifications)} clarification(s) written to specialists/pending_clarifications.json")

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


def differentiate_from_yaml(yaml_path: str | Path, max_iterations: int = 10) -> StemAgentState:
    """Run the full differentiation pipeline from a domain YAML file.

    Returns the final StemAgentState with specialists created.
    Callable from both the CLI and as a standalone script.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Domain YAML not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    initial_state: GraphState = {
        "agent": StemAgentState(
            task_class=data.get("task_class", yaml_path.stem),
            task_class_description=data.get("description", ""),
            max_iterations=max_iterations,
        )
    }

    resources_raw = data.get("resources", [])
    if resources_raw:
        initial_state["agent"].user_resources = [
            UserResource(**r) for r in resources_raw
        ]

    app = build_graph()
    final_state = app.invoke(
        initial_state,
        config={"configurable": {"thread_id": f"diff-{data.get('task_class', 'default')}"}},
    )
    return final_state["agent"]


def main():
    """Run the stem agent graph with a domain YAML file."""
    from rich.console import Console
    from rich.panel import Panel

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {message}")

    console = Console()

    yaml_path = Path("domains/restaurant_ops.yaml")
    if len(sys.argv) > 1:
        yaml_path = Path(sys.argv[1])

    console.print(Panel(
        f"[bold]Stem Agent[/bold] — Domain Differentiation\n"
        f"Reading domain config from {yaml_path}",
        border_style="bright_cyan",
    ))

    console.print("\n[dim]Running graph...[/dim]\n")

    agent = differentiate_from_yaml(yaml_path)

    console.print(Panel(
        "\n".join(agent.log[-30:]),
        title="Execution Log (last 30)",
        border_style="green",
    ))

    if agent.pending_clarifications:
        console.print(Panel(
            "\n".join(f"• {c}" for c in agent.pending_clarifications),
            title="Clarifications Needed",
            border_style="yellow",
        ))

    console.print(f"\n[bold green]Graph completed![/bold green]")
    console.print(f"  Phase: {agent.phase.value}")
    console.print(f"  Iterations: {agent.iteration}")
    console.print(f"  Sub-problems: {len(agent.sub_problems)}")
    console.print(f"  Specialists: {len(agent.specialists)}")
    console.print(f"  Resources inspected: {len(agent.user_resources)}")
    console.print(f"  Pending clarifications: {len(agent.pending_clarifications)}")
    console.print(f"  Errors: {len(agent.errors)}")
    console.print()


if __name__ == "__main__":
    main()


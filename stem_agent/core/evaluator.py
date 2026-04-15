"""
Before/After Evaluation Framework

Runs the same benchmark tasks through:
1. A BASELINE agent (generic prompt + primitives only)
2. A SPECIALIST agent (differentiated prompt + acquired tools)

Produces a quantitative comparison report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from loguru import logger

from stem_agent.core.config import config, get_openai_client
from stem_agent.core.state import BenchmarkTask, SubProblemState
from stem_agent.tools.primitives import PRIMITIVE_TOOLS


JUDGE_PROMPT = """\
You are an objective judge evaluating an AI agent's output.

Task: {instruction}
Success Criteria: {success_criteria}
Expected Approach: {expected_approach}

Agent Output:
{output}

Evaluate the output on these dimensions:
1. **Task Completion** (0.0-1.0): Did the agent actually complete the task?
2. **Quality** (0.0-1.0): How good is the output?
3. **Tool Usage** (0.0-1.0): Did the agent appropriately use tools (vs. hallucinating)?

Return ONLY valid JSON:
{{
    "task_completion": 0.8,
    "quality": 0.7,
    "tool_usage": 0.6,
    "overall": 0.7,
    "reasoning": "Brief explanation"
}}
"""


@dataclass
class TaskResult:
    """Result of running a single benchmark task."""
    task_id: str
    instruction: str
    output: str
    score: float = 0.0
    reasoning: str = ""
    tools_used: list[str] = field(default_factory=list)
    error: str = ""
    dimensions: dict[str, float] = field(default_factory=dict)


@dataclass
class EvalReport:
    """Comparison report for a single sub-problem."""
    sub_problem: str
    baseline_results: list[TaskResult] = field(default_factory=list)
    specialist_results: list[TaskResult] = field(default_factory=list)

    @property
    def baseline_avg(self) -> float:
        scores = [r.score for r in self.baseline_results if r.score > 0]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def specialist_avg(self) -> float:
        scores = [r.score for r in self.specialist_results if r.score > 0]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def improvement(self) -> float:
        if self.baseline_avg == 0:
            return 0.0
        return ((self.specialist_avg - self.baseline_avg) / self.baseline_avg) * 100

    def summary_dict(self) -> dict[str, Any]:
        return {
            "sub_problem": self.sub_problem,
            "baseline_avg_score": round(self.baseline_avg, 3),
            "specialist_avg_score": round(self.specialist_avg, 3),
            "improvement_pct": round(self.improvement, 1),
            "n_tasks": len(self.baseline_results),
        }


def _run_agent_on_task(
    task: BenchmarkTask,
    tools: list[BaseTool],
    system_prompt: str,
    label: str = "agent",
) -> TaskResult:
    """Run a ReAct agent on a single benchmark task and return the result."""
    llm = ChatOpenAI(model=config.model)

    try:
        agent = create_react_agent(llm, tools=tools, prompt=system_prompt)
        result = agent.invoke(
            {"messages": [("user", task.instruction)]},
            config={"recursion_limit": 25},
        )

        # Extract tool calls from messages
        tools_used = []
        for msg in result.get("messages", []):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tools_used.append(tc.get("name", "unknown"))

        output = result["messages"][-1].content if result.get("messages") else ""
        logger.info(f"[{label}] Task '{task.task_id}': {len(output)} chars, {len(tools_used)} tool calls")

        return TaskResult(
            task_id=task.task_id,
            instruction=task.instruction,
            output=output,
            tools_used=tools_used,
        )

    except Exception as e:
        logger.error(f"[{label}] Task '{task.task_id}' failed: {e}")
        return TaskResult(
            task_id=task.task_id,
            instruction=task.instruction,
            output="",
            error=str(e),
        )


def _judge_result(task: BenchmarkTask, result: TaskResult) -> TaskResult:
    """Use LLM-as-judge to score a task result."""
    if result.error:
        result.score = 0.0
        result.reasoning = f"Execution error: {result.error}"
        return result

    client = get_openai_client()
    prompt = JUDGE_PROMPT.format(
        instruction=task.instruction,
        success_criteria=task.success_criteria,
        expected_approach=task.expected_approach,
        output=result.output[:3000],  # Truncate for context window
    )

    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": "You are an objective judge. Return only JSON."},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=300,
        )

        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw[raw.find("{"):raw.rfind("}") + 1]

        data = json.loads(raw)
        result.score = float(data.get("overall", 0.0))
        result.reasoning = data.get("reasoning", "")
        result.dimensions = {
            "task_completion": data.get("task_completion", 0.0),
            "quality": data.get("quality", 0.0),
            "tool_usage": data.get("tool_usage", 0.0),
        }

    except Exception as e:
        logger.error(f"Judging failed for {result.task_id}: {e}")
        result.score = 0.0
        result.reasoning = f"Judging error: {e}"

    return result


def run_baseline_evaluation(
    tasks: list[BenchmarkTask],
    sub_problem_name: str,
) -> list[TaskResult]:
    """Run benchmark tasks with a generic agent (no specialization)."""
    generic_prompt = (
        "You are a general-purpose AI assistant. "
        "Use the available tools to complete the task to the best of your ability."
    )
    results = []
    for task in tasks:
        result = _run_agent_on_task(task, PRIMITIVE_TOOLS, generic_prompt, label=f"baseline:{sub_problem_name}")
        result = _judge_result(task, result)
        results.append(result)
    return results


def run_specialist_evaluation(
    tasks: list[BenchmarkTask],
    sp: SubProblemState,
    specialist_tools: list[BaseTool],
) -> list[TaskResult]:
    """Run benchmark tasks with the differentiated specialist agent."""
    system_prompt = sp.prompt_templates.get("system", "You are a helpful assistant.")
    results = []
    for task in tasks:
        result = _run_agent_on_task(task, specialist_tools, system_prompt, label=f"specialist:{sp.name}")
        result = _judge_result(task, result)
        results.append(result)
    return results


def build_eval_report(
    sp: SubProblemState,
    specialist_tools: list[BaseTool],
    n_tasks: int = 3,
) -> EvalReport:
    """Full before/after evaluation for a single sub-problem."""
    tasks = sp.benchmark_tasks[:n_tasks]
    if not tasks:
        logger.warning(f"No benchmark tasks for '{sp.name}', skipping eval")
        return EvalReport(sub_problem=sp.name)

    logger.info(f"Running baseline evaluation for '{sp.name}' ({len(tasks)} tasks)...")
    baseline = run_baseline_evaluation(tasks, sp.name)

    logger.info(f"Running specialist evaluation for '{sp.name}' ({len(tasks)} tasks)...")
    specialist = run_specialist_evaluation(tasks, sp, specialist_tools)

    report = EvalReport(
        sub_problem=sp.name,
        baseline_results=baseline,
        specialist_results=specialist,
    )

    logger.info(
        f"Eval report for '{sp.name}': baseline={report.baseline_avg:.2f}, "
        f"specialist={report.specialist_avg:.2f}, improvement={report.improvement:+.1f}%"
    )
    return report

"""
Benchmark Task Generator

Generates concrete, tool-requiring test tasks for each sub-problem.
These tasks are what the execution loop actually runs agents against,
replacing the previous "outline a plan" stubs.
"""

from __future__ import annotations

import json
from loguru import logger

from stem_agent.core.config import config, get_openai_client
from stem_agent.core.state import BenchmarkTask


BENCHMARK_PROMPT = """\
You are a benchmark designer for AI agent evaluation. Given a sub-problem 
description and the tools available, generate {n} concrete tasks that an 
AI agent should be able to solve using those tools.

Sub-problem: {name}
Description: {description}
Expert workflow: {workflow}
Available tools: {tools}
Quality rubric: {rubric}

Requirements for each task:
1. It must be CONCRETE and ACTIONABLE (not "create a plan" — something like 
   "research the top 5 inventory management systems and compare their pricing")
2. It must require using at least one of the available tools
3. It should have clear, measurable success criteria
4. Include a mix of difficulties (easy, medium, hard)

Return ONLY valid JSON in this exact format:
{{
    "tasks": [
        {{
            "instruction": "The concrete task the agent must perform",
            "expected_approach": "Brief description of how an expert would solve this",
            "difficulty": "easy|medium|hard",
            "requires_tools": ["tool_name_1"],
            "success_criteria": "What makes the output good"
        }}
    ]
}}
"""


def generate_benchmarks(
    name: str,
    description: str,
    workflow: list[str],
    tools: list[str],
    rubric: str,
    n: int = 5,
) -> list[BenchmarkTask]:
    """Generate benchmark tasks for a sub-problem using LLM."""
    client = get_openai_client()

    prompt = BENCHMARK_PROMPT.format(
        n=n,
        name=name,
        description=description,
        workflow="\n".join(f"- {s}" for s in workflow),
        tools=", ".join(tools),
        rubric=rubric,
    )

    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": "You are a benchmark designer. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=2000,
        )

        if hasattr(response, "usage") and response.usage:
            u = response.usage
            logger.info(f"benchmark_gen [{name}] tokens: {u.prompt_tokens}p / {u.completion_tokens}c")

        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                raw = raw[start:end]

        data = json.loads(raw)
        tasks = []
        for i, t in enumerate(data.get("tasks", [])):
            tasks.append(BenchmarkTask(
                task_id=f"{name}_bench_{i}",
                instruction=t["instruction"],
                expected_approach=t.get("expected_approach", ""),
                difficulty=t.get("difficulty", "medium"),
                requires_tools=t.get("requires_tools", []),
                success_criteria=t.get("success_criteria", ""),
            ))

        logger.info(f"Generated {len(tasks)} benchmark tasks for '{name}'")
        return tasks

    except Exception as e:
        logger.error(f"Benchmark generation failed for '{name}': {e}")
        return []

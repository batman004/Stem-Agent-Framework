"""
TP3 Demo — Environment Probe (standalone)

Takes a task_class like "restaurant_ops" and:
1. Searches the web to understand the domain
2. Uses LLM to discover real sub-problems
3. Outputs a structured domain model

Run: PYTHONPATH=. python demos/tp3_environment_probe_demo.py
"""

import json
import sys
from dotenv import load_dotenv
load_dotenv()

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from stem_agent.core.config import config, get_openai_client
from stem_agent.tools.primitives import web_search

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {message}")

console = Console()
client = get_openai_client()

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

Return ONLY valid JSON in this exact format:
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


def probe_domain(task_class: str, description: str) -> dict:
    """Run the environment probe: search → analyze → structure."""

    # Step 1: Search for domain knowledge
    console.print("\n[bold cyan]Step 1: Searching for domain knowledge...[/bold cyan]")

    search_queries = [
        f"{task_class.replace('_', ' ')} key challenges and best practices",
        f"{task_class.replace('_', ' ')} workflow automation opportunities",
        f"{task_class.replace('_', ' ')} common tasks and operations",
    ]

    all_results = []
    for q in search_queries:
        result = web_search.invoke(q)
        all_results.append(f"--- Query: {q} ---\n{result}")
        console.print(f"  ✅ Searched: [dim]{q}[/dim]")

    combined_results = "\n\n".join(all_results)
    console.print(f"  📄 Collected {len(combined_results)} chars of search data\n")

    # Step 2: LLM analysis
    console.print("[bold cyan]Step 2: LLM analyzing domain...[/bold cyan]")

    prompt = DOMAIN_ANALYSIS_PROMPT.format(
        task_class=task_class,
        description=description,
        search_results=combined_results[:6000],
    )

    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": "You are a domain analysis expert. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=2000,
    )

    raw = response.choices[0].message.content.strip()

    # Extract JSON from markdown fences if present
    if "```" in raw:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]

    domain_model = json.loads(raw)
    console.print(f"  ✅ Discovered {len(domain_model.get('sub_problems', []))} sub-problems\n")

    return domain_model


def main():
    task_class = "restaurant_ops"
    description = "Operational management for restaurants including inventory, scheduling, menu optimization, and cost control."

    console.print(Panel(
        f"🔬 [bold]TP3 Environment Probe Demo[/bold]\n"
        f"Domain: [cyan]{task_class}[/cyan]\n"
        f"{description}",
        border_style="bright_cyan",
    ))

    domain_model = probe_domain(task_class, description)

    # Display results
    console.print("[bold cyan]Step 3: Domain Model Output[/bold cyan]\n")

    formatted = json.dumps(domain_model, indent=2)
    console.print(Syntax(formatted, "json", theme="monokai", line_numbers=True))

    console.print(f"\n[bold green]✅ Environment probe complete![/bold green]")
    console.print(f"  Sub-problems: {len(domain_model.get('sub_problems', []))}")
    for sp in domain_model.get("sub_problems", []):
        caps = ", ".join(sp.get("required_tool_capabilities", []))
        console.print(f"    └─ [cyan]{sp['name']}[/cyan]: {sp['description'][:60]}...")
        console.print(f"       Tools needed: {caps}")
    console.print()


if __name__ == "__main__":
    main()

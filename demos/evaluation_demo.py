"""
Stem Agent — Before/After Evaluation Demo

End-to-end demonstration showing:
1. Running the stem agent lifecycle (differentiation) on a chosen domain
2. Running before/after evaluation comparing baseline vs. specialist agents
3. Producing a rich comparison report with per-task breakdowns

Usage:
    python -m demos.evaluation_demo                     # default: restaurant_ops
    python -m demos.evaluation_demo security_audit      # alternate domain
"""

import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from loguru import logger

from stem_agent.core.graph import build_graph
from stem_agent.core.state import StemAgentState, GraphState
from stem_agent.core.evaluator import build_eval_report, EvalReport
from stem_agent.core.graph import _resolve_tools_for_sp

# Clean logging for CLI
logger.remove()
logger.add(sys.stderr, level="WARNING")


def run_differentiation(domain_file: str, console: Console) -> StemAgentState:
    """Run the full stem agent lifecycle on a domain."""
    yaml_path = Path(domain_file)
    if not yaml_path.exists():
        console.print(f"[red]Error: Domain config not found at {domain_file}[/red]")
        sys.exit(1)

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
        task_class = data.get("task_class", "")
        description = data.get("description", "")

    console.print(f"\n[cyan]Target Domain:[/cyan] {task_class}")
    console.print(f"[dim]{description}[/dim]")

    initial_state: GraphState = {
        "agent": StemAgentState(
            task_class=task_class,
            task_class_description=description,
            max_iterations=3,  # Enough to show iteration-over-iteration improvement
        )
    }

    app = build_graph()

    with console.status("[bold cyan]Running stem agent differentiation lifecycle...", spinner="aesthetic"):
        final_state = app.invoke(
            initial_state,
            config={"configurable": {"thread_id": f"eval-{task_class}"}},
        )

    agent = final_state["agent"]

    # Show differentiation log
    console.print(Panel(
        "\n".join(agent.log[-20:]),  # Last 20 log lines
        title="Differentiation Log (last 20 entries)",
        border_style="blue",
    ))

    console.print(f"\n[bold green]Differentiation Complete[/bold green]")
    console.print(f"  Phase: {agent.phase.value}")
    console.print(f"  Iterations: {agent.iteration}")
    console.print(f"  Specialists: {len(agent.specialists)}")
    return agent


def run_evaluation(agent: StemAgentState, console: Console) -> list[EvalReport]:
    """Run before/after evaluation on differentiated specialists."""
    reports = []

    console.print(Panel(
        "[bold]Running Before/After Evaluation[/bold]\n"
        "Comparing baseline (generic agent + primitives) vs. specialists (differentiated agent + acquired tools)\n"
        "on the same benchmark tasks.",
        border_style="yellow"
    ))

    for name, sp in agent.sub_problems.items():
        if not sp.benchmark_tasks:
            console.print(f"[yellow]Skipping '{name}' — no benchmark tasks[/yellow]")
            continue

        console.print(f"\n[cyan]Evaluating sub-problem:[/cyan] {name}")
        tools = _resolve_tools_for_sp(sp)

        with console.status(f"[bold]Evaluating '{name}'...", spinner="aesthetic"):
            report = build_eval_report(sp, tools, n_tasks=min(3, len(sp.benchmark_tasks)))

        reports.append(report)

    return reports


def display_reports(reports: list[EvalReport], console: Console) -> None:
    """Display rich comparison tables."""
    # Summary table
    summary = Table(
        title="Before/After Evaluation Summary",
        border_style="bright_green",
        show_lines=True,
    )
    summary.add_column("Sub-Problem", justify="left", style="cyan", no_wrap=True)
    summary.add_column("Baseline Avg", justify="center")
    summary.add_column("Specialist Avg", justify="center")
    summary.add_column("Improvement", justify="center")
    summary.add_column("Tasks", justify="center")

    for report in reports:
        imp = report.improvement
        imp_style = "green" if imp > 0 else "red" if imp < 0 else "dim"

        summary.add_row(
            report.sub_problem,
            f"{report.baseline_avg:.2f}",
            f"{report.specialist_avg:.2f}",
            f"[{imp_style}]{imp:+.1f}%[/{imp_style}]",
            str(len(report.baseline_results)),
        )

    console.print("\n")
    console.print(summary)

    # Per-task detail tables
    for report in reports:
        detail = Table(
            title=f"Task Detail — {report.sub_problem}",
            border_style="cyan",
            show_lines=True,
        )
        detail.add_column("Task", justify="left", max_width=50)
        detail.add_column("Baseline", justify="center")
        detail.add_column("Specialist", justify="center")
        detail.add_column("Δ", justify="center")
        detail.add_column("Specialist Tools", justify="left")

        for base, spec in zip(report.baseline_results, report.specialist_results):
            delta = spec.score - base.score
            delta_style = "green" if delta > 0 else "red" if delta < 0 else "dim"
            tools_str = ", ".join(spec.tools_used[:4]) if spec.tools_used else "none"

            detail.add_row(
                base.instruction[:50] + ("..." if len(base.instruction) > 50 else ""),
                f"{base.score:.2f}",
                f"{spec.score:.2f}",
                f"[{delta_style}]{delta:+.2f}[/{delta_style}]",
                tools_str,
            )

        console.print(detail)

    # Overall
    if reports:
        all_baseline = sum(r.baseline_avg for r in reports) / len(reports)
        all_specialist = sum(r.specialist_avg for r in reports) / len(reports)
        overall_imp = ((all_specialist - all_baseline) / all_baseline * 100) if all_baseline > 0 else 0

        console.print(Panel(
            f"[bold]Overall Baseline:[/bold] {all_baseline:.3f}\n"
            f"[bold]Overall Specialist:[/bold] {all_specialist:.3f}\n"
            f"[bold]Overall Improvement:[/bold] {overall_imp:+.1f}%",
            title="Aggregate Results",
            border_style="bright_green",
        ))


def main():
    load_dotenv()
    console = Console()

    domain = sys.argv[1] if len(sys.argv) > 1 else "restaurant_ops"
    domain_file = f"domains/{domain}.yaml"

    console.print(Panel(
        "[bold green]Stem Agent Framework[/bold green] — Before/After Evaluation\n"
        f"Domain: [cyan]{domain}[/cyan]",
        border_style="bright_green",
    ))

    # Phase 1: Differentiate
    agent = run_differentiation(domain_file, console)

    # Phase 2: Evaluate before/after
    reports = run_evaluation(agent, console)

    # Phase 3: Display results
    display_reports(reports, console)

    console.print("\n[bold]Evaluation complete![/bold]\n")


if __name__ == "__main__":
    main()

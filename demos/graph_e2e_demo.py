"""
Graph End-to-End Demo (TP3 Integration)

This script demonstrates the complete Stem Agent lifecycle by executing 
the LangGraph state machine from start to finish.

It will:
1. Initialize the graph
2. Read the domain parameters from `domains/restaurant_ops.yaml`
3. Execute the `environment_probe` (performing real web searches and LLM analysis)
   OR pulling directly from the persistent ChromaDB `SkillStore` if cached.
4. Auto-spawn specialists as competence scores iterate upwards.
"""

import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from loguru import logger

from stem_agent.core.graph import build_graph
from stem_agent.core.state import StemAgentState, GraphState

# Mute heavy debug logging for a cleaner CLI presentation
logger.remove()
logger.add(sys.stderr, level="ERROR")

def main():
    load_dotenv()
    console = Console()

    console.print(Panel(
        "[bold green]Stem Agent Lifecycle — End-to-End Execution[/bold green]\n"
        "Testing the TP3 capabilities involving live Search, LLM generation, "
        "and ChromaDB persistence.",
        border_style="bright_green",
    ))

    # 1. Load configuration
    yaml_path = Path("domains/restaurant_ops.yaml")
    if not yaml_path.exists():
        console.print("[red]Error: Target domain config not found at domains/restaurant_ops.yaml[/red]")
        sys.exit(1)
        
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
        task_class = data.get("task_class", "")
        description = data.get("description", "")

    console.print(f"\n[cyan]Target Domain:[/cyan] {task_class}")
    console.print(f"[dim]{description}[/dim]")

    # 2. Setup initial state
    initial_state: GraphState = {
        "agent": StemAgentState(
            task_class=task_class,
            task_class_description=description,
            max_iterations=8,  # Slightly faster iterations for demo
        )
    }

    # 3. Build & Invoke the graph
    app = build_graph()
    console.print("\n[bold yellow]Invoking target graph (this may take 15s to 30s on cache-miss)[/bold yellow]\n")
    
    with console.status("[bold cyan]Agent Graph Executing...", spinner="aesthetic"):
        final_state = app.invoke(
            initial_state,
            config={"configurable": {"thread_id": "e2e-demo-run"}},
        )

    agent = final_state["agent"]
    
    # 4. Display log history gracefully
    console.print(Panel(
        "\n".join(agent.log),
        title="Internal Graph Execution Logs",
        border_style="blue",
    ))

    # 5. Display Specialist Analytics
    console.print(f"\n[bold green]Agent Reached Terminal Phase:[/bold green] {agent.phase.value}")
    console.print(f"Total loop iterations: [bold]{agent.iteration}[/bold]")
    
    table = Table(title="Spanned Specialists", border_style="cyan")
    table.add_column("Sub-Problem ID", justify="left", style="cyan", no_wrap=True)
    table.add_column("Specialist UUID", style="magenta")
    table.add_column("Competence", justify="right", style="green")

    for name, sid in agent.specialists.items():
        score = agent.sub_problems[name].competence_score
        table.add_row(name, sid, f"{score:.2f}")

    console.print(table)
    console.print("\n[bold]Demo completed successfully![/bold]\n")

if __name__ == "__main__":
    main()

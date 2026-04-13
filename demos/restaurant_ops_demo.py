"""
Restaurant Ops Demo — shows what works now vs what's coming.

This demonstrates:
1. Primitive tools working with restaurant queries (LIVE)
2. Registry querying capabilities (LIVE)
3. Tool composer creating a domain-specific tool (LIVE)  
4. Graph running with stubs (STUB — real logic in TP3+)
"""

import sys
from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {message}")

console = Console()

console.print(Panel(
    "🍽️  [bold]Restaurant Ops Demo[/bold]\n"
    "Testing each TP2 component with restaurant-domain inputs",
    border_style="bright_cyan",
))


# ── 1. Web Search (Serper.dev) ──
console.print("\n[bold cyan]1. Web Search — restaurant domain[/bold cyan]")
from stem_agent.tools.primitives import web_search
result = web_search.invoke("restaurant inventory management best practices")
console.print(Panel(result[:600] + "...", title="🔍 Search Results", border_style="green"))


# ── 2. Python REPL — restaurant calc ──
console.print("\n[bold cyan]2. Python REPL — food cost calculation[/bold cyan]")
from stem_agent.tools.primitives import python_repl
code = """
# Restaurant food cost percentage calculation
food_cost = 3500  # weekly food purchases
revenue = 12000   # weekly revenue
food_cost_pct = (food_cost / revenue) * 100
print(f"Food Cost: ${food_cost}")
print(f"Revenue: ${revenue}")
print(f"Food Cost %: {food_cost_pct:.1f}%")
print(f"Target: 28-32%")
print(f"Status: {'On target' if 28 <= food_cost_pct <= 32 else 'Needs attention'}")
"""
result = python_repl.invoke(code)
console.print(Panel(result, title="🧮 REPL Output", border_style="green"))


# ── 3. Registry — capability matching ──
console.print("\n[bold cyan]3. Registry — what tools match restaurant needs?[/bold cyan]")
from stem_agent.tools.registry import DynamicToolRegistry
registry = DynamicToolRegistry()

table = Table(title="Tool Capability Matching")
table.add_column("Capability Needed", style="cyan")
table.add_column("Tools Found", style="green")

for cap in ["search", "compute", "fetch", "write"]:
    tools = registry.query([cap])
    tool_names = ", ".join(t.name for t in tools)
    table.add_row(cap, tool_names)

missing = registry.missing_capabilities(["search", "compute", "menu_optimization", "scheduling"])
table.add_row("[red]MISSING[/red]", ", ".join(missing) or "none")
console.print(table)


# ── 4. Tool Composer — create a restaurant-specific tool ──
console.print("\n[bold cyan]4. Tool Composer — generating a restaurant-specific tool[/bold cyan]")
from stem_agent.tools.composer import ToolComposer
from stem_agent.tools.validator import ToolValidator

composer = ToolComposer()
tool = composer.compose(
    capability="food_cost_calculator",
    description="Calculate food cost percentage given total food expenses and total revenue as comma-separated values",
)

if tool:
    console.print(f"  Composed tool: [green]{tool.name}[/green]")
    test_result = tool.invoke("3500, 12000")
    console.print(f"  Test run (3500, 12000): {test_result}")

    validator = ToolValidator(num_tests=2)
    valid = validator.validate(tool)
    console.print(f"  {'Validated' if valid else 'Failed validation'}")

    if valid:
        registry.register(tool, capabilities=["compute", "food_cost", "restaurant"])
        console.print(f"  📦 Registered in tool registry ({registry.size} total tools)")
else:
    console.print("  Composition failed")


# ── 5. Graph — stub run with restaurant_ops ──
console.print("\n[bold cyan]5. Graph — running with task_class='restaurant_ops'[/bold cyan]")
console.print("  [dim]Still using stub nodes — real LLM logic comes in TP3[/dim]\n")
from stem_agent.core.graph import build_graph
from stem_agent.core.state import StemAgentState, GraphState

app = build_graph()
state: GraphState = {
    "agent": StemAgentState(
        task_class="restaurant_ops",
        task_class_description="Operational management for restaurants including inventory, scheduling, menu optimization, and cost control.",
        max_iterations=10,
    )
}
final = app.invoke(state, config={"configurable": {"thread_id": "restaurant-demo"}})
agent = final["agent"]

console.print(f"  Phase: [green]{agent.phase.value}[/green]")
console.print(f"  Iterations: {agent.iteration}")
console.print(f"  Specialists spawned: {len(agent.specialists)}")
for name, sid in agent.specialists.items():
    score = agent.sub_problems[name].competence_score
    console.print(f"    └─ {name}: {sid} (score: {score:.2f})")


# ── Summary ──
console.print(Panel(
    "[green]Working NOW:[/green] web search, python repl, read_url, write_file,\n"
    "   tool registry, tool composer, tool validator, graph structure\n\n"
    "[yellow]⬜ Coming in TP3:[/yellow] LLM-powered environment_probe that will:\n"
    "   • Search 'restaurant operations' via web_search\n"
    "   • Use GPT-4o to discover REAL sub-problems (inventory, scheduling, etc.)\n"
    "   • Replace the dummy sub_problem_1/sub_problem_2 with actual domain analysis",
    title="Status",
    border_style="bright_cyan",
))

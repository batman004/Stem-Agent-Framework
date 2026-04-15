"""
Stem Agent — Unified CLI

Single entry point: discover domains, differentiate specialists,
chat with them, and switch between domains on the fly.

Commands:
  /domains          List available domain YAML configs
  /specialists      List loaded specialist proxies
  /differentiate    Run differentiation for a domain YAML
  /switch <domain>  Switch active domain (loads its specialists)
  /clear            Clear specialist directory and start fresh
  exit              Quit
"""

import sys
import shutil
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from loguru import logger

from stem_agent.core.orchestrator import SpecialistRouter
from stem_agent.core.config import config

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

DOMAINS_DIR = Path("domains")
SPECIALISTS_DIR = Path("specialists")

console = Console()


def discover_domains() -> dict[str, Path]:
    """Scan the domains/ directory for YAML configs (excluding template)."""
    import yaml
    domains = {}
    for f in sorted(DOMAINS_DIR.glob("*.yaml")):
        if f.stem == "template":
            continue
        try:
            with open(f) as fh:
                data = yaml.safe_load(fh)
            domains[data.get("task_class", f.stem)] = f
        except Exception:
            domains[f.stem] = f
    return domains


def print_domains(domains: dict[str, Path]) -> None:
    table = Table(title="Available Domains", border_style="cyan")
    table.add_column("#", style="dim")
    table.add_column("Domain", style="bold")
    table.add_column("YAML Path", style="dim")
    for i, (name, path) in enumerate(domains.items(), 1):
        table.add_row(str(i), name, str(path))
    console.print(table)


def print_specialists(router: SpecialistRouter) -> None:
    proxies = router.get_proxy_tools()
    if not proxies:
        console.print("[yellow]No specialists loaded.[/yellow]")
        return
    table = Table(title="Loaded Specialists", border_style="green")
    table.add_column("#", style="dim")
    table.add_column("Name", style="bold")
    table.add_column("Description")
    for i, p in enumerate(proxies, 1):
        table.add_row(str(i), p.name, p.description[:80])
    console.print(table)


def run_differentiation(yaml_path: Path) -> bool:
    """Trigger the full differentiation pipeline and return success bool."""
    from stem_agent.core.graph import differentiate_from_yaml

    console.print(f"\n[bold cyan]Differentiating from {yaml_path}...[/bold cyan]")
    console.print("[dim]This will probe the domain, inspect resources, architect sub-problems,")
    console.print("run benchmark tasks, evaluate competence, and branch specialists.[/dim]\n")

    try:
        agent = differentiate_from_yaml(yaml_path, max_iterations=3)

        console.print(Panel(
            "\n".join(agent.log[-20:]),
            title="Differentiation Log (last 20)",
            border_style="green",
        ))

        if agent.pending_clarifications:
            console.print(Panel(
                "\n".join(f"- {c}" for c in agent.pending_clarifications),
                title="Clarifications Needed",
                border_style="yellow",
            ))

        console.print(
            f"[bold green]Done![/bold green] "
            f"{len(agent.specialists)} specialist(s) created in {agent.iteration} iteration(s).\n"
        )
        return bool(agent.specialists)

    except Exception as e:
        console.print(f"[bold red]Differentiation failed:[/bold red] {e}")
        return False


def build_supervisor(proxy_tools: list):
    """Build a ReAct supervisor agent with the given proxy tools."""
    llm = ChatOpenAI(model=config.model)
    system_prompt = (
        "You are the Stem Agent Supervisor. You facilitate conversations with the user. "
        "If the user's task requires a specialized capability, invoke the corresponding proxy tool "
        "and delegate the specific requirement to it. Combine and return the specialist's answer back "
        "to the user organically. If you do not have a specialist for this, answer directly "
        "using your own foundation knowledge.\n\n"
        "Available specialists:\n"
        + "\n".join(f"- {t.name}: {t.description[:100]}" for t in proxy_tools)
    )
    return create_react_agent(llm, tools=proxy_tools, prompt=system_prompt)


def main():
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    console.print(Panel(
        "[bold cyan]Stem Agent[/bold cyan] — Supervisor & Orchestrator CLI\n"
        "Type [bold]/domains[/bold] to see available domains, "
        "[bold]/differentiate[/bold] to create specialists,\n"
        "or just start chatting. Type [bold]exit[/bold] to quit.",
        border_style="bright_cyan"
    ))

    domains = discover_domains()

    # Load existing specialists
    router = SpecialistRouter()
    proxy_tools = router.get_proxy_tools()

    # Surface pending clarifications
    clarifications = router.get_pending_clarifications()
    if clarifications:
        console.print(Panel(
            "\n".join(f"- {c}" for c in clarifications),
            title="Pending Clarifications from Last Run",
            border_style="yellow",
        ))

    if proxy_tools:
        console.print(f"Loaded [bold green]{len(proxy_tools)}[/bold green] specialist(s) from previous runs.\n")
        print_specialists(router)
    else:
        console.print("[yellow]No specialists found yet.[/yellow]")
        print_domains(domains)
        console.print(
            "\n[dim]Pick a domain with[/dim] [bold]/differentiate <domain>[/bold] "
            "[dim]to get started.[/dim]\n"
        )

    agent_executor = build_supervisor(proxy_tools) if proxy_tools else None
    chat_history: list = []

    console.print("[bold green]Ready.[/bold green] (Type 'exit' to quit)\n")

    while True:
        try:
            user_input = Prompt.ask("[bold magenta]You[/bold magenta]")
            stripped = user_input.strip()

            if stripped.lower() in ("exit", "quit", "q"):
                console.print("\n[dim]Shutting down.[/dim]")
                break

            # --- Slash commands ---
            if stripped.startswith("/"):
                parts = stripped.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1].strip() if len(parts) > 1 else ""

                if cmd == "/domains":
                    domains = discover_domains()
                    print_domains(domains)

                elif cmd == "/specialists":
                    print_specialists(router)

                elif cmd == "/differentiate":
                    target = arg
                    if not target:
                        domains = discover_domains()
                        print_domains(domains)
                        target = Prompt.ask("Which domain?")

                    # Resolve: accept name or number
                    domain_list = list(domains.items())
                    yaml_path = None
                    if target.isdigit():
                        idx = int(target) - 1
                        if 0 <= idx < len(domain_list):
                            yaml_path = domain_list[idx][1]
                    elif target in domains:
                        yaml_path = domains[target]
                    else:
                        # Try matching partial name
                        for name, path in domains.items():
                            if target in name:
                                yaml_path = path
                                break

                    if yaml_path is None:
                        console.print(f"[red]Unknown domain '{target}'. Use /domains to list.[/red]")
                        continue

                    success = run_differentiation(yaml_path)
                    if success:
                        router = SpecialistRouter()
                        proxy_tools = router.get_proxy_tools()
                        agent_executor = build_supervisor(proxy_tools)
                        chat_history = []
                        print_specialists(router)

                elif cmd == "/switch":
                    if not arg:
                        console.print("[red]Usage: /switch <domain_name>[/red]")
                        continue
                    # Just re-differentiate for the new domain
                    target = arg
                    yaml_path = domains.get(target)
                    if yaml_path is None:
                        for name, path in domains.items():
                            if target in name:
                                yaml_path = path
                                break
                    if yaml_path is None:
                        console.print(f"[red]Unknown domain '{target}'. Use /domains to list.[/red]")
                        continue

                    success = run_differentiation(yaml_path)
                    if success:
                        router = SpecialistRouter()
                        proxy_tools = router.get_proxy_tools()
                        agent_executor = build_supervisor(proxy_tools)
                        chat_history = []
                        print_specialists(router)

                elif cmd == "/clear":
                    if SPECIALISTS_DIR.exists():
                        shutil.rmtree(SPECIALISTS_DIR)
                        SPECIALISTS_DIR.mkdir()
                    router = SpecialistRouter()
                    proxy_tools = []
                    agent_executor = None
                    chat_history = []
                    console.print("[yellow]Cleared all specialists.[/yellow]")

                else:
                    console.print(f"[red]Unknown command: {cmd}[/red]")
                    console.print("[dim]/domains, /specialists, /differentiate, /switch, /clear[/dim]")

                continue

            # --- Chat with supervisor ---
            if agent_executor is None:
                console.print(
                    "[yellow]No specialists loaded. The supervisor will answer with "
                    "general knowledge only.[/yellow]"
                )
                # Build a generalist supervisor with no specialist tools
                llm = ChatOpenAI(model=config.model)
                agent_executor = create_react_agent(
                    llm, tools=[],
                    prompt="You are a helpful AI assistant. Answer the user's question directly."
                )

            console.print("[dim italic]Thinking and routing...[/dim italic]")

            chat_history.append(("user", user_input))

            result = agent_executor.invoke({"messages": chat_history})
            output = result["messages"][-1].content

            chat_history.append(("assistant", output))

            console.print()
            console.print(Panel(
                output,
                title="Stem Agent Supervisor",
                border_style="magenta",
                padding=(1, 2)
            ))

        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Shutting down.[/dim]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")


if __name__ == "__main__":
    main()

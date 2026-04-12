"""
Agent Tool-Calling Demo

This script demonstrates an LLM-powered agent autonomously selecting
and executing primitive tools from the Stem Agent Framework registry.
"""

import sys
import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from loguru import logger
from rich.console import Console
from rich.markdown import Markdown

# Turn off verbose loguru to make the demo output clean
logger.remove()

from stem_agent.core.config import config
from stem_agent.tools.registry import DynamicToolRegistry
from langchain_openai import ChatOpenAI

def main():
    console = Console()
    load_dotenv()

    # Ensure required secrets exist
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]Error: OPENAI_API_KEY environment variable is missing.[/red]")
        sys.exit(1)

    console.print("[bold green]Initializing Agent Tool-Calling Demo...[/bold green]")
    
    # 1. Initialize Registry and fetch tools
    registry = DynamicToolRegistry(load_primitives=True)
    all_tools = [entry.tool for entry in registry._tools.values()]
    
    console.print(f"Loaded {len(all_tools)} tools from DynamicToolRegistry:")
    for t in all_tools:
        console.print(f"  - [cyan]{t.name}[/cyan]: [dim]{t.description}[/dim]")
        
    # 2. Setup LLM
    console.print(f"\n[dim]Connecting to ChatOpenAI with model: {config.model}...[/dim]")
    llm = ChatOpenAI(model=config.model, temperature=config.temperature)

    # 3. Create the LangGraph tool-calling ReAct agent
    # This automatically handles the observation loop (Prompt -> LLM -> Tool Call -> Observation -> LLM -> Final Answer)
    agent_executor = create_react_agent(llm, tools=all_tools)

    console.print("\n[bold cyan]Agent initialized. Processing predefined prompts...[/bold cyan]\n")
    
    prompts = [
        "What is 345 multiplied by 876? Use python to calculate it.",
        "Create a file called 'greeting.txt' in the current directory that says 'Hello from the Stem Agent!'"
    ]
    
    if os.getenv("SERPER_API_KEY"):
        prompts.append("Search the web for the current CEO of JetBrains and tell me their name.")
    else:
        console.print("[yellow]Notice: SERPER_API_KEY not found. Skipping web search example.[/yellow]")

    for prompt in prompts:
        console.print(f"============================================================")
        console.print(f"[bold yellow]User Prompt:[/bold yellow] {prompt}")
        console.print("[dim]Agent thinking and executing tools...[/dim]\n")
        
        try:
            result = agent_executor.invoke({"messages": [HumanMessage(content=prompt)]})
            
            # Trace the messages to show the thought process
            for message in result['messages']:
                if message.type == "ai" and message.tool_calls:
                    for tc in message.tool_calls:
                        console.print(f"🤖 [bold magenta]Agent Action:[/bold magenta] Calling tool '[cyan]{tc['name']}[/cyan]' with args: {tc['args']}")
                elif message.type == "tool":
                    output = message.content[:250] + "..." if len(message.content) > 250 else message.content
                    console.print(f"🔧 [bold blue]Tool Output:[/bold blue] [dim]{output}[/dim]")
            
            # Print final reply
            final_response = result['messages'][-1].content
            console.print("\n[bold green]Final Response:[/bold green]")
            console.print(Markdown(final_response))
            console.print("\n")
            
        except Exception as e:
            console.print(f"[red]Error during execution: {e}[/red]")

    console.print("[bold]Demo completed successfully![/bold]")

if __name__ == "__main__":
    main()

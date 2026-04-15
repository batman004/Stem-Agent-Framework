"""
Lazy-Loading Orchestrator Node.

Scans the `specialists/` directory, extracts their tool signatures,
and mounts them as Proxy Tools for a Supervisor. The actual python strings
within those specialists are only compiled if/when invoked.
"""

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from loguru import logger

from stem_agent.core.config import config
from stem_agent.tools.registry import DynamicToolRegistry

class SpecialistRouter:
    """Discovers specialist JSON artifacts and generates lazy-loaded proxy tools."""
    
    def __init__(self, specialists_dir: str = "specialists"):
        self.specialists_dir = Path(specialists_dir)
        self.specialists_dir.mkdir(parents=True, exist_ok=True)
        self._proxy_tools: list[StructuredTool] = []
        self._load_proxies()

    def _load_proxies(self) -> None:
        """Scan directory and create a proxy tool for each JSON config."""
        for file in self.specialists_dir.glob("specialist_*.json"):
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                
                name = data.get("name", file.stem)
                desc = data.get("description", "A specialist agent.")
                
                proxy_tool = self._create_proxy(name, desc, file)
                self._proxy_tools.append(proxy_tool)
                logger.info(f"Mounted proxy for specialist: {name}")
            except Exception as e:
                logger.error(f"Failed to load specialist proxy for {file}: {e}")

    def _create_proxy(self, name: str, desc: str, filepath: Path) -> StructuredTool:
        """Create a lazy-loading StructuredTool proxy."""
        
        def run_proxy(task_instruction: str) -> str:
            """Invoke the specialist with the given task instruction."""
            logger.info(f"Rehydrating and dispatching task to specialist: {name}")
            return self._rehydrate_and_run(filepath, task_instruction)
            
        return StructuredTool.from_function(
            func=run_proxy,
            name=name,
            description=desc,
        )

    def _rehydrate_and_run(self, filepath: Path, task_instruction: str) -> str:
        """The actual heavy-lifting. Compiles the custom code and invokes the agent."""
        with open(filepath, "r") as f:
            data = json.load(f)
            
        # Re-build custom tools
        custom_tool_code = data.get("custom_tool_code", {})
        acquired_tool_names = data.get("tools", [])
        
        # Load registry solely to fetch primitives
        registry = DynamicToolRegistry(load_primitives=True)
        active_tools = []
        
        # Hydrate Custom Tools using exec()
        for tool_name, source_code in custom_tool_code.items():
            try:
                namespace: dict[str, Any] = {}
                exec(source_code, namespace)  # noqa: S102
                func = namespace.get(tool_name)
                if func:
                    custom_tool = StructuredTool.from_function(
                        func=func, 
                        name=tool_name, 
                        description=f"Custom dynamically synthesized tool {tool_name}"
                    )
                    registry.register(custom_tool, capabilities=[])
            except Exception as e:
                logger.error(f"Failed to compile custom tool '{tool_name}' for specialist: {e}")
                
        # Resolve all required tools
        for t_name in acquired_tool_names:
            tool = registry.get_by_name(t_name)
            if tool:
                active_tools.append(tool)
            else:
                logger.warning(f"Could not resolve tool: {t_name}")

        # Add sql_query if it's in acquired tools OR if it's a primitive
        # We bind it to the grounded database url if available in metadata
        grounding = data.get("metadata", {}).get("grounding_resources", [])
        db_url = next((r["url"] for r in grounding if r["type"] == "database"), None)
        
        if db_url:
            from stem_agent.tools.primitives import sql_query
            
            def bound_sql_query(query: str) -> str:
                """Execute a read-only SQL query against the specialists' grounded database."""
                return sql_query.invoke({"query": query, "connection_url": db_url})
                
            bound_tool = StructuredTool.from_function(
                func=bound_sql_query,
                name="sql_query",
                description="Query the live database for this specialist to get real-time info."
            )
            active_tools.append(bound_tool)
            logger.info(f"Bound sql_query to {data['name']} using {db_url}")

        system_prompt = data.get("prompt_templates", {}).get("system", "You are a helpful assistant.")
        if db_url:
            system_prompt += f"\n\nLIVE GROUNDING: You are connected to a database at {db_url}. Use 'sql_query' to fetch live data."
        
        from langgraph.prebuilt import create_react_agent
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=config.model)
        
        agent_executor = create_react_agent(llm, tools=active_tools, prompt=system_prompt)
        
        try:
            result = agent_executor.invoke({"messages": [("user", task_instruction)]})
            return result["messages"][-1].content
        except Exception as e:
            logger.error(f"Execution failed for {data.get('name')}: {e}")
            return f"Specialist error: {e}"

    def get_proxy_tools(self) -> list[StructuredTool]:
        return self._proxy_tools

    def get_pending_clarifications(self) -> list[str]:
        """Load any clarifications the stem agent flagged during its last run."""
        clarifications_path = self.specialists_dir / "pending_clarifications.json"
        if clarifications_path.exists():
            try:
                import json
                with open(clarifications_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

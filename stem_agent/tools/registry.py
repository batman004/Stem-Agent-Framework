"""
Dynamic Tool Registry — discover, store, and query tools by capability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from langchain_core.tools import BaseTool
from loguru import logger

from stem_agent.tools.primitives import PRIMITIVE_TOOLS, PRIMITIVE_CAPABILITIES


@dataclass
class ToolEntry:
    """A registered tool with its capability tags."""
    tool: BaseTool
    capabilities: list[str] = field(default_factory=list)
    is_primitive: bool = False
    is_composed: bool = False


class DynamicToolRegistry:
    """
    Stores tools indexed by name and queryable by capability keywords.
    Pre-loaded with primitive tools on init.
    """

    def __init__(self, load_primitives: bool = True):
        self._tools: dict[str, ToolEntry] = {}
        if load_primitives:
            self._register_primitives()

    def _register_primitives(self) -> None:
        for tool in PRIMITIVE_TOOLS:
            caps = PRIMITIVE_CAPABILITIES.get(tool.name, [])
            self._tools[tool.name] = ToolEntry(
                tool=tool, capabilities=caps, is_primitive=True,
            )
        logger.info(f"Registry loaded {len(PRIMITIVE_TOOLS)} primitive tools")

    def register(self, tool: BaseTool, capabilities: list[str] | None = None) -> None:
        """Register a new tool with optional capability tags."""
        caps = capabilities or []
        self._tools[tool.name] = ToolEntry(
            tool=tool, capabilities=caps, is_composed=True,
        )
        logger.info(f"Registered tool '{tool.name}' with capabilities {caps}")

    def query(self, capabilities: list[str]) -> list[BaseTool]:
        """Return tools that match any of the requested capability keywords."""
        matched = []
        for entry in self._tools.values():
            if any(cap in entry.capabilities for cap in capabilities):
                matched.append(entry.tool)
        return matched

    def get_tools_for(self, required_capabilities: list[str]) -> list[BaseTool]:
        """Find tools matching required capabilities. Alias for query."""
        return self.query(required_capabilities)

    def get_by_name(self, name: str) -> BaseTool | None:
        """Look up a tool by exact name."""
        entry = self._tools.get(name)
        return entry.tool if entry else None

    def get_entry(self, name: str) -> ToolEntry | None:
        """Look up a registry entry by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    def list_capabilities(self) -> dict[str, list[str]]:
        """Return a map of tool_name → capabilities."""
        return {name: entry.capabilities for name, entry in self._tools.items()}

    def missing_capabilities(self, required: list[str]) -> list[str]:
        """Return capabilities not covered by any registered tool."""
        all_caps = set()
        for entry in self._tools.values():
            all_caps.update(entry.capabilities)
        return [cap for cap in required if cap not in all_caps]

    @property
    def size(self) -> int:
        return len(self._tools)

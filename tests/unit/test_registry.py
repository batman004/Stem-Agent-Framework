"""Tests for DynamicToolRegistry — query, register, and missing capability detection."""

import pytest
from langchain_core.tools import tool as lc_tool
from stem_agent.tools.registry import DynamicToolRegistry


@lc_tool
def dummy_tool(input_data: str) -> str:
    """A dummy tool for testing."""
    return f"processed: {input_data}"


class TestRegistryInit:
    def test_loads_primitives_by_default(self):
        registry = DynamicToolRegistry()
        assert registry.size == 4
        assert "web_search" in registry.list_tools()

    def test_empty_without_primitives(self):
        registry = DynamicToolRegistry(load_primitives=False)
        assert registry.size == 0


class TestRegistryOperations:
    def test_register_tool(self):
        registry = DynamicToolRegistry(load_primitives=False)
        registry.register(dummy_tool, capabilities=["process", "transform"])
        assert registry.size == 1
        assert "dummy_tool" in registry.list_tools()

    def test_query_by_capability(self):
        registry = DynamicToolRegistry()
        tools = registry.query(["search"])
        names = [t.name for t in tools]
        assert "web_search" in names

    def test_query_multiple_capabilities(self):
        registry = DynamicToolRegistry()
        tools = registry.query(["search", "compute"])
        names = [t.name for t in tools]
        assert "web_search" in names
        assert "python_repl" in names

    def test_query_no_match(self):
        registry = DynamicToolRegistry()
        tools = registry.query(["quantum_entanglement"])
        assert len(tools) == 0

    def test_get_by_name(self):
        registry = DynamicToolRegistry()
        t = registry.get_by_name("python_repl")
        assert t is not None
        assert t.name == "python_repl"

    def test_get_by_name_missing(self):
        registry = DynamicToolRegistry()
        assert registry.get_by_name("nonexistent") is None

    def test_missing_capabilities(self):
        registry = DynamicToolRegistry()
        missing = registry.missing_capabilities(["search", "teleportation"])
        assert "teleportation" in missing
        assert "search" not in missing

    def test_list_capabilities(self):
        registry = DynamicToolRegistry()
        caps = registry.list_capabilities()
        assert "web_search" in caps
        assert "search" in caps["web_search"]

    def test_get_entry_metadata(self):
        registry = DynamicToolRegistry()
        entry = registry.get_entry("web_search")
        assert entry is not None
        assert entry.is_primitive is True
        assert entry.is_composed is False

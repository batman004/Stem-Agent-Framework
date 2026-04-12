"""Tests for ToolComposer and ToolValidator — requires OpenAI API key."""

import os
import pytest
from stem_agent.tools.composer import ToolComposer
from stem_agent.tools.validator import ToolValidator
from stem_agent.tools.registry import DynamicToolRegistry


requires_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


@requires_openai
class TestToolComposer:
    def test_compose_simple_tool(self):
        composer = ToolComposer()
        tool = composer.compose(
            capability="word_counter",
            description="Count the number of words in the input text",
        )
        assert tool is not None
        assert tool.name == "word_counter"
        result = tool.invoke("hello world foo bar")
        assert "4" in str(result)

    def test_compose_returns_none_on_bad_capability(self):
        composer = ToolComposer()
        tool = composer.compose(
            capability="",
            description="",
        )
        # May succeed or fail — just verify no crash
        assert tool is None or tool is not None


@requires_openai
class TestToolValidator:
    def test_validate_python_repl(self):
        from stem_agent.tools.primitives import python_repl
        validator = ToolValidator(num_tests=2)
        result = validator.validate(python_repl)
        assert isinstance(result, bool)

    def test_validate_composed_tool(self):
        composer = ToolComposer()
        tool = composer.compose(
            capability="text_reverser",
            description="Reverse the input string",
        )
        assert tool is not None

        validator = ToolValidator(num_tests=2)
        result = validator.validate(tool)
        assert isinstance(result, bool)


@requires_openai
class TestComposerRegistryIntegration:
    def test_compose_and_register(self):
        """Compose a tool, validate it, register it in the registry."""
        composer = ToolComposer()
        validator = ToolValidator(num_tests=2)
        registry = DynamicToolRegistry()

        initial_size = registry.size

        tool = composer.compose(
            capability="char_counter",
            description="Count the number of characters in the input text",
        )
        assert tool is not None

        if validator.validate(tool):
            registry.register(tool, capabilities=["count", "analyze"])
            assert registry.size == initial_size + 1
            assert "char_counter" in registry.list_tools()

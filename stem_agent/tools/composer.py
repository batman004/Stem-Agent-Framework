"""
Tool Composer — uses LLM to generate new tools from capability descriptions.

Given a capability like "csv_analyzer", the composer asks the LLM to write
a Python function, then wraps it as a LangChain tool.
"""

from __future__ import annotations

import textwrap
from typing import Any

from langchain_core.tools import StructuredTool
from loguru import logger

from stem_agent.core.config import config, get_openai_client

COMPOSE_PROMPT = textwrap.dedent("""\
    Write a Python function that implements the following capability:
    
    Capability: {capability}
    Description: {description}
    Available primitives: {primitives}
    
    Requirements:
    - The function must be named `{func_name}`
    - It must accept a single string argument `input_data` and return a string
    - It should use only the Python standard library
    - Include a one-line docstring
    - Handle errors gracefully with try/except
    - Do NOT import any external packages
    
    Return ONLY the Python function definition, no explanation or markdown fences.
""")


class ToolComposer:
    """Generates new tool functions via LLM and wraps them as LangChain tools."""

    def __init__(self, model: str | None = None):
        self.client = get_openai_client()
        self.model = model or config.model

    def compose(
        self,
        capability: str,
        description: str = "",
        primitive_names: list[str] | None = None,
    ) -> StructuredTool | None:
        """Use LLM to generate a Python function for the capability, wrap as tool."""
        func_name = capability.lower().replace(" ", "_").replace("-", "_")
        desc = description or f"A tool that performs: {capability}"
        primitives = ", ".join(primitive_names or ["python_repl", "web_search"])

        prompt = COMPOSE_PROMPT.format(
            capability=capability,
            description=desc,
            primitives=primitives,
            func_name=func_name,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a Python tool-generation expert. Return only valid Python code."},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=1000,
            )

            code = response.choices[0].message.content.strip()
            code = self._clean_code(code)

            namespace: dict[str, Any] = {}
            exec(code, namespace)  # noqa: S102

            func = namespace.get(func_name)
            if func is None:
                logger.error(f"Composed code did not define function '{func_name}'")
                return None

            tool = StructuredTool.from_function(
                func=func,
                name=func_name,
                description=desc,
            )

            logger.info(f"Composed tool '{func_name}' successfully")
            return tool

        except Exception as e:
            logger.error(f"Tool composition failed for '{capability}': {e}")
            return None

    def _clean_code(self, code: str) -> str:
        """Strip markdown fences and leading/trailing whitespace from LLM output."""
        if code.startswith("```"):
            lines = code.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            code = "\n".join(lines)
        return code.strip()

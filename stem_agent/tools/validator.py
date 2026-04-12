"""
Tool Validator — sandbox-test composed tools before adoption.

Generates synthetic test cases via LLM, runs the tool against them,
and only approves if all pass without exceptions.
"""

from __future__ import annotations

import json

from langchain_core.tools import BaseTool
from loguru import logger

from stem_agent.core.config import config, get_openai_client

TEST_CASE_PROMPT = """\
Generate {n} test cases for a tool with this description:

Tool name: {name}
Description: {description}

Each test case should have:
- "input": a realistic string input for the tool
- "should_contain": a substring the output should reasonably contain (or null if any output is acceptable)

Return ONLY a JSON array, no markdown fences or explanation.
Example: [{{"input": "hello world", "should_contain": "hello"}}]
"""


class ToolValidator:
    """Validates composed tools by running them against LLM-generated test cases."""

    def __init__(self, model: str | None = None, num_tests: int = 3):
        self.client = get_openai_client()
        self.model = model or config.model
        self.num_tests = num_tests

    def validate(self, tool: BaseTool) -> bool:
        """Run tool against synthetic test cases. Returns True if all pass."""
        test_cases = self._generate_test_cases(tool)
        if not test_cases:
            logger.warning(f"No test cases generated for '{tool.name}', failing validation")
            return False

        passed = 0
        for i, tc in enumerate(test_cases):
            try:
                result = tool.invoke(tc["input"])
                result_str = str(result)

                expected = tc.get("should_contain")
                if expected and expected.lower() not in result_str.lower():
                    logger.warning(
                        f"Test {i+1}/{len(test_cases)} for '{tool.name}': "
                        f"output missing expected '{expected}'"
                    )
                else:
                    passed += 1
                    logger.debug(f"Test {i+1}/{len(test_cases)} for '{tool.name}': PASS")

            except Exception as e:
                logger.warning(f"Test {i+1}/{len(test_cases)} for '{tool.name}' raised: {e}")

        success = passed == len(test_cases)
        logger.info(
            f"Validation for '{tool.name}': {passed}/{len(test_cases)} passed — "
            f"{'✅ APPROVED' if success else '❌ REJECTED'}"
        )
        return success

    def _generate_test_cases(self, tool: BaseTool) -> list[dict]:
        """Use LLM to generate synthetic test cases for a tool."""
        prompt = TEST_CASE_PROMPT.format(
            n=self.num_tests,
            name=tool.name,
            description=tool.description,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Return only valid JSON arrays."},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=500,
            )

            content = response.choices[0].message.content.strip()
            content = content.strip("`").lstrip("json").strip()
            cases = json.loads(content)

            if not isinstance(cases, list):
                logger.error("LLM returned non-list test cases")
                return []

            logger.info(f"Generated {len(cases)} test cases for '{tool.name}'")
            return cases

        except Exception as e:
            logger.error(f"Test case generation failed for '{tool.name}': {e}")
            return []

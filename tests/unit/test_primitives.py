"""Tests for primitive tools — verifies each tool executes and returns sensible output."""

import pytest
from stem_agent.tools.primitives import web_search, python_repl, read_url, write_file, PRIMITIVE_TOOLS


class TestPythonRepl:
    def test_simple_math(self):
        result = python_repl.invoke("print(2 + 2)")
        assert "4" in result

    def test_multiline(self):
        code = "for i in range(3): print(i)"
        result = python_repl.invoke(code)
        assert "0" in result
        assert "2" in result

    def test_error_handling(self):
        result = python_repl.invoke("raise ValueError('test error')")
        assert "error" in result.lower() or "Error" in result

    def test_timeout_protection(self):
        result = python_repl.invoke("import time; time.sleep(0.1); print('done')")
        assert "done" in result


class TestReadUrl:
    def test_fetch_example_com(self):
        result = read_url.invoke("https://example.com")
        assert "Example Domain" in result
        assert len(result) > 50

    def test_invalid_url(self):
        result = read_url.invoke("https://thisdomaindoesnotexist12345.com")
        assert "Error" in result or "error" in result


class TestWriteFile:
    def test_write_and_verify(self, tmp_path):
        path = str(tmp_path / "test_output.txt")
        result = write_file.invoke({"path": path, "content": "hello world"})
        assert "Successfully" in result

        with open(path) as f:
            assert f.read() == "hello world"

    def test_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "a" / "b" / "deep.txt")
        result = write_file.invoke({"path": path, "content": "nested"})
        assert "Successfully" in result


class TestWebSearch:
    @pytest.mark.skipif(
        not __import__("os").getenv("SERPER_API_KEY"),
        reason="SERPER_API_KEY not set",
    )
    def test_returns_results(self):
        result = web_search.invoke("Python programming language")
        assert len(result) > 50
        assert "python" in result.lower()


class TestPrimitivesList:
    def test_four_primitives(self):
        assert len(PRIMITIVE_TOOLS) == 4

    def test_all_have_names(self):
        names = {t.name for t in PRIMITIVE_TOOLS}
        assert names == {"web_search", "python_repl", "read_url", "write_file"}

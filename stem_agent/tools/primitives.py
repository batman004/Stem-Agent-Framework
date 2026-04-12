"""
Primitive tools — always-available base capabilities.

These are the "genome" tools that exist before any specialization.
The stem agent can compose higher-level tools from these primitives.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import truststore
truststore.inject_into_ssl()

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.tools import tool
from loguru import logger

load_dotenv()


@tool
def web_search(query: str) -> str:
    """Search the web using Serper.dev and return top results as text."""
    api_key = os.getenv("SERPER_API_KEY", "")
    if not api_key:
        return "Error: SERPER_API_KEY not set"

    try:
        response = httpx.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": 5},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        snippets = []
        for r in data.get("organic", [])[:5]:
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            link = r.get("link", "")
            snippets.append(f"**{title}**\n{snippet}\n{link}")

        output = "\n\n".join(snippets) if snippets else "No results found."
        logger.info(f"web_search: {len(snippets)} results for '{query[:50]}'")
        return output

    except Exception as e:
        logger.error(f"web_search failed: {e}")
        return f"Search error: {e}"


@tool
def python_repl(code: str) -> str:
    """Execute Python code in a sandboxed subprocess and return stdout/stderr."""
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

        output = result.stdout.strip()
        if result.returncode != 0:
            output = f"{output}\nSTDERR: {result.stderr.strip()}" if output else result.stderr.strip()

        logger.info(f"python_repl: exit={result.returncode}, output_len={len(output)}")
        return output or "(no output)"

    except subprocess.TimeoutExpired:
        logger.warning("python_repl: execution timed out (30s)")
        return "Error: execution timed out after 30 seconds"
    except Exception as e:
        logger.error(f"python_repl failed: {e}")
        return f"Execution error: {e}"


@tool
def read_url(url: str) -> str:
    """Fetch a URL and return its text content (HTML stripped)."""
    try:
        response = httpx.get(url, timeout=15, follow_redirects=True)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Truncate to avoid blowing up context windows
        max_chars = 8000
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n... [truncated, {len(text)} total chars]"

        logger.info(f"read_url: fetched {len(text)} chars from {url[:60]}")
        return text

    except Exception as e:
        logger.error(f"read_url failed for {url}: {e}")
        return f"Error fetching URL: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories if needed."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        logger.info(f"write_file: wrote {len(content)} chars to {path}")
        return f"Successfully wrote {len(content)} chars to {path}"
    except Exception as e:
        logger.error(f"write_file failed: {e}")
        return f"Error writing file: {e}"


PRIMITIVE_TOOLS = [web_search, python_repl, read_url, write_file]

PRIMITIVE_CAPABILITIES = {
    "web_search": ["search", "research", "lookup", "find", "query"],
    "python_repl": ["compute", "calculate", "analyze", "code", "execute", "transform"],
    "read_url": ["fetch", "scrape", "read", "download", "extract"],
    "write_file": ["write", "save", "export", "output", "store"],
}
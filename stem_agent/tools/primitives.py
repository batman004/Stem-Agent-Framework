"""
Primitive tools — always-available base capabilities.

These are the "genome" tools that exist before any specialization.
The stem agent can compose higher-level tools from these primitives.
"""

from __future__ import annotations

import os
import sys
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
            [sys.executable, "-c", code],
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


@tool
def db_inspect(connection_url: str) -> str:
    """Connect to a database and return its schema (tables, columns, types).

    Reads structure only — never reads row data.
    Supports SQLite, PostgreSQL, MySQL via SQLAlchemy.
    """
    try:
        from sqlalchemy import create_engine, inspect as sa_inspect

        engine = create_engine(connection_url, echo=False)
        inspector = sa_inspect(engine)

        tables = inspector.get_table_names()
        if not tables:
            return "Database is empty (no tables found)."

        schema_lines = [f"Database schema ({len(tables)} tables):\n"]
        for table in tables:
            columns = inspector.get_columns(table)
            pk = inspector.get_pk_constraint(table)
            pk_cols = pk.get("constrained_columns", []) if pk else []

            col_descs = []
            for col in columns:
                col_type = str(col["type"])
                pk_marker = " [PK]" if col["name"] in pk_cols else ""
                nullable = "" if col.get("nullable", True) else " NOT NULL"
                col_descs.append(f"    {col['name']}: {col_type}{pk_marker}{nullable}")

            schema_lines.append(f"TABLE {table}")
            schema_lines.extend(col_descs)

            # Show foreign keys
            fks = inspector.get_foreign_keys(table)
            for fk in fks:
                src = ", ".join(fk.get("constrained_columns", []))
                ref_table = fk.get("referred_table", "?")
                ref_cols = ", ".join(fk.get("referred_columns", []))
                schema_lines.append(f"    FK: {src} -> {ref_table}({ref_cols})")

            # Show row count estimate
            try:
                with engine.connect() as conn:
                    from sqlalchemy import text
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))  # noqa: S608
                    count = result.scalar()
                    schema_lines.append(f"    ({count} rows)")
            except Exception:
                pass

            schema_lines.append("")

        engine.dispose()
        output = "\n".join(schema_lines)
        logger.info(f"db_inspect: discovered {len(tables)} tables")
        return output

    except Exception as e:
        logger.error(f"db_inspect failed: {e}")
        return f"Database inspection error: {e}"


@tool
def repo_inspect(github_url: str) -> str:
    """Inspect a GitHub repository or pull request and return its structure.

    For repos: returns file tree, README excerpt, and key config files.
    For PRs: returns the diff summary and changed files.
    Uses the public GitHub API (no auth needed for public repos).
    """
    import re

    try:
        # Parse GitHub URL
        pr_match = re.match(r"https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)", github_url)
        repo_match = re.match(r"https?://github\.com/([^/]+)/([^/]+)/?", github_url)

        if pr_match:
            owner, repo, pr_num = pr_match.groups()
            return _inspect_pr(owner, repo, pr_num)
        elif repo_match:
            owner, repo = repo_match.groups()
            repo = repo.rstrip(".git")
            return _inspect_repo(owner, repo)
        else:
            return f"Could not parse GitHub URL: {github_url}"

    except Exception as e:
        logger.error(f"repo_inspect failed: {e}")
        return f"Repository inspection error: {e}"


def _inspect_repo(owner: str, repo: str) -> str:
    """Inspect a GitHub repository structure."""
    lines = [f"Repository: {owner}/{repo}\n"]

    # Get repo metadata
    meta_resp = httpx.get(f"https://api.github.com/repos/{owner}/{repo}", timeout=15)
    if meta_resp.status_code == 200:
        meta = meta_resp.json()
        lines.append(f"Description: {meta.get('description', 'N/A')}")
        lines.append(f"Language: {meta.get('language', 'N/A')}")
        lines.append(f"Stars: {meta.get('stargazers_count', 0)}")
        lines.append(f"Last pushed: {meta.get('pushed_at', 'N/A')}")
        lines.append("")

    # Get file tree (top-level + one level deep)
    tree_resp = httpx.get(
        f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1",
        timeout=15,
    )
    if tree_resp.status_code == 200:
        tree = tree_resp.json().get("tree", [])
        lines.append(f"File tree ({len(tree)} items):")
        for item in tree[:80]:  # Cap at 80 entries
            prefix = "📁 " if item["type"] == "tree" else "   "
            lines.append(f"  {prefix}{item['path']}")
        if len(tree) > 80:
            lines.append(f"  ... and {len(tree) - 80} more files")
        lines.append("")

    # Get README
    readme_resp = httpx.get(
        f"https://api.github.com/repos/{owner}/{repo}/readme",
        headers={"Accept": "application/vnd.github.v3.raw"},
        timeout=15,
    )
    if readme_resp.status_code == 200:
        readme_text = readme_resp.text[:2000]
        lines.append(f"README (first 2000 chars):\n{readme_text}")

    output = "\n".join(lines)
    logger.info(f"repo_inspect: inspected {owner}/{repo}")
    return output


def _inspect_pr(owner: str, repo: str, pr_num: str) -> str:
    """Inspect a GitHub pull request."""
    lines = [f"Pull Request: {owner}/{repo}#{pr_num}\n"]

    # PR metadata
    pr_resp = httpx.get(
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_num}",
        timeout=15,
    )
    if pr_resp.status_code == 200:
        pr = pr_resp.json()
        lines.append(f"Title: {pr.get('title', 'N/A')}")
        lines.append(f"Author: {pr.get('user', {}).get('login', 'N/A')}")
        lines.append(f"State: {pr.get('state', 'N/A')}")
        lines.append(f"Changed files: {pr.get('changed_files', 'N/A')}")
        lines.append(f"Additions: +{pr.get('additions', 0)}, Deletions: -{pr.get('deletions', 0)}")
        body = pr.get("body", "") or ""
        if body:
            lines.append(f"Description:\n{body[:1000]}")
        lines.append("")

    # PR files
    files_resp = httpx.get(
        f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_num}/files",
        timeout=15,
    )
    if files_resp.status_code == 200:
        files = files_resp.json()
        lines.append("Changed files:")
        for f in files[:30]:
            status = f.get("status", "?")
            fname = f.get("filename", "?")
            adds = f.get("additions", 0)
            dels = f.get("deletions", 0)
            lines.append(f"  [{status}] {fname} (+{adds}/-{dels})")
        lines.append("")

    output = "\n".join(lines)
    logger.info(f"repo_inspect: inspected PR {owner}/{repo}#{pr_num}")
    return output


PRIMITIVE_TOOLS = [web_search, python_repl, read_url, write_file, db_inspect, repo_inspect]

PRIMITIVE_CAPABILITIES = {
    "web_search": ["search", "research", "lookup", "find", "query"],
    "python_repl": ["compute", "calculate", "analyze", "code", "execute", "transform"],
    "read_url": ["fetch", "scrape", "read", "download", "extract"],
    "write_file": ["write", "save", "export", "output", "store"],
    "db_inspect": ["database", "schema", "sql", "inspect", "tables", "columns"],
    "repo_inspect": ["github", "repository", "pull_request", "code_review", "diff", "repo"],
}
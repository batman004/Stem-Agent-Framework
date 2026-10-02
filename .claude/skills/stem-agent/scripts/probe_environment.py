#!/usr/bin/env python3
"""Inventory a project and its Claude Code environment for designing a specialist subagent.

Reports languages/manifests, run/test commands, instruction files, existing agents
and skills, configured MCP servers (names and transport only), useful CLIs on PATH,
data files, and env var names. Never prints secret values.

Usage:
    python3 probe_environment.py [--root DIR] [--json] [--max-files N]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SKIP_DIRS = {
    ".git", "node_modules", ".venv", "venv", "env", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "dist", "build", "target", ".next", ".nuxt",
    ".turbo", ".gradle", ".idea", ".vscode", "vendor", ".tox", ".cache", "coverage",
    ".chroma", ".terraform",
}

MANIFESTS = {
    "package.json": "JavaScript/TypeScript (npm)",
    "pnpm-workspace.yaml": "pnpm workspace",
    "deno.json": "Deno",
    "pyproject.toml": "Python",
    "requirements.txt": "Python (pip)",
    "setup.py": "Python (setuptools)",
    "Pipfile": "Python (pipenv)",
    "go.mod": "Go",
    "Cargo.toml": "Rust",
    "pom.xml": "Java (Maven)",
    "build.gradle": "JVM (Gradle)",
    "build.gradle.kts": "Kotlin/JVM (Gradle KTS)",
    "Gemfile": "Ruby",
    "composer.json": "PHP",
    "mix.exs": "Elixir",
    "Package.swift": "Swift",
    "pubspec.yaml": "Dart/Flutter",
    "CMakeLists.txt": "C/C++ (CMake)",
    "Makefile": "Make",
    "Dockerfile": "Docker",
    "docker-compose.yml": "Docker Compose",
    "docker-compose.yaml": "Docker Compose",
    "compose.yaml": "Docker Compose",
    "Chart.yaml": "Helm chart",
    "serverless.yml": "Serverless Framework",
    "dbt_project.yml": "dbt",
    "schema.prisma": "Prisma",
    "alembic.ini": "Alembic migrations",
}

SUFFIX_KINDS = {
    ".tf": "Terraform",
    ".csproj": "C#/.NET",
    ".sln": ".NET solution",
    ".ipynb": "Jupyter notebooks",
    ".proto": "Protobuf",
    ".graphql": "GraphQL",
}

DATA_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".sql", ".csv", ".parquet", ".duckdb"}

INSTRUCTION_FILES = [
    "CLAUDE.md", "CLAUDE.local.md", ".claude/CLAUDE.md", "AGENTS.md",
    ".cursorrules", ".github/copilot-instructions.md", "CONTRIBUTING.md", "README.md",
]

CLIS = [
    "git", "gh", "glab", "jq", "rg", "curl", "make", "just",
    "docker", "kubectl", "helm", "terraform", "pulumi", "aws", "gcloud", "az", "flyctl", "vercel",
    "psql", "mysql", "sqlite3", "duckdb", "redis-cli", "mongosh", "bq", "dbt",
    "node", "npm", "pnpm", "yarn", "bun", "deno", "npx",
    "python3", "uv", "poetry", "pytest", "ruff", "mypy",
    "go", "cargo", "java", "mvn", "gradle", "kotlin", "dotnet", "ruby", "bundle", "php",
    "ffmpeg", "playwright", "claude",
]


def find_root(start: Path) -> Path:
    try:
        out = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return Path(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return start


def walk(root: Path, max_files: int):
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        for name in filenames:
            yield Path(dirpath) / name
            count += 1
            if count >= max_files:
                return


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def frontmatter_fields(path: Path, keys=("name", "description")) -> dict:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    block = text[3:end] if end != -1 else ""
    result = {}
    for key in keys:
        m = re.search(rf"^{key}:\s*(.+)$", block, re.MULTILINE)
        if m:
            val = m.group(1).strip().strip("'\"")
            if val in (">", "|", ">-", "|-"):
                after = block[m.end():].lstrip("\n")
                val = " ".join(line.strip() for line in after.splitlines()[:3] if line.startswith(" "))
            result[key] = val[:160] + ("…" if len(val) > 160 else "")
    return result


def describe_mcp(servers: dict) -> list[dict]:
    out = []
    for name, cfg in (servers or {}).items():
        cfg = cfg if isinstance(cfg, dict) else {}
        transport = cfg.get("type") or ("stdio" if "command" in cfg else "http" if "url" in cfg else "?")
        target = cfg.get("command") or ""
        if target and cfg.get("args"):
            first_args = [a for a in cfg["args"][:3] if isinstance(a, str) and "=" not in a and "://" not in a]
            target = " ".join([target, *first_args])
        if not target and cfg.get("url"):
            target = re.sub(r"//[^/@]+@", "//***@", str(cfg["url"])).split("?")[0]
        out.append({"name": name, "transport": transport, "target": target})
    return out


def probe(root: Path, max_files: int) -> dict:
    home = Path.home()
    report: dict = {"root": str(root)}

    stacks, manifests, data_files, top_dirs = {}, [], [], []
    suffix_counts: dict[str, int] = {}
    for p in walk(root, max_files):
        rel = p.relative_to(root)
        if p.name in MANIFESTS:
            manifests.append(str(rel))
            stacks[MANIFESTS[p.name]] = True
        kind = SUFFIX_KINDS.get(p.suffix)
        if kind:
            stacks[kind] = True
        if p.suffix in DATA_SUFFIXES and len(data_files) < 25:
            data_files.append(str(rel))
        if p.suffix:
            suffix_counts[p.suffix] = suffix_counts.get(p.suffix, 0) + 1
        if rel.parts[:2] == (".github", "workflows"):
            stacks["GitHub Actions"] = True
    try:
        top_dirs = sorted(
            d.name for d in root.iterdir()
            if d.is_dir() and d.name not in SKIP_DIRS and not d.name.startswith(".") and not d.name.endswith(".egg-info")
        )
    except OSError:
        pass
    report["stacks"] = sorted(stacks)
    report["manifests"] = sorted(manifests)[:40]
    report["top_level_dirs"] = top_dirs
    report["top_file_types"] = sorted(suffix_counts.items(), key=lambda kv: -kv[1])[:12]
    report["data_files"] = data_files

    commands: dict[str, dict] = {}
    pkg = read_json(root / "package.json")
    if isinstance(pkg, dict) and pkg.get("scripts"):
        commands["package.json scripts"] = dict(list(pkg["scripts"].items())[:20])
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text(encoding="utf-8", errors="replace")
        tools = sorted(set(re.findall(r"^\[tool\.([\w-]+)", text, re.MULTILINE)))
        scripts = re.search(r"^\[project\.scripts\]\n((?:[^\[\n].*\n?)*)", text, re.MULTILINE)
        commands["pyproject"] = {
            "tool_sections": ", ".join(tools) or "-",
            "project.scripts": (scripts.group(1).strip().replace("\n", "; ") if scripts else "-"),
        }
    makefile = root / "Makefile"
    if makefile.exists():
        targets = re.findall(r"^([A-Za-z0-9_.-]+):(?!=)", makefile.read_text(encoding="utf-8", errors="replace"), re.MULTILINE)
        commands["Makefile targets"] = {"targets": ", ".join(dict.fromkeys(targets[:25]))}
    workflows = root / ".github" / "workflows"
    if workflows.is_dir():
        commands["CI workflows"] = {"files": ", ".join(sorted(p.name for p in workflows.glob("*.y*ml")))}
    report["commands"] = commands

    report["instruction_files"] = [f for f in INSTRUCTION_FILES if (root / f).exists()]
    rules_dirs = [d for d in (".claude/rules", ".cursor/rules") if (root / d).is_dir()]
    report["rules_dirs"] = rules_dirs

    agents = []
    for scope, base in (("project", root / ".claude" / "agents"), ("user", home / ".claude" / "agents")):
        if base.is_dir():
            for f in sorted(base.rglob("*.md")):
                fm = frontmatter_fields(f)
                agents.append({"scope": scope, "file": str(f), **fm})
    report["existing_agents"] = agents

    skills = []
    for scope, base in (("project", root / ".claude" / "skills"), ("user", home / ".claude" / "skills")):
        if base.is_dir():
            for f in sorted(base.rglob("SKILL.md")):
                fm = frontmatter_fields(f)
                skills.append({"scope": scope, "name": fm.get("name", f.parent.name), "description": fm.get("description", "")})
    report["existing_skills"] = skills[:60]

    mcp: list[dict] = []
    project_mcp = read_json(root / ".mcp.json")
    if isinstance(project_mcp, dict):
        mcp += [{"scope": "project (.mcp.json)", **s} for s in describe_mcp(project_mcp.get("mcpServers", {}))]
    claude_json = read_json(home / ".claude.json")
    if isinstance(claude_json, dict):
        mcp += [{"scope": "user (~/.claude.json)", **s} for s in describe_mcp(claude_json.get("mcpServers", {}))]
        proj = (claude_json.get("projects") or {}).get(str(root)) or {}
        mcp += [{"scope": "local (~/.claude.json project)", **s} for s in describe_mcp(proj.get("mcpServers", {}))]
    for settings in (root / ".claude" / "settings.json", root / ".claude" / "settings.local.json", home / ".claude" / "settings.json"):
        data = read_json(settings)
        if isinstance(data, dict) and data.get("enabledMcpjsonServers"):
            mcp.append({"scope": f"enabled in {settings.name}", "name": ", ".join(data["enabledMcpjsonServers"]), "transport": "-", "target": "-"})
    report["mcp_servers"] = mcp

    report["clis_on_path"] = [c for c in CLIS if shutil.which(c)]

    env_names = set()
    for f in (".env.example", ".env.sample", ".env.template", ".env"):
        p = root / f
        if p.exists():
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                m = re.match(r"\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=", line)
                if m:
                    env_names.add(m.group(1))
    report["env_var_names"] = sorted(env_names)

    return report


def to_markdown(r: dict) -> str:
    lines = [f"# Environment probe: `{r['root']}`", ""]

    def section(title, items, fmt=lambda x: f"- {x}"):
        lines.append(f"## {title}")
        if items:
            lines.extend(fmt(i) for i in items)
        else:
            lines.append("- (none found)")
        lines.append("")

    section("Stacks detected", r["stacks"])
    section("Manifests", r["manifests"], lambda m: f"- `{m}`")
    section("Top-level directories", r["top_level_dirs"], lambda d: f"- `{d}/`")
    section("Most common file types", r["top_file_types"], lambda kv: f"- `{kv[0]}`: {kv[1]}")
    section("Data files", r["data_files"], lambda d: f"- `{d}`")

    lines.append("## Run / test / build commands")
    if r["commands"]:
        for src, cmds in r["commands"].items():
            lines.append(f"- **{src}**")
            for k, v in cmds.items():
                lines.append(f"  - `{k}`: {v}")
    else:
        lines.append("- (none found)")
    lines.append("")

    section("Instruction files", r["instruction_files"] + r["rules_dirs"], lambda f: f"- `{f}`")
    section("Existing agents (check for overlap)", r["existing_agents"],
            lambda a: f"- [{a['scope']}] **{a.get('name', '?')}**: {a.get('description', '')}  (`{a['file']}`)")
    section("Existing skills (candidates for `skills:` preload)", r["existing_skills"],
            lambda s: f"- [{s['scope']}] **{s['name']}**: {s['description']}")
    section("Configured MCP servers (reference by name in `mcpServers:`)", r["mcp_servers"],
            lambda s: f"- [{s['scope']}] **{s['name']}** ({s['transport']}) {s['target']}")
    section("CLIs on PATH", [", ".join(r["clis_on_path"])] if r["clis_on_path"] else [])
    section("Env var names (values never read into this report)", [", ".join(r["env_var_names"])] if r["env_var_names"] else [])

    lines.append("Next: read deeper into the parts relevant to the problem area, and verify the commands you plan to hand the agent.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="project directory (defaults to the git root of cwd)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of Markdown")
    ap.add_argument("--max-files", type=int, default=20000, help="cap on files walked")
    args = ap.parse_args()

    root = find_root(Path(args.root).resolve())
    report = probe(root, args.max_files)
    print(json.dumps(report, indent=2) if args.json else to_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())

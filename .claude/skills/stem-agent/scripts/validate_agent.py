#!/usr/bin/env python3
"""Lint Claude Code subagent files before shipping them.

Claude Code silently skips agents with broken frontmatter and silently ignores
unknown fields, so problems otherwise only show up as "the agent never runs".

Checks: frontmatter placement and parse, required fields, name format, known
fields and enum values, tool names, hook script existence and permissions,
embedded secrets, and heuristic prompt-quality signals.

Usage:
    python3 validate_agent.py .claude/agents/my-agent.md [more.md ...]
Exit code 1 if any file has errors.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

KNOWN_FIELDS = {
    "name", "description", "tools", "disallowedTools", "model", "permissionMode",
    "mcpServers", "hooks", "maxTurns", "skills", "initialPrompt", "memory", "effort",
    "background", "omitClaudeMd", "isolation", "color", "experimental",
}
ENUMS = {
    "permissionMode": {"default", "manual", "acceptEdits", "auto", "dontAsk", "bypassPermissions", "plan"},
    "memory": {"user", "project", "local"},
    "effort": {"low", "medium", "high", "xhigh", "max"},
    "isolation": {"worktree"},
    "color": {"red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan"},
}
MODEL_ALIASES = {"sonnet", "opus", "haiku", "fable", "inherit"}
BUILTIN_TOOLS = {
    "Read", "Write", "Edit", "MultiEdit", "Glob", "Grep", "LSP", "Bash", "PowerShell",
    "NotebookEdit", "NotebookRead", "WebFetch", "WebSearch", "TodoWrite", "Skill", "ToolSearch",
    "Agent", "Task", "EnterWorktree", "ExitWorktree", "Monitor", "TaskStop", "SendMessage",
    "Artifact", "SubagentHandback", "BashOutput", "KillShell", "ListAgents",
    "TaskCreate", "TaskGet", "TaskList", "TaskUpdate", "CronCreate", "CronDelete", "CronList",
}
STRIPPED_FOR_SUBAGENTS = {
    "AskUserQuestion", "EndConversation", "EnterPlanMode", "ScheduleWakeup",
    "WaitForMcpServers", "Workflow",
}
SECRET_PATTERNS = [
    (r"sk-[A-Za-z0-9_-]{20,}", "API key (sk-…)"),
    (r"ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}", "GitHub token"),
    (r"AKIA[0-9A-Z]{16}", "AWS access key id"),
    (r"xox[abprs]-[A-Za-z0-9-]{10,}", "Slack token"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private key"),
    (r"[a-z][a-z0-9+.-]*://[^\s:/@]+:[^\s@/$]{3,}@", "credentials in URL"),
    (r"(?i)\b(password|passwd|secret|api_key|token)\s*[:=]\s*['\"]?[^\s'\"$<{]{8,}", "inline secret assignment"),
]
SECTION_HINTS = {
    "grounding": r"(?im)^#+\s*(grounding|environment|context|codebase|system facts)",
    "workflow": r"(?im)^#+\s*(workflow|process|method|procedure|steps)",
    "output contract": r"(?im)^#+\s*(output|report|return|deliverable|response format)",
    "boundaries": r"(?im)^#+\s*(boundar|scope|constraints|guardrails|limits|don'?t)",
}


# ---------------------------------------------------------------- parsing

def split_frontmatter(text: str):
    if not text.startswith("---"):
        return None, text
    m = re.match(r"---[ \t]*\n(.*?)\n---[ \t]*(?:\n|$)", text, re.DOTALL)
    if not m:
        return None, text
    return m.group(1), text[m.end():]


def _scalar(v: str):
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        return [_scalar(x) for x in _split_top(v[1:-1]) if x.strip()]
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        return v[1:-1]
    if v in ("true", "false"):
        return v == "true"
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    return v


def _split_top(s: str) -> list[str]:
    parts, depth, cur, quote = [], 0, "", None
    for ch in s:
        if quote:
            cur += ch
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
            continue
        cur += ch
    if cur.strip():
        parts.append(cur)
    return [p.strip() for p in parts]


def minimal_yaml(block: str) -> dict:
    """Parse top-level keys of a frontmatter block. Nested mappings are kept as raw text."""
    data: dict = {}
    lines = block.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        m = re.match(r"^([A-Za-z_][\w-]*):(?:\s+(.*))?$", line)
        if not m or line[0].isspace():
            raise ValueError(f"cannot parse line {i + 1}: {line!r}")
        key, val = m.group(1), (m.group(2) or "").strip()
        i += 1
        child = []
        while i < len(lines) and (not lines[i].strip() or lines[i][0].isspace() or lines[i].startswith("- ")):
            child.append(lines[i])
            i += 1
        if val in (">", "|", ">-", "|-", ">+", "|+"):
            joiner = " " if val.startswith(">") else "\n"
            data[key] = joiner.join(c.strip() for c in child if c.strip())
        elif val:
            if val.startswith("#"):
                val = ""
            data[key] = _scalar(re.sub(r"\s+#.*$", "", val)) if val else None
        elif child:
            items = [c for c in child if c.strip()]
            if all(re.match(r"^\s*- ", c) for c in items) and not any(":" in c for c in items):
                data[key] = [_scalar(re.sub(r"^\s*- ", "", c)) for c in items]
            else:
                data[key] = {"__raw__": "\n".join(child)}
        else:
            data[key] = None
    return data


def parse(block: str):
    if yaml is not None:
        loaded = yaml.safe_load(block)
        return loaded if isinstance(loaded, dict) else {}
    return minimal_yaml(block)


def as_list(v) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [t for t in _split_top(str(v)) if t]


def hook_commands(hooks) -> list[str]:
    if isinstance(hooks, dict) and "__raw__" in hooks:
        return re.findall(r"command:\s*['\"]?([^'\"\n]+)", hooks["__raw__"])
    found = []

    def rec(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "command" and isinstance(v, str):
                    found.append(v)
                else:
                    rec(v)
        elif isinstance(x, list):
            for v in x:
                rec(v)

    rec(hooks)
    return found


# ---------------------------------------------------------------- checks

def validate(path: Path) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        return [f"cannot read file: {e}"], []

    block, body = split_frontmatter(text)
    if block is None:
        msg = "frontmatter must start on line 1 with '---' and be closed by '---'"
        if text.lstrip().startswith("---"):
            msg += " (found leading whitespace/blank lines before it)"
        return [msg + "; Claude Code will treat this file as documentation"], []

    try:
        fm = parse(block)
    except Exception as e:  # noqa: BLE001
        return [f"frontmatter does not parse ({e}); Claude Code will skip this agent"], []

    for key in fm:
        if key not in KNOWN_FIELDS:
            close = [k for k in KNOWN_FIELDS if k.lower() == key.lower().replace("_", "")]
            hint = f" (did you mean '{close[0]}'?)" if close else ""
            warnings.append(f"unknown field '{key}' will be silently ignored{hint}")

    name = fm.get("name")
    if not name:
        errors.append("missing 'name': the file will be treated as documentation")
    else:
        name = str(name)
        if ":" in name or name.startswith("-"):
            errors.append(f"name '{name}' contains ':' or starts with '-': the file will be skipped")
        elif not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
            warnings.append(f"name '{name}' should be lowercase kebab-case")
        if path.stem != name:
            warnings.append(f"filename '{path.name}' differs from name '{name}' (allowed, but harder to find)")

    desc = fm.get("description")
    if not desc:
        errors.append("missing 'description': the agent will be skipped")
    else:
        desc = str(desc)
        words = len(desc.split())
        if words < 15:
            warnings.append(f"description is only {words} words; the router needs trigger situations and the artifacts it owns")
        if words > 120:
            warnings.append(f"description is {words} words; combined descriptions share a startup budget, aim for ≤ ~80")
        if not re.search(r"(?i)\b(use|when|for|whenever|proactively)\b", desc):
            warnings.append("description doesn't say when to delegate (e.g. 'Use proactively when …')")
        if re.search(r"(?i)^you are\b", desc):
            warnings.append("description reads like a persona; describe trigger situations for the router instead")

    for key, allowed in ENUMS.items():
        if key in fm and fm[key] is not None and str(fm[key]) not in allowed:
            errors.append(f"{key}: '{fm[key]}' is not one of {sorted(allowed)}")
    if fm.get("permissionMode") == "bypassPermissions":
        warnings.append("permissionMode: bypassPermissions only applies when the parent already bypasses; avoid it in generated agents")

    model = fm.get("model")
    if model is not None and str(model) not in MODEL_ALIASES and not str(model).startswith("claude-"):
        warnings.append(f"model '{model}' is not an alias {sorted(MODEL_ALIASES)} or a 'claude-…' ID")

    if "maxTurns" in fm and not (isinstance(fm["maxTurns"], int) and fm["maxTurns"] > 0):
        errors.append("maxTurns must be a positive integer")
    for key in ("background", "omitClaudeMd"):
        if key in fm and not isinstance(fm[key], bool):
            errors.append(f"{key} must be true or false")

    tools = as_list(fm.get("tools"))
    disallowed = as_list(fm.get("disallowedTools"))
    for t in tools + disallowed:
        base = t.split("(", 1)[0].strip()
        if base.startswith("mcp__"):
            continue
        if base in STRIPPED_FOR_SUBAGENTS:
            warnings.append(f"tool '{base}' is never available to subagents; return questions to the parent instead")
        elif base == "ExitPlanMode" and fm.get("permissionMode") != "plan":
            warnings.append("ExitPlanMode is only available with permissionMode: plan")
        elif base not in BUILTIN_TOOLS:
            warnings.append(f"tool '{t}' is not a known built-in; check spelling (unresolved tools can stop the agent launching)")
        if t in disallowed and "(" in t:
            warnings.append(f"disallowedTools entry '{t}' removes the whole '{base}' tool, not just matching calls")
    if "tools" not in fm and "disallowedTools" not in fm:
        warnings.append("no 'tools' allowlist: the agent inherits every tool. Specialists usually need a least-privilege list")
    if tools and "mcpServers" in fm and not any(t.startswith("mcp__") for t in tools):
        warnings.append("'tools' allowlist has no mcp__<server> entries, so the MCP servers in 'mcpServers' may be unusable")

    root = find_project_root(path)
    for cmd in hook_commands(fm.get("hooks")):
        script = cmd.strip().split()[0]
        if "/" not in script or script.startswith("$"):
            continue
        sp = Path(os.path.expanduser(script))
        candidates = [sp] if sp.is_absolute() else [root / sp, path.parent / sp, Path.cwd() / sp]
        hit = next((c for c in candidates if c.exists()), None)
        if hit is None:
            errors.append(f"hook script not found: {script}")
        elif not os.access(hit, os.X_OK):
            errors.append(f"hook script is not executable (chmod +x {hit}); the hook will fail instead of blocking")

    for pattern, label in SECRET_PATTERNS:
        m = re.search(pattern, text)
        if m:
            errors.append(f"possible {label} embedded in the file near: {m.group(0)[:24]}…; reference an env var instead")

    body_stripped = body.strip()
    if not body_stripped:
        errors.append("empty body: the system prompt is the agent's whole context")
    else:
        if len(body_stripped) < 600:
            warnings.append(f"body is only {len(body_stripped)} chars; it's likely too generic to beat the main agent")
        missing = [label for label, rx in SECTION_HINTS.items() if not re.search(rx, body)]
        if missing:
            warnings.append(f"body has no section for: {', '.join(missing)}")
        if not re.search(r"`[^`]*[/.][^`]*`", body):
            warnings.append("body has no concrete paths or commands in backticks; is it grounded in this environment?")
        if re.search(r"<[A-Za-z][^<>\n]*[ -][^<>\n]*>|\bTODO\b|\bTBD\b", body):
            warnings.append("body still contains template placeholders or TODOs")
        if re.search(r"(?i)\b(let me know|would you like|feel free to ask)\b", body):
            warnings.append("body talks to the user; subagents report to the parent agent")
        if fm.get("memory") and not re.search(r"(?i)memory", body):
            warnings.append("memory is enabled but the body never says what to remember or when to consult it")
    return errors, warnings


def find_project_root(path: Path) -> Path:
    for parent in [path.resolve().parent, *path.resolve().parents]:
        if (parent / ".git").exists() or parent.name == ".claude":
            return parent.parent if parent.name == ".claude" else parent
    return Path.cwd()


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    failed = False
    for arg in argv:
        path = Path(arg)
        files = sorted(path.glob("*.md")) if path.is_dir() else [path]
        for f in files:
            errors, warnings = validate(f)
            status = "FAIL" if errors else ("WARN" if warnings else "OK")
            print(f"[{status}] {f}")
            for e in errors:
                print(f"  error: {e}")
            for w in warnings:
                print(f"  warn:  {w}")
            failed |= bool(errors)
    if yaml is None:
        print("(PyYAML not installed: used a minimal parser. Also run `claude plugin validate .claude/agents` for a full YAML check.)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

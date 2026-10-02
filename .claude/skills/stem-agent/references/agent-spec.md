# Agent spec: choosing the capability profile

How to fill in each frontmatter field of a Claude Code subagent, and why. Field names are case-sensitive camelCase. Claude Code **silently ignores unknown fields** and **silently skips files whose YAML doesn't parse**, so lint with `scripts/validate_agent.py`.

## Contents

1. File basics
2. Field reference
3. Tools: least privilege by role
4. Model and effort
5. Permission mode and isolation
6. MCP servers
7. Skills
8. Memory
9. Hooks (guardrails)
10. Description: writing for the router

---

## 1. File basics

```markdown
---
name: kebab-case-name
description: When to delegate to this agent…
tools: Read, Grep, Glob, Bash
model: sonnet
---

System prompt body in Markdown.
```

- `---` must be the very first line of the file.
- Locations: `.claude/agents/` for the project (check it into version control), or `~/.claude/agents/` for all projects. A project agent overrides a user agent with the same name.
- Claude Code watches these directories and picks up edits within seconds. A directory **created** after the session started needs a restart.
- Before shipping, `claude plugin validate .claude/agents` will also check that the YAML parses.

## 2. Field reference

| Field | Required | Values | Notes |
| --- | --- | --- | --- |
| `name` | yes | lowercase, digits, hyphens | No `:` and no leading `-`, or the file is skipped. Doesn't have to match the filename, but matching is tidy. |
| `description` | yes | text | The routing signal. See section 10. |
| `tools` | no | comma string or YAML list | Allowlist. If omitted, inherits everything. |
| `disallowedTools` | no | same | Denylist, applied before `tools`. An entry like `Bash(git push *)` removes **all** of Bash. |
| `model` | no | `sonnet`, `opus`, `haiku`, `fable`, full ID, `inherit` | |
| `effort` | no | `low`, `medium`, `high`, `xhigh`, `max` | Overrides the session level. |
| `permissionMode` | no | `default`, `acceptEdits`, `auto`, `dontAsk`, `plan`, `bypassPermissions` | Ignored when the parent session is in `bypassPermissions`, `acceptEdits` or `auto`. |
| `maxTurns` | no | int | Caps runaway loops. When hit, the output is marked partial and can be resumed. |
| `skills` | no | list of skill names | Full skill content is injected at startup. |
| `mcpServers` | no | list of names or inline configs | Inline servers connect only for this agent. |
| `hooks` | no | `PreToolUse`, `PostToolUse`, `Stop` | Scoped to this agent. |
| `memory` | no | `project`, `local`, `user` | Gives the agent a persistent memory dir with a `MEMORY.md` index. |
| `isolation` | no | `worktree` | Runs in a temporary git worktree. |
| `background` | no | bool | Always run in the background. |
| `omitClaudeMd` | no | bool | Launch without CLAUDE.md files. |
| `color` | no | `red`, `blue`, `green`, `yellow`, `purple`, `orange`, `pink`, `cyan` | |
| `initialPrompt` | no | text | Used only when the agent runs as the main session (`--agent`). |

## 3. Tools: least privilege by role

Fewer tools means less context spent on tool descriptions, fewer wrong turns and smaller blast radius. Start from a role profile and adjust:

| Role | `tools` | Typical additions |
| --- | --- | --- |
| Read-only analyst or auditor | `Read, Grep, Glob` | `Bash` (guarded by a hook) for queries and introspection |
| Researcher | `Read, Grep, Glob, WebSearch, WebFetch` | domain MCP servers |
| Implementer | `Read, Grep, Glob, Edit, Write, Bash` | `LSP`, `NotebookEdit`, `TodoWrite` for long tasks |
| Reviewer | `Read, Grep, Glob, Bash` | `mcp__github` for PR context |
| Ops / runbook executor | `Read, Grep, Glob, Bash` | ops MCP servers. Consider `permissionMode: default` so mutations prompt. |
| Coordinator (main session via `--agent`) | `Agent(worker-a, worker-b), Read` | |

Facts about tool resolution:

- Built-in names: `Read`, `Write`, `Edit`, `Glob`, `Grep`, `LSP`, `Bash`, `PowerShell`, `NotebookEdit`, `WebFetch`, `WebSearch`, `TodoWrite`, `Skill`, `ToolSearch`, `Agent`, `EnterWorktree`, `ExitWorktree`, `Monitor`, `TaskStop`, `SendMessage`, `Artifact`.
- Subagents never get `AskUserQuestion`, `EnterPlanMode`, `ScheduleWakeup`, `Workflow` or `EndConversation`. `ExitPlanMode` is available only with `permissionMode: plan`. Design the agent to **return questions to the parent** instead of asking the user.
- Background subagents (the default in interactive sessions) keep only the built-in tools listed above, plus all MCP tools.
- MCP tools use the form `mcp__<server>__<tool>`. Write `mcp__<server>` to grant the whole server.
- If nothing in `tools` resolves, the agent fails to launch. Check spelling.
- Omitting `Skill` from `tools` stops the agent from invoking unlisted skills at runtime. Preloaded `skills:` still load.
- Prefer an allowlist (`tools`) for specialists. Use `disallowedTools` only when the agent truly needs "everything except X", for example to inherit all MCP tools minus one server.

## 4. Model and effort

| Choose | When |
| --- | --- |
| `haiku` | High-volume, well-specified lookups, triage, formatting and log scanning, where speed and cost matter more than depth |
| `sonnet` | Most specialists: coding, analysis, reviews |
| `opus` | Deep reasoning: architecture, security analysis, subtle debugging, novel or ambiguous problems |
| `inherit` | Quality should track whatever the user runs the main session on |

Set `effort` only when the work clearly calls for it: `high` for investigations where missing something is costly, `low` for mechanical tasks. Otherwise, leave it out.

## 5. Permission mode and isolation

- Read-only specialists: an allowlist without `Edit`/`Write` is the real control. `permissionMode: plan` adds a belt-and-braces read-only mode.
- Implementers working on code in the repo: `acceptEdits` cuts down on prompt fatigue. Pair it with `isolation: worktree` when the agent makes sweeping or experimental changes, so the user can review a branch.
- Anything that touches shared or prod systems: leave `permissionMode` unset or use `default`, so mutating calls surface for approval, and add hooks for hard blocks.
- Don't use `bypassPermissions` in a generated agent. It only applies when the parent already bypasses, and it signals the wrong intent.

## 6. MCP servers

- Reference a server **by name** when it's already configured (the probe lists them). This shares the parent's connection: `mcpServers: [github]`.
- Define one **inline** when only this specialist needs it. That keeps its tool descriptions out of the main context:

  ```yaml
  mcpServers:
    - postgres:
        type: stdio
        command: npx
        args: ["-y", "@modelcontextprotocol/server-postgres", "${DATABASE_URL}"]
  ```

  Use env var references, never literal credentials. Inline servers in project agent files load only after the user has trusted the folder.
- Ignored in plugin agents. Prefer a CLI available on PATH (`psql`, `gh`, `kubectl`) plus a guard hook when no good MCP server exists. CLIs are transparent and easy to verify.
- Research well-maintained MCP servers for the domain when the probe shows a gap, and mention them as optional additions in the summary rather than installing them silently.

## 7. Skills

- `skills: [name, …]` injects the **full** content of each skill at startup. It's great for conventions and domain playbooks the agent always needs, and costly if they're large and rarely needed.
- Skills with `disable-model-invocation: true` can't be preloaded.
- Create a companion skill (`<agent-name>-knowledge`) when the domain knowledge is longer than ~150 lines or is shared by several agents in a team. Keep the agent body for identity, workflow and contract.
- Skills that are useful only sometimes: don't preload them. Mention them in the body ("for X, invoke the `foo` skill") and keep `Skill` in `tools`.

## 8. Memory

`memory: project` (the recommended default when used) gives the agent `.claude/agent-memory/<name>/` with a `MEMORY.md` index. Its first ~200 lines are auto-injected, and Read/Write/Edit are enabled for it.

Enable memory when the specialist is used repeatedly on the **same evolving system** and learns things that aren't in the code: recurring failure signatures, decisions, flaky areas, data quirks. Skip it for one-shot or stateless jobs.

When it's enabled, add a short memory section to the body: what's worth saving (non-obvious, durable and verified facts), what isn't (anything derivable from the code, or secrets), and an instruction to check memory before starting. Use `local` for knowledge that shouldn't be committed, and `user` for cross-project expertise.

## 9. Hooks (guardrails)

Use a `PreToolUse` hook when the agent needs a tool but only a safe subset of it. The hook receives JSON on stdin (`.tool_input.command` for Bash). Exit code `2` blocks the call and shows stderr to the agent.

```yaml
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./.claude/hooks/<agent-name>/guard.sh"
```

Common guards:

- **Read-only SQL**: block `INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT`.
- **No pushes or force operations**: block `git push`, `--force` and `reset --hard`.
- **Environment fence**: block commands that mention prod contexts or profiles (`--context prod`, `AWS_PROFILE=prod`).
- **Path fence**: block writes outside the module the agent owns. Use a `PreToolUse` hook on `Edit|Write` that checks `.tool_input.file_path`.

Use `PostToolUse` on `Edit|Write` to run a formatter or linter automatically. Use `Stop` for a final check, for example running the module's tests.

Keep the scripts small, and make them executable (`chmod +x`). Use `jq` if the probe found it, otherwise a `python3 -c` one-liner. Ignored in plugin agents.

## 10. Description: writing for the router

The parent reads every agent's description to decide where to delegate, so treat it as the agent's API. A good one:

- opens with the **situations** that should trigger it, not the persona;
- names the concrete **artifacts** it owns (tables, services, directories, file types);
- says what it **returns**;
- includes "Use proactively when…" if it should fire unprompted, for example after edits to its area;
- stays under ~80 words, because combined descriptions share a startup budget;
- doesn't overlap with sibling agents. If two descriptions could match the same request, sharpen the boundary.

Weak: `Expert in databases.`

Strong: `Analyses and optimises queries against the orders Postgres DB (schema in db/migrations, models in app/models). Use proactively when a query is slow, an EXPLAIN plan needs reading, or a migration touches indexes. Returns ranked findings with the plan evidence and a proposed migration; never applies schema changes itself.`

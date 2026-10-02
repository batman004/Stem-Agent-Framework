---
name: stem-agent
description: Differentiate a problem area into a specialised Claude Code subagent. Probes the real project and environment, researches the domain, then writes a ready-to-use `.claude/agents/<name>.md` with a grounded system prompt, least-privilege tools, MCP servers, preloaded skills, memory and guardrail hooks, and test-drives it on a realistic task. Use this whenever the user wants to create, generate, design or scaffold a subagent, specialist, expert agent or agent team for a domain or sub-problem (e.g. "make me an agent for our Postgres migrations", "I need a specialist for flaky test triage", "spin up an expert on the billing service"), or wants to fix a subagent that underperforms, even if they never say the word "subagent".
---

# Stem Agent

Turn a problem area into a specialist subagent that is grounded in *this* environment rather than a generic persona. The name comes from stem cells: start undifferentiated, sense the surroundings, then commit to a specialised form that fits them.

## Why specialists fail, and what this skill fixes

A subagent starts cold. It gets its own system prompt, the CLAUDE.md files and basic environment details. It does **not** get the parent conversation, the Claude Code system prompt, or the ability to ask the user questions. The parent sees only its final message. So everything that makes a specialist good has to live in the agent file: the facts, the method, the quality bar and the shape of the answer.

The typical failure is "You are an expert in X" plus every inherited tool. That agent is no better than the main thread. It re-discovers the same facts on every run, and it burns context on tools it never needs. A good specialist is better because it:

- **knows things**: exact paths, schemas, commands, conventions and gotchas that were verified in this environment;
- **thinks like a practitioner**: domain heuristics, decision rules and the common failure modes;
- **has exactly the right capabilities**: the tools, MCP servers, skills and guardrails the job needs, and nothing more;
- **returns something the parent can use**: a clear output contract with evidence.

## Pipeline

Work through these phases in order. Keep the user informed with a sentence or two at each phase boundary. Don't stop for approval unless a decision is risky or truly the user's to make.

### 0. Intake

From the request, pull out:

- **Problem area**: what the specialist owns. Restate it in one sentence. If it is broad ("our backend"), plan to narrow it or split it in phase 3.
- **Scope**: the project's `.claude/agents/` when the area is tied to this repo (the default). Use `~/.claude/agents/` when the specialist is useful across projects.
- **Autonomy**: is it a read-only advisor or analyst, or an implementer that edits files and runs commands? This drives most of the capability decisions.
- **Named resources**: databases, services, repos, dashboards, docs URLs and tickets.

Ask at most one round of questions, and only about things that change the design and can't be discovered. Autonomy is the usual one, along with access to external systems. For everything else, pick a sensible default and list it under assumptions in the final summary. Ambiguity you settle now is ambiguity the subagent never has to guess at.

### 1. Probe the environment

Run the inventory script from this skill's directory:

```bash
python3 <skill-dir>/scripts/probe_environment.py --root <project-root>
```

It reports languages and manifests, test and build commands, CLAUDE.md and rules files, existing agents and skills, configured MCP servers (names only, never secrets), useful CLIs on PATH, data files and env var names.

Then **go deep on the parts relevant to the problem area**. This is where the specialist's edge comes from, so spend real effort here. Some examples:

| Area | Look for |
| --- | --- |
| Database / data | schema (run read-only introspection), migrations dir, ORM models, connection env var names, row counts, naming conventions |
| A service / module | entry points, config loading, public interfaces, test layout and how to run one test, logging and error patterns |
| CI / release | workflow files, required checks, caching, deploy steps, secrets *names* |
| Infra | IaC modules, environments, state backends, which commands are safe to run |
| Frontend | framework and version, component conventions, styling system, test and storybook setup |
| External system | which MCP server or CLI reaches it, auth method, rate limits, read-only and mutating operations |

Verify the commands you plan to hand the agent with cheap, read-only runs (`--help`, listing tests, a `SELECT` with `LIMIT`). Keep a list of each fact with its source, because facts you didn't verify become hallucinations in the prompt. Record env var **names**, never values.

Check the existing agents for overlap. If one already covers most of this area, propose extending it instead of creating a near-duplicate. Overlapping descriptions make delegation unpredictable.

### 2. Research the domain

Use WebSearch and WebFetch for what a senior practitioner knows and a generalist misses: official docs for the **exact versions** found in the probe, known failure modes, checklists, standards and performance or security pitfalls. Distil the findings into heuristics and decision rules, and don't paste articles. Skip this phase only when the area is purely internal and the probe already covered it.

### 3. Architect

Decide between **one agent and a small team**. Default to one. Split only when the sub-problems need different permissions (a read-only auditor and a writer), different models, or more knowledge than one prompt can hold. Keep a team to about four agents, each with a crisp, non-overlapping boundary.

For each agent, settle:

- **Mission and boundary**: what it owns, and what it hands back to the parent.
- **Inputs**: what the parent should pass in the delegation prompt.
- **Output contract**: the exact shape of the final message.
- **Capability profile**: `tools`/`disallowedTools`, `model`, `effort`, `permissionMode`, `mcpServers`, `skills`, `memory`, `hooks`, `isolation`, `maxTurns`, `color`.

Read `references/agent-spec.md` for how to choose each field. Share a compact design brief with the user (name, mission, capabilities and why), then continue.

### 4. Synthesize

Start from `assets/agent-template.md` and read `references/prompt-anatomy.md` for the section-by-section guide and a worked example. Principles:

- **Ground every section.** Each section should contain something that is only true of this environment or domain. If a line would fit any project, cut it or make it specific.
- **Explain the why** behind rules, so the agent can handle cases the rules don't cover.
- **Write the description for the router.** It decides when Claude delegates. Lead with the trigger situations, name the concrete artifacts (paths, tables, services), and add "Use proactively when…" if it should fire without being named.
- **Mind the size.** A body of 100–300 lines is typical. When domain knowledge outgrows that, move it into a companion skill at `.claude/skills/<agent-name>-knowledge/SKILL.md` and preload it with `skills:`, or put reference files in the repo and tell the agent when to read them.
- **Add guardrails where an allowed tool has a dangerous subset.** For example, Bash against a database gets a read-only query hook. Put hook scripts in `.claude/hooks/<agent-name>/` and `chmod +x` them.
- **Never embed secrets.** Reference env vars and MCP servers by name.

### 5. Validate and trial

1. Lint every file you wrote:

   ```bash
   python3 <skill-dir>/scripts/validate_agent.py .claude/agents/<name>.md
   ```

   Fix the errors, because Claude Code silently skips agents with bad frontmatter. Weigh each warning before ignoring it.

2. Write 2–3 realistic **benchmark tasks**: the kind of request the parent would actually delegate, each with a note on what a great answer contains, taken from the agent's quality bar.

3. **Run at least one.** Delegate it to the new agent by name. If you created the `agents` directory during this session, Claude Code won't hot-load it until restart. In that case, spawn a general-purpose subagent and pass the agent body as its instructions. Prefer read-only tasks for the trial, or set `isolation: worktree` for implementers.

4. **Grade it honestly** against the quality bar and read the transcript, not only the answer:
   - Did it re-discover facts? Add them to the grounding section.
   - Did it reach for a missing tool, or flail with an unsuitable one? Fix the capabilities.
   - Did it ignore a heuristic? Explain the reasoning behind it better.
   - Was the return shape unusable for a parent? Tighten the output contract.

   Revise, then re-run once if the gaps were meaningful. Stop after two iterations unless the user wants more.

### 6. Deliver

Finish with a summary for the user:

- the files created or changed (agent, companion skill, hook scripts);
- one line on what the specialist is for, and how to invoke it: automatic delegation through its description, `@agent-<name>`, "use the <name> subagent", or `claude --agent <name>` for a whole session;
- its capability profile (tools, model, MCP servers, memory, hooks) with a short reason for each non-default choice;
- the trial result: the benchmark task and how it did against the bar;
- the assumptions you made and the most valuable next improvements;
- a restart note if a new `agents` directory was created.

## Improving an existing agent

Run the same pipeline, but start by reading the current file and, if possible, a transcript where it underperformed. Snapshot the original first. In phase 5, trial the old and new versions on the same task, so the improvement is shown rather than assumed.

## Files in this skill

- `scripts/probe_environment.py`: environment inventory (Markdown, or `--json`).
- `scripts/validate_agent.py`: frontmatter and quality lint for agent files.
- `references/agent-spec.md`: every frontmatter field, plus how to pick tools, model, MCP, skills, memory, hooks and isolation. Read it in phase 3.
- `references/prompt-anatomy.md`: system prompt structure, a worked example and anti-patterns. Read it in phase 4.
- `assets/agent-template.md`: the skeleton to fill in.

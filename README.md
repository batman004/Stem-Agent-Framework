# Stem Agent Framework

A self-specializing AI agent framework inspired by biological stem cells. Give it a problem domain and real infrastructure (databases, repos), and it autonomously differentiates into a team of specialist agents grounded in your actual systems.

## How It Works

The framework follows a biological metaphor: a single **stem agent** probes an unfamiliar domain, builds internal structure, trains against benchmarks, and then **branches** into permanent specialist agents — just like a stem cell differentiating into specialized tissue.

```
YAML Domain Config
       │
       ▼
┌─────────────┐     Inspects DBs, repos, URLs
│  PROBING    │────────────────────────────────►  Resource Context
└─────┬───────┘
      ▼
┌─────────────┐     Web search + LLM analysis
│ ARCHITECTING│────────────────────────────────►  Sub-problems + Tools
└─────┬───────┘
      ▼
┌─────────────┐     ReAct agents run benchmarks
│  EXECUTING  │◄──────────────┐
└─────┬───────┘               │
      ▼                       │  iterate until
┌─────────────┐               │  competent
│ EVALUATING  │───────────────┘
└─────┬───────┘
      ▼
┌─────────────┐
│  BRANCHING  │────────────────────────────────►  Specialist JSON Artifacts
└─────┬───────┘
      ▼
┌─────────────┐
│  COMPLETE   │────────────────────────────────►  Orchestrator CLI
└─────────────┘
```

**Key idea:** The agent discovers your infrastructure *before* it specializes. A restaurant domain YAML pointing at a SQLite DB produces specialists that know about `orders`, `menu_items`, and `shifts`. A code review domain pointing at a GitHub repo produces specialists that know about `docs/*.rst`, `.github/workflows`, and Flask's API surface.

## Quickstart

### 1. Install

```bash
# Recommended: keep packaging tooling current
python -m pip install -U pip setuptools wheel

# Editable install (for development)
python -m pip install -e .

# If you're using conda and hit:
#   AssertionError: .../lib/python3.11/distutils/core.py
# then install with:
SETUPTOOLS_USE_DISTUTILS=stdlib python -m pip install -e .
```

### 2. Configure

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
SERPER_API_KEY=...        # optional, for web search grounding
```

### 3. Seed test data (optional)

```bash
python -m scripts.seed_restaurant_db
```

This creates `data/restaurant.db` with 10 tables of realistic restaurant data (orders, inventory, staff, menus).

### 4. Run differentiation

```bash
# Via the CLI (recommended — interactive)
python -m demos.orchestrator_cli

# Or standalone for a specific domain
python -m stem_agent.core.graph domains/restaurant_ops.yaml
```

### 5. Chat with your specialists

```bash
python -m demos.orchestrator_cli
```

The CLI loads all specialist agents from `specialists/` and routes your queries to the right one.

## CLI Commands

| Command | Description |
|---|---|
| `/domains` | List available domain YAML configs |
| `/specialists` | List loaded specialist proxies |
| `/differentiate <domain>` | Run differentiation for a domain |
| `/switch <domain>` | Switch to a different domain |
| `/clear` | Wipe all specialists and start fresh |
| `exit` | Quit |

## Creating a New Domain

1. Copy `domains/template.yaml` to `domains/your_domain.yaml`
2. Set `task_class` and `description`
3. Optionally add `resources` (databases, GitHub repos, URLs)
4. Run `/differentiate your_domain` from the CLI

```yaml
task_class: "ecommerce_ops"
description: >
  Operational management for an online store including order
  fulfillment, product catalog, and customer support.

resources:
  - type: database
    url: "postgresql://user:pass@localhost:5432/shop"
    label: "Shop database"
  - type: github_repo
    url: "https://github.com/your-org/storefront"
    label: "Storefront codebase"
```

Supported resource types: `database` (any SQLAlchemy URL), `github_repo`, `github_pr`, `url`.

## Project Structure

```
stem_agent/
  core/
    graph.py          State machine: probe → architect → execute → evaluate → branch
    state.py          Pydantic models for agent state, sub-problems, checkpoints
    orchestrator.py   Lazy-loading specialist router and proxy tool generation
    benchmark.py      LLM-generated benchmark tasks for competence evaluation
    evaluator.py      LLM-as-judge scoring
    config.py         Environment-based configuration (models, thresholds)
    skill_store.py    ChromaDB cache for domain models
  tools/
    primitives.py     Base tools: web_search, python_repl, read_url, db_inspect, repo_inspect
    composer.py       LLM-powered tool generation from capability descriptions
    registry.py       Dynamic tool registry with capability-based lookup
    validator.py      Sandbox testing of composed tools before adoption

domains/              YAML configs for each problem domain
  restaurant_ops.yaml
  code_review.yaml
  security_audit.yaml
  template.yaml       Annotated template for new domains

demos/
  orchestrator_cli.py Unified CLI entry point

specialists/          Auto-generated JSON artifacts (one per specialist agent)

scripts/
  seed_restaurant_db.py  Generate test SQLite database

tests/unit/           Unit tests
```

## Architecture

### Differentiation Pipeline (`graph.py`)

Built on [LangGraph](https://github.com/langchain-ai/langgraph) as a state machine with six phases:

- **Probing** — Inspects user-provided resources (DB schemas, repo trees) and searches the web. Feeds real infrastructure context into the LLM domain analysis.
- **Architecting** — Identifies 3-5 sub-problems, composes custom tools for each via LLM, validates them in a sandbox, and generates benchmark tasks.
- **Executing** — Runs ReAct agents against benchmark tasks using the composed tools.
- **Evaluating** — LLM-as-judge scores outputs against quality rubrics. Feedback is fed back into the next iteration.
- **Branching** — Sub-problems that exceed the competence threshold (default 0.75) are exported as standalone specialist JSON artifacts.
- **Complete** — All specialists saved. Pending clarifications (e.g., inaccessible DBs) are written for the CLI to surface.

### Orchestrator (`orchestrator.py`)

Scans `specialists/` for JSON artifacts and creates lazy-loaded proxy tools. A supervisor ReAct agent routes user queries to the right specialist. Custom tools are rehydrated from source code stored in the artifact.

### Tool Lifecycle

1. **Primitive tools** (`primitives.py`) — always available: `web_search`, `python_repl`, `read_url`, `write_file`, `db_inspect`, `repo_inspect`
2. **Composed tools** (`composer.py`) — LLM writes Python functions for capabilities like "threshold-based alerting"
3. **Validated** (`validator.py`) — synthetic test cases ensure the tool works before adoption
4. **Registered** (`registry.py`) — capability-tagged for lookup during architecture phase

## Included Domains

| Domain | Resource | Specialists Produced |
|---|---|---|
| `restaurant_ops` | SQLite DB (10 tables) | Menu costing, inventory planning, staff scheduling, supplier management, sales monitoring |
| `code_review` | GitHub repo (Flask) | Security detection, PR diff analysis, code quality, CI/dependency audit, docs review |
| `security_audit` | None (web-only) | Varies based on web research |

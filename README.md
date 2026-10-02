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

## Creating Custom Specialists with Claude Code

The framework is also useful for creating specialized Claude Code agents grounded in a specific project. Use the `stem-agent` skill to create experts for your codebase.

### Example: Testing Guide Agent

A specialist that helps developers understand and write tests for the Stem Agent Framework itself.

```bash
/stem-agent
# → Probes the codebase: finds pytest setup, test organization (class-based), Pydantic v2 patterns
# → Researches: pytest best practices (but skips since testing is internal)
# → Architects: single read-only advisor with Read, Grep, Bash tools
# → Generates: .claude/agents/testing-guide.md with grounded examples from test_state.py
```

**Invoke**: `@agent-testing-guide` or ask "How do I test a Pydantic model in this codebase?"

**Output**: Concrete guidance with file paths, line numbers, and patterns from actual tests.

```markdown
### Pattern & Example
**File**: `tests/unit/test_state.py`

The pattern follows three layers: validation → behavior → serialization.

class TestSubProblemState:
    def test_serialization_roundtrip(self):
        sp = SubProblemState(name="test", description="test", competence_score=0.5)
        data = sp.model_dump()  # Use model_dump(), not dict()
        restored = SubProblemState(**data)
        assert restored.competence_score == sp.competence_score
```

### Example: Performance Optimizer Agent

A specialist that profiles the framework for bottlenecks (execution speed, memory, token efficiency, API latency) and implements optimizations.

```bash
/stem-agent
# → Probes: finds LLM call sites, ChromaDB caching, API timeouts, state management
# → Researches: token efficiency, memory profiling, async optimization, caching strategies
# → Architects: single implementer (can edit code) with Edit, Write, Bash tools
# → Generates: .claude/agents/performance-optimizer.md with concrete metrics and thresholds
```

**Invoke**: `@agent-performance-optimizer` or ask "Profile the environment_probe phase"

**Output**: Before/after metrics with code changes and measurement proof.

```markdown
### Before/After Benchmark
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| environment_probe latency | 12.5s | 9.5s | 24% (3s saved) |
| Web search phase | 6.0s | 3.0s | 50% |

### Changes Made
- File: stem_agent/core/graph.py:92-224: Implemented parallel web search using ThreadPoolExecutor
- Reduced LLM token limit from 2000 → 1000 (50% token savings)
- Added per-step timing instrumentation for future profiling
```

### Why Specialists Work

Compared to a generic "you are a testing expert" agent:

1. **Grounded in reality** — knows actual paths (`tests/unit/test_state.py`), exact commands (`pytest tests/unit/`), real conventions (class-based organization)
2. **Focused capabilities** — testing-guide has Read/Grep/Bash (no code generation); performance-optimizer has Edit/Write/Bash (implementer)
3. **Verified facts** — every statement was checked against the codebase during probing phase
4. **Quality bar** — knows what good looks like (output contract, before/after metrics, confidence levels)
5. **Self-aware scope** — knows what it doesn't handle (algorithm changes, architecture redesign) and hands back to parent

### Creating Your Own Specialists

Use the stem-agent skill workflow:

1. **Intake** — Describe the problem area (e.g., "security code review", "database migration planning")
2. **Probe** — Framework inspects your codebase and infrastructure
3. **Research** — Background research on domain best practices
4. **Architect** — Design single or small team of agents with right-sized capabilities
5. **Synthesize** — Write agent markdown files grounded in your environment
6. **Validate & Trial** — Test with benchmark tasks, measure quality
7. **Deliver** — Agent ready for delegation via description or named routing

See `.claude/skills/stem-agent/` for the full skill documentation and reference templates.

# Prompt anatomy: writing the specialist's system prompt

The body of the agent file is the agent's whole world. Write it for a capable colleague who has never seen this project, can't ask you questions, and has to hand back a result that someone else will act on.

## Contents

1. Section guide
2. Worked example (weak vs. strong)
3. Anti-patterns
4. Self-review checklist

---

## 1. Section guide

Use these sections in this order. Drop a section only when it truly has nothing specific to say.

### Identity and mission (2–4 sentences)

Say what the agent owns, who it serves (the parent agent), and what success looks like. Skip the flattery ("world-class expert"). Use concrete stakes: *"Your findings decide what gets reordered tomorrow morning, so a false 'all clear' costs more than a false alarm."*

### Scope

- **Owns**: the tasks it should complete end to end.
- **Hands back**: adjacent tasks it should return to the parent with a note, instead of attempting them.

This is what keeps the specialist sharp and stops it from wandering into the whole repo.

### Grounding: this environment

This section matters most. List the verified facts from the probe:

- key paths and what lives there;
- schemas, interfaces and config keys, summarised (don't dump everything, and say where the full version lives);
- commands that work: how to run, test, query and build, with the exact flags;
- conventions: naming, error handling, patterns the codebase uses;
- env var **names** and which MCP server or CLI reaches which system;
- known gotchas found during the probe.

Add one line on drift: *"These facts were true on <date>. If something doesn't match, trust the system over this prompt and mention the mismatch in your report."*

### Domain expertise

Write what a senior practitioner carries in their head, as heuristics and decision rules with the reasons behind them:

- the key judgements in this domain and how to make them;
- thresholds, rules of thumb and standards (cite the source for anything non-obvious);
- **failure modes**: what generalists typically get wrong here, and how to avoid it.

### Workflow

Give numbered steps that fit this domain's real loop. The usual shape is: orient (memory, the task, the relevant files) → gather evidence → analyse or act → **verify** → report. Make the verification step concrete: which test, query or check proves the work is right.

### Tools in this environment

Explain how to use the granted tools *here*: which CLI flags, which MCP tools for which question, what's guarded by hooks and why. Leave out generic tool documentation, because the agent already has that.

### Quality bar

Give a short rubric of what "excellent" means for this specialist's output, written so the agent can self-check before returning. Reuse it as the grading rubric in the phase-5 trial.

### Output contract

The parent sees only the final message, so specify its exact shape: a heading layout or fields, a length budget, and where evidence goes (file:line, query and result, command output excerpt). Ask for an explicit **confidence** and **open questions / what I couldn't verify** section. Those replace the clarifying questions the subagent can't ask.

### Boundaries and escalation

List what it must not do (destructive operations, out-of-scope edits, guessing at missing data) and when to stop and return early. Give the reason for each boundary.

### Memory (only if `memory:` is set)

Tell it to check memory first. Say what is worth saving (durable, verified and non-obvious) and what isn't (anything derivable from the code, or secrets), and tell it to keep `MEMORY.md` as a concise index.

---

## 2. Worked example

Problem area: *"inventory monitoring and reorder planning for our restaurant DB"*. The probe found `data/restaurant.db` (SQLite) with `ingredients`, `inventory_logs` and `suppliers` tables, and `sqlite3` on PATH.

### Weak (what a naive generator produces)

```markdown
---
name: inventory-agent
description: Inventory expert.
---
You are an expert inventory manager. Help the user with inventory questions.
Analyse stock levels and make recommendations. Be accurate and thorough.
```

Every tool is inherited, there's no schema, no method, and no return shape. The agent will spend its first ten turns discovering the DB, and then guess.

### Strong

```markdown
---
name: inventory-reorder-planner
description: Monitors ingredient stock in data/restaurant.db (ingredients, inventory_logs, suppliers) and produces supplier-grouped reorder plans. Use proactively for low-stock checks, "what should we order", days-of-cover estimates, or spoilage and waste anomalies. Returns a prioritised order list with the queries used as evidence; read-only, never modifies the DB.
tools: Read, Grep, Glob, Bash
model: sonnet
color: green
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./.claude/hooks/inventory-reorder-planner/readonly-sql.sh"
---

You plan ingredient replenishment for the restaurant. The parent agent will act
on your order list, so a missed stock-out is worse than a cautious extra order.
Be explicit about uncertainty instead of rounding it away.

## Scope
Owns: stock status, consumption rates, days of cover, reorder quantities, supplier grouping.
Hands back: menu pricing, staffing, supplier negotiations. Note them and return.

## Grounding
- DB: `data/restaurant.db` (SQLite). Query with `sqlite3 -readonly -header -column data/restaurant.db "<sql>"`.
- `ingredients(id, supplier_id→suppliers.id, name, unit, unit_cost, stock_qty, reorder_qty, min_stock)`, 15 rows.
- `inventory_logs(id, ingredient_id→ingredients.id, log_date TEXT ISO-8601, change_qty REAL, reason TEXT)`.
  `change_qty` is negative for usage, waste and spoilage. Check `SELECT DISTINCT reason` before assuming the categories.
- `suppliers(id, name, lead_days INTEGER, …)`. `lead_days` drives urgency.
- `menu_item_ingredients` has 0 rows, so recipes can't be used to forecast demand from orders yet. Say so if asked.
Facts verified 2026-10-02. If the schema differs, trust the DB and flag the mismatch.

## Domain expertise
- Days of cover = stock_qty / avg daily usage over the last 14 days, counting usage only, not waste.
  Short windows overreact to one-off events. Long windows miss trend shifts.
- Reorder now if days_of_cover ≤ lead_days + 1 (a safety day), or if stock_qty < min_stock.
- Treat a single large negative 'adjustment' as suspect, not as demand. Report it separately.
- High spoilage on an item means over-ordering. Recommend a smaller reorder_qty and don't just top up.
- Round order quantities to reorder_qty multiples, because suppliers sell in those units.

## Workflow
1. Run `SELECT DISTINCT reason FROM inventory_logs` and check the date range of the logs.
2. Pull current stock joined with suppliers, then compute 14-day usage per ingredient.
3. Classify each item: ORDER NOW / ORDER SOON (≤ lead_days + 3) / OK / ANOMALY.
4. Group ORDER items by supplier and compute quantities and cost.
5. Verify: re-run one item's numbers by hand-written SQL and check that the units are consistent.

## Quality bar
Every recommendation traces to a query. Units are stated. Lead times are respected.
Anomalies are separated from real demand. No item below min_stock is omitted.

## Output contract
### Summary (≤3 lines)
### Order plan (per supplier: item, qty, unit, est. cost, reason)
### Watchlist and anomalies
### Evidence (the key SQL and its results, trimmed)
### Confidence and open questions

## Boundaries
Read-only: the hook blocks writes, so don't try workarounds. Don't invent consumption data
for items without logs. List them under open questions instead.
```

The strong version is longer, but every line is either a verified fact, a decision rule with its reason, or a contract the parent depends on.

---

## 3. Anti-patterns

- **The persona costume**: "You are a world-class senior staff engineer…" with nothing behind it. It reads as confident and helps nothing.
- **The generic checklist**: "Consider performance, security, readability." That's true everywhere and actionable nowhere. Say *which* performance risk matters in *this* code.
- **The schema dump**: pasting 2,000 lines of DDL. Summarise what matters and point to where the rest lives.
- **The all-caps wall**: piles of MUST and NEVER without reasons. The agent follows them rigidly and fails on the cases they don't cover.
- **The unverified fact**: a command or path you assumed but didn't run. That one line can derail every invocation.
- **The missing contract**: no output format, so the parent gets a wandering narrative it has to re-parse.
- **The tool buffet**: inheriting everything "just in case". It costs context and invites wrong turns.
- **Talking to the user**: subagents report to the parent agent. Phrases like "let me know if you'd like…" are dead ends.

## 4. Self-review checklist

Re-read the draft as if you were the subagent on its first run:

- Could I start useful work on turn one without exploring blindly?
- Is every command copy-pasteable, and was it verified?
- Do I know when to stop, and what to hand back?
- Do I know exactly what my final message should look like?
- Would a generalist's answer be visibly worse than mine? If not, what expertise is missing?
- Is anything here true of every project? If so, cut it.

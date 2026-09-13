# Skill Generation for Coding Agents

Turns a coding agent's raw execution history into compact, reusable **skills** —
procedures the agent can retrieve, adapt, and follow on future tasks — and
improves those skills from execution feedback (test results, tool errors)
without fine-tuning the underlying model.

This is a working implementation of all five phases described in
[`PROJECT_SPEC.md`](PROJECT_SPEC.md), plus an orchestrator and CLI tying them
together. If you want the full design rationale, read that file. If you want
to know exactly what's built, tested, and verified (and how), read
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md). This file is the short
version: how to install it, run the tests, and actually use it.

## What's here

| Phase | Package | What it does |
|---|---|---|
| P1 | `dspy_modules/` | Segments a trace into transferable-knowledge spans and drafts a six-field candidate skill from each one |
| P2 | `skill_library/` | Stores skills as versioned `SKILL.md`/`metadata.json`; decides create/revise/merge/specialize/reject/retire |
| P3 | `evolution/` | Evolves a skill's text from execution feedback via [GEPA](https://github.com/gepa-ai/gepa) — no weight updates |
| P4 | `validation/` | Gates a candidate on in-domain held-out score *and* cross-domain regression + cost before promoting it |
| P5 | `retrieval/` | Retrieves, activates, adapts, and safely injects a relevant skill into a new task |
| — | `orchestrator/` | Ties all five into one loop (`driver.py`) with a CLI (`cli.py`) driving it against real task execution |
| — | `trace_collection/` | Runs the actual agent loop (local sandbox or a real [Terminal-Bench](https://github.com/laude-institute/terminal-bench) container) and records what happened |

Every module accepts its external dependencies (the backend LM, the task
executor) as arguments — nothing is hardwired to one model or one benchmark.

## Install

```bash
pip install -r requirements.txt          # core: anthropic, dspy, gepa
pip install -r requirements-dev.txt      # + pytest, to run the test suite
```

Real Terminal-Bench execution needs a separate environment — see
[Running against real tasks](#running-against-real-tasks) below.

## Run the tests

The full suite runs with no API key, no Docker, and no network access — every
LLM call is stubbed with `dspy.utils.DummyLM`, and GEPA's optimizer runs for
real against fake (but deterministic) task scoring:

```bash
python -m pytest
```

This is the fastest way to confirm your setup is sane and to see the whole
pipeline exercised end to end (`tests/test_orchestrator.py` and
`tests/test_p3_to_p4_integration.py` are good starting points — they chain
several phases together in one test).

## Running for real

Two things gate real execution, and the CLI fails with a clear error message
if either is missing rather than silently doing the wrong thing:

- **`ANTHROPIC_API_KEY`** — needed for `dspy_modules` (skill extraction,
  library decisions, retrieval/activation/adaptation) and for
  `trace_collection`'s agent loop.
- **Docker + the `terminal-bench` package** — needed for real task execution.
  `terminal-bench` requires **Python ≥3.12**, stricter than this repo's core
  dependencies, so install it into its own virtualenv:

  ```bash
  python3.12 -m venv .venv-tbench
  .venv-tbench/bin/pip install -r requirements.txt -r requirements-tbench.txt
  ```

  Then get a Terminal-Bench task set — e.g. clone
  [`laude-institute/terminal-bench`](https://github.com/laude-institute/terminal-bench)
  and point `--tasks-dir` at its `original-tasks/` directory.

### Collect one trace and extract a skill from it

```bash
# Run a task locally (no Terminal-Bench needed) and record the trace
python -m trace_collection.cli collect "Add a /health endpoint to app.py" \
    --workdir ./sandbox --out-dir ./traces

# Extract candidate skills from it
python -m trace_collection.cli extract-skills "./traces/*.json" --out-dir ./skill_drafts
```

### Run a real Terminal-Bench task through the whole live loop

This is P5 (retrieve + inject) → the agent → P1 (extract) → P2 (file the
result), in one command:

```bash
.venv-tbench/bin/python -m orchestrator.cli handle-task \
    --library-dir ./skill_library_data \
    --tasks-dir ./terminal-bench/original-tasks \
    --task-id acl-permissions-inheritance
```

First run: the library is empty, so nothing gets retrieved/injected, and a
successful task produces your first candidate skill. Run it again on a
related task and, once that skill has been promoted to `active` (see below),
you'll see it retrieved and injected.

### Evolve and promote a skill

Once a skill has at least one candidate version, run GEPA over a batch of
tasks it should handle, then gate the winner through validation:

```bash
.venv-tbench/bin/python -m orchestrator.cli evolve \
    --library-dir ./skill_library_data \
    --tasks-dir ./terminal-bench/original-tasks \
    --skill-id <skill-id-from-the-library> \
    --gepa-train task-a,task-b,task-c \
    --gepa-val task-d,task-e \
    --in-domain task-f,task-g \
    --regression task-h,task-i \
    --reflection-model anthropic/claude-opus-5
```

Output is one of four outcomes — `no_improvement`, `lite_gate_rejected`,
`promotion_rejected`, or `promoted` — never a silent failure. Only
`promoted` actually changes what's active in the library; every other
outcome leaves the current active skill untouched and logs why.

## Where things live on disk

```
skill_library_data/
  <skill-id>/
    1/metadata.json      # a version's source of truth (the six required fields + provenance)
    1/SKILL.md            # human-readable rendering of the same version
    2/...                  # a pending candidate revision can coexist with an active version
    pinned                  # marker file; present = exempt from all automated changes
  .archive/<skill-id>/...    # retired skills -- archived, never deleted; restorable

traces/
  <trace-id>.json    # one collected execution trace, secrets redacted before write
```

`SKILL.md` is generated for humans/agents to read; `metadata.json` is what
every tool in this repo actually reads and writes.

## Notes on scope

- **Retrieval quality** depends on the embedding function in
  `skill_library/index.py`. The default is a dependency-free hashing-trick
  vectorizer — good enough to prove the pipeline works, not production
  retrieval quality. Swap in a real embedding provider by passing a
  different `embed_fn`; nothing else changes.
- **The safety gate** in `retrieval/safety_gate.py` catches obviously
  destructive literal commands (`rm -rf /`, fork bombs, curl-pipe-to-shell).
  It is not a sandbox — the actual tool execution still runs inside
  `tool_handlers.py`/`container_tool_handlers.py`'s isolation.
- **Live shadow-traffic mirroring** (running an unpromoted candidate
  silently alongside the active skill on real traffic) isn't built.
  `validation/shadow.py`'s ledger exists and is ready for it.
- Nothing here has been run continuously against a real, ongoing task
  stream — every phase is built and tested against its neighbors (several
  tests run a real `gepa.optimize()` end to end), but the environment this
  was built in had no Docker daemon and no live API key. See
  `IMPLEMENTATION_PLAN.md` for the exact verification status of every piece.

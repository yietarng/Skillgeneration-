# Skill Generation

Turns an agent's execution traces into reusable **Skills** (concise, general
procedures an agent can follow next time), using the real Claude API rather
than a template or heuristic. Skill generation is grounded using
**environment-probing curation**, the approach from
["Grounding Agent Memory: Environment-Probing Curation for Enterprise Agents"](https://arxiv.org/abs/2609.11060)
(arXiv:2609.11060): instead of trusting a single, partial, sometimes
mistake-laden trace at face value, the curator flags claims it's unsure of
and checks them against the real environment with read-only tools before
committing anything.

Just want to run it? See [QUICKSTART.md](QUICKSTART.md).

## Requirements

```
pip install -r requirements.txt   # anthropic>=0.70
export ANTHROPIC_API_KEY=...      # needed for collect / generate-skill / eval
```

`fetch-public-traces` needs outbound access to `github.com` /
`raw.githubusercontent.com` (and a `git` binary on `PATH`) but no API key.

## The pipeline

```
 ┌───────────────┐      ┌──────────────────┐
 │ collect        │      │ fetch-public-    │
 │ (live agent    │  or  │ traces           │
 │  run)          │      │ (public GitHub   │
 │                │      │  trajectories)   │
 └───────┬────────┘      └────────┬─────────┘
         │                        │
         └──────────┬─────────────┘
                     ▼
            trace.json (+ workdir/)
                     │
                     ▼
         ┌─────────────────────────┐
         │  generate-skill          │
         │  propose → probe → commit│
         └────────────┬─────────────┘
                       ▼
         SKILL.md (+ GROUNDING.md), or "skip"

 eval  compares generate-skill's output with probing on vs off,
       on synthetic fixtures designed to contain a stale/wrong claim.
```

1. **Get a trace.** Either run a real task and record what the agent did
   (`collect`), or pull in a real trace someone else already produced
   (`fetch-public-traces`). Either way you end up with a `trace.json` in the
   shape defined by `schema.py`, and — ideally — a `workdir` on disk that
   still reflects the environment the trace ran in.
2. **Curate a skill from it** (`generate-skill`). The curator:
   - **proposes** a draft skill from the trace(s) alone, listing any
     concrete environment facts it isn't fully sure of (`uncertain_claims`);
   - **probes** those claims, if a live `workdir` is available, using
     read-only tools that can only look, never write or execute;
   - **commits** a final skill that corrects/narrows claims the probe
     contradicted or couldn't confirm — or **skips** entirely if probing
     shows the draft would be misleading.
3. **Measure the effect** (`eval`) by running the same curator with probing
   forced on vs off against small fixtures that contain a deliberately
   stale claim, and checking which condition actually catches it.

## File-by-file guide

### `trace_collection/schema.py`
Plain dataclasses defining the on-disk trace format everything else reads
and writes: `ToolCallRecord` (one tool call: name, input, output, error,
duration), `StepRecord` (one model turn: its tool calls plus raw usage/
response), and `Trace` (a whole task attempt: task text, model, `workdir`,
outcome, final text, and the list of steps). `Trace.to_dict()` is what gets
serialized to `trace.json`.

### `trace_collection/tool_handlers.py`
`SandboxedToolRunner` — executes Claude's native `bash` and
`str_replace_based_edit_tool` tool calls confined to a fixed directory, used
while *collecting* a trace (the agent being observed can read, write, and
run commands here; this is not the read-only probing layer).

### `trace_collection/collector.py`
`TraceCollector.run(task)` drives a real agentic loop against the Anthropic
API: sends the task, executes whatever tools Claude calls via
`SandboxedToolRunner`, feeds results back, and records every turn into a
`Trace` until the model stops or `max_turns` is hit. `save_trace()` writes
the result to `<out_dir>/<trace_id>.json`.

### `trace_collection/probe_tools.py`
The read-only, least-privilege tool surface used during *curation* (not
collection): `view_file`, `list_directory`, `search_files`, all implemented
by `ReadOnlyProbeRunner` and hard-confined to a workdir (path-escape and
symlink-escape are both rejected). `search_files` also skips VCS metadata
(`.git`, `.hg`, `.svn`) so a real repo checkout's binary internals don't
crowd out genuine source hits. There is no write or shell-execution tool
here by design — this is what makes probing safe to run automatically after
the fact.

### `trace_collection/skill_generator.py`
The curator itself — `generate_skill(trace_paths, ...)` runs the
propose → probe → commit cycle described above and returns a dict with
`action: "commit"` (skill fields populated) or `action: "skip"`
(`skip_reason` populated, nothing should be written). `write_skill()` takes
that result and writes `<out_dir>/<slug>/SKILL.md`, plus a
`GROUNDING.md` log of what was checked and found when probing happened.
Internally: `_propose` (one structured-output call), `_probe` (a bounded
tool-use loop using `probe_tools.py`, called only when there's a workdir and
uncertain claims to check), `_commit` (one structured-output call that
folds probe findings into a final decision).

### `trace_collection/adapters/swe_agent.py`
Retrieves real, publicly released coding-agent trajectories (from the
[SWE-agent](https://github.com/SWE-agent/SWE-agent) project's own repo,
pinned to a fixed commit) and reshapes them into this project's `Trace`
schema, so `generate_skill` has genuine traces to work with instead of only
synthetic fixtures. `SOURCES` lists each trajectory plus how to materialize
a real `workdir` for it: `"git_commit"` shallow-fetches the exact repo
commit the trajectory ran against; `"from_observations"` is for
self-contained tasks with no external repo, and reconstructs whatever
file(s) the trajectory's own `open`/`create` steps show. `fetch_all()`
downloads, converts, and materializes every source, skipping (and
reporting) any individual source that fails rather than aborting the batch.

### `trace_collection/eval/fixtures.py` and `trace_collection/eval/harness.py`
A minimal evaluation harness in the spirit of the paper's CLBench
comparison. `fixtures.py` defines small synthetic scenarios, each pairing a
tiny real workdir (e.g. a `config.yaml` with the real port) with a trace
whose trajectory asserts something that contradicts it (e.g. a stale
README mentioning a different port) — plus a `check()` that says whether a
generated skill states the correct fact. `harness.py`'s `run_eval()` runs
`generate_skill` on each scenario with probing forced off and on, scoring
pass/fail and counting API calls (a cheap proxy for the paper's per-query
cost metric) in each condition; `format_report()` renders the comparison
table.

### `trace_collection/cli.py`
Wires all of the above into one command:
- `collect <task> [--workdir] [--out-dir] [--model] [--max-turns]`
- `generate-skill <trace(s)> [--out-dir] [--model] [--no-probing] [--max-probe-turns]`
- `eval [--model]`
- `fetch-public-traces [--out-dir] [--no-workdir]`

### `sample_traces/`
Real traces already retrieved via `fetch-public-traces`, checked in as
ready-to-use examples (see below). Each instance's `workdir/` is
regenerated by re-running the fetch command rather than committed — it's a
real `git` checkout, not something that belongs in this repo's history.

## Using the CLI

Run everything as a module from the repo root:

```bash
# 1a. Record a real trace by actually running a task:
python -m trace_collection.cli collect "Add a /health endpoint to app.py" \
    --workdir ./sandbox --out-dir ./traces

# 1b. ...or pull in real public traces instead (no API key needed):
python -m trace_collection.cli fetch-public-traces --out-dir ./sample_traces

# 2. Curate a skill from one or more traces:
python -m trace_collection.cli generate-skill "./sample_traces/*/trace.json" \
    --out-dir ./skills
# add --no-probing to see what the curator would have shipped without grounding

# 3. See the measured effect of probing on synthetic fixtures:
python -m trace_collection.cli eval
```

`generate-skill` prints either `Skill '<name>' written to <path> (<n>
grounding note(s))` or `Curator skipped this skill: <reason>` if probing
showed the draft wasn't trustworthy enough to commit.

## Currently retrieved public trace instances

| Instance | Real environment | Bug the trace fixes |
|---|---|---|
| `marshmallow-code__marshmallow-1867` | Real library, shallow-fetched at its exact base commit | `TimeDelta` serialization truncates instead of rounding |
| `SWE-agent__test-repo-missing-colon` | Tiny dedicated demo repo, pinned to a resolved commit | Missing colon in a function signature |
| `swe-bench-humanevalfix-python-0` | No external repo — reconstructed from the trace itself | Missing `abs()` in a float-comparison function |

Add more by extending `SOURCES` in `trace_collection/adapters/swe_agent.py`.

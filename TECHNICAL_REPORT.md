# Technical Report: Function-Level Reference

This document describes every class, function, and method in the codebase —
signature, parameters, return value, and behavior. It complements the other
three root documents rather than replacing them:

- [`PROJECT_SPEC.md`](PROJECT_SPEC.md) — architecture and design rationale (the *why*).
- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — build order and verification log (the *how it was checked*).
- [`README.md`](README.md) — install and usage instructions (the *how to run it*).
- This file — what each unit of code actually does (the *what*).

Organized by package, in the same P1→P5 + orchestrator order used
throughout the project. Within each file, items are listed in source order.
Line numbers refer to the file as of this writing and will drift as the
code changes; use them as a starting point, not a guarantee.

---

## Table of contents

1. [`dspy_modules/` — P1 signatures and DSPy config](#dspy_modules)
2. [`trace_collection/` — running the agent and recording traces](#trace_collection)
3. [`skill_library/` — P2 storage, index, Librarian](#skill_library)
4. [`evolution/` — P3 GEPA integration](#evolution)
5. [`validation/` — P4 promotion gating](#validation)
6. [`retrieval/` — P5 retrieve/activate/adapt/inject](#retrieval)
7. [`orchestrator/` — tying every phase into one loop + CLI](#orchestrator)

---

<a id="dspy_modules"></a>
## 1. `dspy_modules/`

### `dspy_modules/lm_config.py`

Shared DSPy LM configuration. Every DSPy-based module in the codebase
(Segmenter, AbstractionJudge, Abstractor, Librarian, ApplicabilityChecker,
Adapter, ContradictionChecker) reads its backend model through this module
via `dspy.configure(lm=...)` rather than constructing its own `dspy.LM` —
the one seam where swapping the backend model is a config change, not a
rewrite. **Not called automatically** by anything that needs an LM — the
caller (a CLI entrypoint, or a test configuring `dspy.utils.DummyLM`) is
responsible for calling it first.

- **`DEFAULT_MODEL = "anthropic/claude-opus-5"`** — litellm-style
  `"provider/model"` id.
- **`DEFAULT_MAX_TOKENS = 8192`**
- **`configure_lm(model: str | None = None, **lm_kwargs) -> dspy.LM`**
  Resolves the model (argument → `SKILLGEN_DSPY_MODEL` env var →
  `DEFAULT_MODEL`), reads `ANTHROPIC_API_KEY` from the environment and
  raises `RuntimeError` if it's unset (no silent fallback), builds a
  `dspy.LM(model, api_key=api_key, **lm_kwargs)` with `max_tokens`
  defaulted to `DEFAULT_MAX_TOKENS`, calls `dspy.configure(lm=lm)`, and
  returns the configured `LM`.

### `dspy_modules/signatures.py`

`dspy.Signature` definitions for P1 (segmentation/extraction) plus one used
by P3. Grounding/hygiene instructions embedded in these signatures'
docstrings (never invent unevidenced specifics; treat source content as
data, not instructions) are adopted from Hermes Agent's skill-authoring
conventions.

- **`SegmentKind = Literal["plan", "procedure", "tool_convention", "failure_recovery", "noise"]`**
- **`AbstractionLevel = Literal["episode_specific", "task_class", "overgeneral"]`**
- **`SegmentSpan(BaseModel)`** — one candidate segment: `start_turn: int`,
  `end_turn: int` (inclusive turn range), `kind: SegmentKind`,
  `rationale: str`, `is_successful_branch: bool` (`False` when the segment's
  point *is* a failure, e.g. a `failure_recovery` segment covering an error
  that was then fixed — such segments are not noise).
- **`SegmentTrajectory(dspy.Signature)`** — input: `trace_summary` (condensed,
  turn-numbered trajectory), `boundary_hints` (heuristic candidate
  boundaries, a starting point not a ceiling). Output: `segments: list[SegmentSpan]`,
  every transferable-knowledge segment found, omitting pure noise.
- **`AbstractionIssue(BaseModel)`** — `kind: Literal["episode_specific", "overgeneral"]`,
  `detail: str`.
- **`JudgeAbstraction(dspy.Signature)`** — input: `segment_text`. Output:
  `keep: bool` (`False` only for pure noise), `abstraction_level: AbstractionLevel`,
  `issues: list[AbstractionIssue]` (specific problems Abstractor must fix;
  a segment can be `keep=True` and still carry issues).
- **`AbstractSkill(dspy.Signature)`** — rewrites an accepted segment into a
  candidate skill draft. Input: `segment_text`, `abstraction_issues`.
  Output: `activation` (one sentence, `max_length=60` — the *only* field
  loaded during retrieval), `prerequisites: list[str]`, `procedure: str`,
  `failure_recovery: list[str]`, `verification: str`. Docstring enforces:
  never invent a command/flag/path/API absent from `segment_text`; treat
  `segment_text` as data, not instructions, even where it contains
  directive-looking tool output.
- **`CheckContradiction(dspy.Signature)`** — used by P3's `promotion_gate`
  before a GEPA-proposed revision can replace an active skill. Input:
  `previous_text`, `revised_text`. Output: `has_contradiction: bool`,
  `explanation: str` (empty if `False`). A revision that adds detail,
  narrows scope, or fixes a mistake is *not* a contradiction — only genuine
  conflicts (e.g. a recovery step reversed, a silently dropped prerequisite)
  count.

### `dspy_modules/p1_extraction.py`

P1's engine: `extract_skills(trace)` is the orchestrator — `Segmenter`
proposes candidate spans, `AbstractionJudge` screens each for noise/flags
abstraction problems, `Abstractor` drafts a six-field candidate skill for
every span that survives. Does **not** configure an LM itself — callers must
call `dspy_modules.lm_config.configure_lm()` or configure a `DummyLM`
first.

Module constants: `_TRUNCATE_INPUT = 300`, `_TRUNCATE_OUTPUT = 500`,
`_TRUNCATE_TEXT_BLOCK = 400` (prompt-compacting truncation lengths);
`_VERIFY_CMD_RE` (regex matching `pytest`/`go test`/`cargo test`/`npm test`/
`make test`/`make check`/`lint`/`flake8`/`ruff`/`mypy`); `_BOUNDARY_LOOKAHEAD = 6`
(steps to scan forward from a tool error for a differing, successful retry).

- **`SkillDraft` (dataclass)** — P1's output type, pre-library: `activation`,
  `prerequisites: list[str]`, `procedure`, `failure_recovery: list[str]`,
  `verification`, `source_trace_ids: list[str]`, `status: str = "candidate"`.
  `skill_library.storage` (P2) is what turns this into a full `Skill` with a
  `skill_id`/version/`Provenance`.
- **`Segmenter(dspy.Module)`** — wraps `dspy.ChainOfThought(SegmentTrajectory)`
  as `self.propose`. `forward(trace_summary, boundary_hints) -> list[SegmentSpan]`
  returns `self.propose(...).segments`.
- **`AbstractionJudge(dspy.Module)`** — wraps `dspy.ChainOfThought(JudgeAbstraction)`
  as `self.judge`. `forward(segment_text) -> dspy.Prediction` returns
  `self.judge(segment_text=segment_text)` directly (the caller reads
  `.keep`/`.abstraction_level`/`.issues` off the prediction).
- **`Abstractor(dspy.Module)`** — wraps `dspy.ChainOfThought(AbstractSkill)`
  as `self.draft` and an `AbstractionJudge()` as `self.recheck`.
  `forward(segment_text, abstraction_issues) -> dspy.Prediction`: drafts once,
  then re-runs `self.recheck` against the drafted procedure/prerequisites; if
  the recheck still flags any `episode_specific` issue, re-drafts **once**
  with the accumulated issues appended (a single bounded retry, not an
  open-ended loop).
- **`_step_texts(step: StepRecord) -> list[str]`** — pulls any `text`/`thinking`
  blocks out of a step's raw API `response["content"]`.
- **`summarize_trace(trace: Trace) -> str`** — condensed, turn-numbered
  trajectory (`task`, `outcome`, each tool call truncated to
  `_TRUNCATE_INPUT`/`_TRUNCATE_OUTPUT` chars, plus `final_text` truncated to
  800 chars) fed to `Segmenter` as `trace_summary`. Deliberately drops full
  API payloads and thinking text to keep the prompt compact.
- **`render_segment(trace, start_turn, end_turn) -> str`** — renders one turn
  range (task line, text/thinking blocks, tool calls with truncated
  input/output) as text, for `JudgeAbstraction`/`Abstractor` input.
- **`detect_boundary_hints(trace: Trace) -> str`** — heuristically detects two
  boundary types: (1) a failed tool call followed within `_BOUNDARY_LOOKAHEAD`
  steps by a differing, successful retry of the *same* tool name (candidate
  `failure_recovery` boundary), and (2) any command matching `_VERIFY_CMD_RE`
  (candidate verification/termination point). Returns a placeholder string
  if nothing is found; this is a seed for `Segmenter`'s search, not a
  detector for plan/sub-goal boundaries.
- **`extract_skills(trace: Trace) -> list[SkillDraft]`** — runs
  `Segmenter → AbstractionJudge → Abstractor` over one trace. For each span
  Segmenter proposes: if `span.kind == "noise"`, appends a `Segment` to
  `trace.segments` and skips the (wasted) `AbstractionJudge` call entirely
  (a real bug found and fixed during P1's build — noise segments used to
  still reach the judge). Otherwise renders the segment, judges it, always
  appends a `Segment` record (kept or not), and — only if `verdict.keep` —
  runs `Abstractor` and appends a `SkillDraft` to the result. Mutates
  `trace.segments` as a side effect; returns zero or more `SkillDraft`s.

---

<a id="trace_collection"></a>
## 2. `trace_collection/`

### `trace_collection/schema.py`

Data structures for one collected agent execution trace.

- **`new_trace_id() -> str`** — `f"trace_{uuid.uuid4().hex[:12]}"`.
- **`ToolCallRecord` (dataclass)** — `tool_use_id`, `name`, `input: dict`,
  `output: str`, `is_error: bool`, `duration_ms: float`.
- **`StepRecord` (dataclass)** — `turn: int`, `timestamp`, `stop_reason: str | None`,
  `usage: dict`, `response: dict` (full serialized API response for the
  turn), `tool_calls: list[ToolCallRecord]`.
- **`Segment` (dataclass)** — a P1-identified span: `trace_id`,
  `turn_range: tuple[int, int]`, `kind: str`, `abstraction_level: str | None`,
  `rationale: str = ""`, `is_successful_branch: bool = True`. Filled in by
  `dspy_modules.p1_extraction`, not at collection time.
- **`Trace` (dataclass)** — the top-level record: `trace_id`, `task`, `model`,
  `workdir`, `started_at`, `ended_at: str | None`, `steps: list[StepRecord]`,
  `outcome: str = "in_progress"` (`success`/`max_turns`/`error`/`refusal`/...),
  `final_text: str | None`, `total_usage: dict[str, int]` (defaults to zeroed
  `input_tokens`/`output_tokens`/`cache_creation_input_tokens`/
  `cache_read_input_tokens`), `outcome_signal: dict | None` (`{"kind", "score", "detail"}`),
  `repo_context: dict | None` (`{"repo", "language", "task_type"}`),
  `segments: list[Segment]`. `to_dict() -> dict` via `dataclasses.asdict`.
- **`trace_from_dict(data: dict) -> Trace`** — reconstructs a full `Trace`
  (with nested `StepRecord`/`ToolCallRecord`/`Segment`) from a plain dict,
  e.g. `json.loads` of a saved trace file.

### `trace_collection/collector.py`

Runs a manual agentic loop (not the SDK's tool runner) against the real
Anthropic API so every raw request/response, thinking summary, tool call,
and token-usage figure is captured.

Constants: `DEFAULT_MODEL = "claude-opus-5"`, `DEFAULT_MAX_TURNS = 30`,
`DEFAULT_MAX_TOKENS = 16000`, `SYSTEM_PROMPT` (base instructions for an
autonomous sandboxed coding agent, told to prefix its final reply with
`"DONE:"`).

- **`_now() -> str`** — current UTC time, ISO format.
- **`TraceCollector`**
  - `__init__(workdir=None, model=DEFAULT_MODEL, max_turns=DEFAULT_MAX_TURNS, max_tokens=DEFAULT_MAX_TOKENS, client=None, tool_runner=None, tool_defs=None, system_prompt_addendum=None)`
    — raises `ValueError` if both `tool_runner` and `workdir` are `None`.
    Defaults `self.tools` to `SandboxedToolRunner(workdir)` if no
    `tool_runner` given (added during Terminal-Bench integration so a
    different runner — e.g. `ContainerToolRunner` — can be injected without
    forking this class); defaults `self.tool_defs` to
    `[BASH_TOOL, TEXT_EDITOR_TOOL]`; appends `system_prompt_addendum` to
    `SYSTEM_PROMPT` if given (this is P3/P5's only injection point into the
    agent loop — the loop itself is otherwise unaware skills exist).
  - `run(task: str) -> Trace` — the agentic loop. Builds a fresh `Trace`,
    seeds `messages` with the task as a user turn, then for up to
    `max_turns`: calls `client.messages.create(...)` with `thinking={"type": "adaptive", "display": "summarized"}`;
    on `anthropic.APIStatusError`/`APIConnectionError` sets `outcome="error"`
    and breaks; records a `StepRecord`, accumulates usage; on
    `stop_reason == "refusal"` records the step, sets `outcome="refusal"`,
    breaks; on `"pause_turn"` (server-side tool loop paused mid-turn) appends
    the step and continues without sending tool results; if the response has
    no `tool_use` blocks, records `final_text` and sets
    `outcome = "success" if stop_reason == "end_turn" else stop_reason`,
    breaks; otherwise executes every tool-use block via `self.tools.run(name, input)`,
    timing each call, appends `ToolCallRecord`s and sends `tool_result`
    blocks back as the next user turn. If the loop exhausts `max_turns`
    without breaking, sets `outcome = "max_turns"` (via the `for...else`
    clause). Always sets `trace.ended_at` before returning.
  - `_accumulate_usage(trace, usage)` (staticmethod) — adds each of
    `trace.total_usage`'s four keys from the turn's `usage` dict (defaulting
    missing/`None` values to 0).
- **`save_trace(trace: Trace, out_dir: str) -> Path`** — writes
  `<out_dir>/<trace_id>.json`, running the trace's `to_dict()` through
  `trace_collection.redact.redact_value` first. Returns the written path.

### `trace_collection/redact.py`

Secret redaction applied before any trace or skill artifact touches disk.
Best-effort, not a guarantee.

- **`_REDACTED = "[REDACTED]"`**
- **`_PATTERNS`** — compiled regexes for: PEM private key blocks, AWS access
  key IDs (`AKIA...`), Anthropic API keys (`sk-ant-...`), OpenAI-style keys
  (`sk-...`), GitHub tokens (`gh[pousr]_...`), `Bearer <token>` headers, and
  a generic `key/secret/password/token: <value>`-shaped pattern.
- **`redact_text(text: str) -> str`** — runs every pattern's `.sub(_REDACTED, ...)`
  over `text` in sequence; returns `text` unchanged if falsy.
- **`redact_value(value: Any) -> Any`** — recursively applies `redact_text`
  to every string leaf of a JSON-shaped value (dict/list/tuple/str/other),
  used on a trace's or skill's `to_dict()` output right before writing.

### `trace_collection/tool_handlers.py`

Client-side execution of Claude's Anthropic-defined `bash` and text-editor
tools, confined to a fixed local sandbox directory. Both tool defs are
schema-less (`{"type": ..., "name": ...}` only).

`BASH_TOOL = {"type": "bash_20250124", "name": "bash"}`,
`TEXT_EDITOR_TOOL = {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}`,
`DEFAULT_BASH_TIMEOUT_S = 60`.

- **`ToolExecutionError(Exception)`** — raised for tool-input problems;
  caught internally and returned as an `is_error` result rather than
  propagating.
- **`SandboxedToolRunner`**
  - `__init__(workdir: str, bash_timeout=DEFAULT_BASH_TIMEOUT_S)` — resolves
    and creates `workdir`.
  - `run(name, tool_input) -> tuple[str, bool]` — dispatches to
    `_run_bash`/`_run_text_editor` by `name`; returns `(f"Unknown tool: {name}", True)`
    for anything else. Catches `ToolExecutionError`, `subprocess.TimeoutExpired`,
    and any other `Exception` (to keep the agentic loop alive on unexpected
    failures), always returning `(output_text, is_error)`.
  - `_run_bash(tool_input) -> str` — `{"restart": true}` short-circuits to
    `"bash session reset"` (no persistent session actually exists in this
    per-call subprocess model — see `ContainerToolRunner` for the version
    with real persistent state). Otherwise requires `"command"`, runs it via
    `subprocess.run(..., shell=True, cwd=self.workdir, timeout=self.bash_timeout)`,
    concatenates stdout+stderr, appends `"[exit code N]"` on nonzero return,
    and returns `"(no output)"` if the result is empty.
  - `_resolve(raw_path) -> Path` — resolves `raw_path` against `workdir` and
    raises `ToolExecutionError` if the resolved path escapes `workdir` (path
    traversal guard).
  - `_run_text_editor(tool_input) -> str` — implements `view` (lists a
    directory, or returns line-numbered file content honoring an optional
    `view_range`), `create` (backs up an existing file to `.bak` first, then
    writes `file_text`), `str_replace` (requires the `old_str` to match
    *exactly once*, else raises), `insert` (splices `insert_text` after
    `insert_line`). Raises `ToolExecutionError` for an unknown command.

### `trace_collection/container_tool_handlers.py`

Same interface as `tool_handlers.py`, backed by a live Terminal-Bench
container's `TmuxSession` instead of a local sandbox directory. `TmuxSession`
is imported only under `TYPE_CHECKING` so this module stays importable
without the `terminal_bench` package (which requires Python ≥3.12). **Not
exercised against a live container** in the environment this was built in
(no Docker daemon available) — flagged as unverified in
`IMPLEMENTATION_PLAN.md`.

Same `BASH_TOOL`/`TEXT_EDITOR_TOOL` constants as `tool_handlers.py`.
`DEFAULT_BASH_TIMEOUT_S = 60.0`, `DEFAULT_CWD = "/root"`.

- **`ToolExecutionError(Exception)`** — same role as in `tool_handlers.py`.
- **`ContainerToolRunner`**
  - `__init__(session: TmuxSession, bash_timeout=DEFAULT_BASH_TIMEOUT_S, default_cwd=DEFAULT_CWD)`.
  - `run(name, tool_input) -> tuple[str, bool]` — same dispatch/exception
    shape as `SandboxedToolRunner.run`, but catches `TimeoutError` (tmux's
    timeout signal) instead of `subprocess.TimeoutExpired`.
  - `_run_bash(tool_input) -> str` — `{"restart": true}` calls
    `session.clear_history()` (the nearest equivalent to a restart, since
    the tmux session itself persists for the container's lifetime — unlike
    `SandboxedToolRunner`, this *does* give genuinely persistent shell state
    across calls). Otherwise sends `f'{command}; echo "[exit code $?]"'` via
    `session.send_keys(..., block=True, max_timeout_sec=self.bash_timeout)`
    and reads back `session.get_incremental_output()` — the `[exit code N]`
    marker is manually appended via shell since tmux gives screen output,
    not a return code.
  - `_resolve(raw_path) -> str` — absolute paths pass through; relative paths
    are joined onto `self.default_cwd`. Raises if `raw_path` is empty.
  - `_exec(cmd: list[str])` — thin wrapper over `session.container.exec_run(cmd)`.
  - `_run_text_editor(tool_input) -> str` — same four commands as
    `SandboxedToolRunner._run_text_editor`, but implemented via direct
    container exec (`test -d`/`test -f`/`ls -A`/`cat`) instead of Python
    filesystem calls — bypasses tmux entirely since these are structured
    file I/O, not something that needs to appear in the recorded terminal
    transcript.
  - `_read_file(path) -> str` — `cat`s the file via `_exec`, raises
    `ToolExecutionError` on nonzero exit.
  - `_write_file(path, content)` — writes `content` to a local temp file,
    then uses `session.copy_to_container(...)` (docker copy-in) to place it
    — deliberately avoids a shell heredoc, which would need escaping
    arbitrary file content.

### `trace_collection/tbench_adapter.py`

Given a Terminal-Bench task id, runs `TraceCollector` against its container
and fills in the trace's `outcome_signal` from the task's own test suite.
Reuses `terminal_bench`'s own `Task`/`TrialHandler`/`Terminal`/parser
machinery for everything except the agent step itself. Requires the
`terminal-bench` package and a running Docker daemon — **not run against a
live container** in the environment this was built in; flagged as
unverified end-to-end (though the isolated Python 3.12 venv verification
covered the CLI-facing functions that call into this module).

- **`run_tbench_task(tasks_dir, task_id, model=DEFAULT_MODEL, max_turns=DEFAULT_MAX_TURNS, no_rebuild=False, system_prompt_addendum=None) -> Trace`**
  Builds a `TrialHandler` for the task, spins up its container via
  `spin_up_terminal(...)` (disables asciinema recording — `TraceCollector`
  is the record of record), creates an `"agent"` tmux session, wraps it in a
  `ContainerToolRunner`, constructs a `TraceCollector` with that runner and
  `system_prompt_addendum` forwarded unchanged (this parameter is what lets
  this function double as an `orchestrator.driver.CollectTraceFn`,
  `evolution.adapter.RunTaskFn`, or a P4 `run_task_fn` — retrieval's
  injected guidance and a GEPA/validation candidate's rendered text both
  reach the real agent loop through this one parameter), and calls
  `collector.run(task.instruction)`. Sets `trace.repo_context` from the
  task's id/category. Copies the task's real `run-tests.sh` + `tests/` into
  the container and runs it (matching exactly what Terminal-Bench's own
  harness does), on a separate test session unless
  `task.run_tests_in_same_shell`. On a test-run timeout or a parser
  exception, sets a `0.0`-score `outcome_signal` with the failure detail and
  returns early. Otherwise parses the post-test pane with
  `trial_handler.parser.parse(...)`, computes
  `passed / total` across `UnitTestStatus.PASSED` results, and sets
  `trace.outcome_signal = {"kind": "test", "score": ..., "detail": ...}`.
- **`main(argv=None) -> int`** — CLI: `--tasks-dir`, `--task-id`, `--out-dir`
  (default `./traces`), `--model`, `--max-turns`, `--no-rebuild`. Runs the
  task, saves the trace via `save_trace`, prints outcome/score/turn count.

### `trace_collection/skill_generator.py`

The original, naive single-LLM-call skill distiller — kept as a fallback
alongside P1's real segmentation pipeline, not removed. Calls the Claude API
once per invocation with structured JSON-schema output; no segmentation, no
abstraction judging, no multi-skill extraction from one trace.

`DEFAULT_MODEL = "claude-opus-5"`, `SKILL_GEN_SYSTEM` (system prompt: distill
traces into one general, reusable Skill), `SKILL_OUTPUT_SCHEMA` (JSON schema
for `skill_name`/`description`/`prerequisites`/`steps`/`pitfalls`/`markdown`).

- **`_summarize_trace_for_prompt(trace: dict) -> str`** — reduces a raw
  trace dict to task/outcome/tool-calls-with-truncated-input-output/final
  text, dropping full API payloads and thinking text.
- **`generate_skill(trace_paths, model=DEFAULT_MODEL, client=None) -> dict`**
  — loads each trace JSON file, summarizes it, and calls
  `client.messages.create(..., output_config={"effort": "high", "format": {"type": "json_schema", "schema": SKILL_OUTPUT_SCHEMA}})`
  with all summaries concatenated in one user message asking for one skill
  generalizing across all the given traces. Parses and returns the JSON
  text block from the response.
- **`write_skill(skill: dict, out_dir: str) -> Path`** — slugifies
  `skill["skill_name"]`, writes `skill["markdown"]` to
  `<out_dir>/<slug>/SKILL.md`, returns the path.

### `trace_collection/cli.py`

CLI entrypoint: collect real execution traces and generate/extract skills
from them.

- **`main(argv=None) -> int`** — three subcommands via `argparse`:
  - `collect <task> [--workdir] [--out-dir] [--model] [--max-turns]` —
    constructs a `TraceCollector`, runs the task, saves the trace, prints
    outcome/turns/token usage.
  - `generate-skill <traces...> [--out-dir] [--model]` — globs each trace
    argument (falling back to the literal string if the glob matches
    nothing), calls `skill_generator.generate_skill`, writes the result via
    `write_skill`.
  - `extract-skills <traces...> [--out-dir] [--model]` — imports
    `dspy_modules` lazily (so the other two subcommands don't need `dspy`
    installed), calls `configure_lm(model=args.model)`, loads each trace via
    `trace_from_dict`, runs `extract_skills(trace)`, writes each resulting
    `SkillDraft` as its own JSON file under `<out_dir>/<trace_id>-<i>.json`.

---

<a id="skill_library"></a>
## 3. `skill_library/`

### `skill_library/storage.py`

P2's schema (`Skill`/`Provenance`/`EvaluationRecord`) and the `SkillLibrary`
class owning all reading/writing/archiving of skill versions. `metadata.json`
is the source of truth for every stored version; `SKILL.md` is a rendering
generated from it and never parsed back.

On-disk layout:
```
<root>/<skill_id>/<version>/metadata.json   # source of truth
<root>/<skill_id>/<version>/SKILL.md        # rendered, for humans/agents
<root>/<skill_id>/pinned                    # marker file (skill-wide, not per-version)
<root>/.archive/<skill_id>/...              # archived skills, same layout
```

`_SLUG_RE`, `_MAX_SLUG_LEN = 40`.

- **`SkillLibraryError(Exception)`** — the one error type raised for every
  invalid-state condition in this module (unknown skill_id, pinned skill,
  missing version, etc).
- **`_now() -> str`** — current UTC ISO timestamp.
- **`_slugify(text, max_len=_MAX_SLUG_LEN) -> str`** — lowercase, non-alnum
  runs collapsed to `-`, trimmed, truncated, falls back to `"skill"` if
  empty.
- **`EvaluationRecord` (dataclass)** — `skill_id`, `skill_version`,
  `suite: str` (`"in_domain_heldout"` | `"system_regression"`), `task_id`,
  `score: float`, `cost: dict`, `feedback_text: str`.
- **`Provenance` (dataclass)** — `source_trace_ids: list[str]`, `created_at`,
  `revised_from: str | None`, `validation_evidence: list[EvaluationRecord]`,
  `limitations: str = ""`.
- **`Skill` (dataclass)** — `skill_id`, `version: int`, `name`, `activation`,
  `prerequisites: list[str]`, `procedure`, `failure_recovery: list[str]`,
  `verification`, `provenance: Provenance`, `related_skills: list[str]`,
  `pinned: bool = False`, `status: str = "candidate"`
  (`candidate`/`active`/`deprecated`/`archived`). `to_dict()` via
  `dataclasses.asdict`.
- **`skill_from_dict(data: dict) -> Skill`** — inverse of `to_dict`,
  reconstructing nested `Provenance`/`EvaluationRecord`s.
- **`skill_to_markdown(skill: Skill) -> str`** — renders the
  `PROJECT_SPEC.md`-§12 template: YAML-ish frontmatter (`skill_id`,
  `version`, `status`, `pinned`, `related_skills`) followed by
  Activation/Prerequisites/Procedure/Failure Recovery/Verification &
  Termination/Provenance sections.
- **`SkillLibrary`**
  - `__init__(root: str | Path)` — creates `root` and `root/.archive`.
  - `_skill_dir(skill_id, archived=False) -> Path`, `_version_dir(skill_id, version, archived=False) -> Path`
    — path helpers.
  - `list_versions(skill_id) -> list[int]` — sorted integer version dirs
    under the skill's active-tree directory (`[]` if the skill doesn't
    exist).
  - `exists(skill_id) -> bool` — `bool(list_versions(skill_id))`.
  - `_resolve_version(skill_id) -> int` — scans versions for one with
    `status == "active"`; falls back to the latest (highest-numbered)
    version if none is active. Raises `SkillLibraryError` if the skill
    doesn't exist.
  - `read(skill_id, version=None) -> Skill` — resolves `version` via
    `_resolve_version` if not given, loads `metadata.json`, sets
    `.pinned` from `is_pinned(skill_id)` (pin state lives in a separate
    marker file, not in each version's metadata).
  - `all_skill_ids(include_archived=False) -> list[str]` — sorted skill_ids
    from `root` (and `archive_root` if `include_archived`), excluding the
    `.archive` directory name itself.
  - `_write(skill: Skill) -> Skill` — writes `metadata.json` (redacted via
    `redact_value`) and `SKILL.md` for `skill.version`; creates the version
    directory if needed.
  - `save(skill: Skill) -> Skill` — public entrypoint for overwriting an
    *already-created* version's content in place (e.g. attaching rejection
    evidence) without bumping the version or touching status. Raises if the
    skill_id doesn't exist yet.
  - `_unique_slug(text) -> str` — `_slugify(text)`, appending `-2`, `-3`, ...
    until a skill_id that doesn't already exist is found.
  - `create(draft: SkillDraft, name=None, skill_id=None) -> Skill` — assigns
    a fresh `skill_id` (from `draft.activation` if not given), raises if it
    already exists, builds version-1 `Skill` with `status="candidate"`,
    writes it.
  - `revise(skill_id, draft: SkillDraft, limitations="") -> Skill` — raises
    if pinned or unknown; reads current, bumps to `versions[-1] + 1`,
    stamps `Provenance.revised_from = f"{skill_id}@{current.version}"`,
    preserves `related_skills`, writes as a new `candidate` version.
  - `specialize(general_skill_id, draft, name=None) -> Skill` — forks a
    brand-new skill_id via `create`, adds `general_skill_id` to
    `related_skills`, re-writes. Does not touch `general_skill_id` itself.
  - `merge(primary_skill_id, secondary_skill_id, draft) -> Skill` — revises
    the primary, cross-references the secondary in `related_skills`, notes
    the merge in `limitations`, writes, then archives the secondary.
  - `_set_status(skill_id, version, status)` — low-level metadata status
    write.
  - `active_version(skill_id) -> int | None` — the currently `"active"`
    version, or `None` if the skill has never been promoted.
  - `promote(skill_id, version) -> Skill` — flips `version` to `"active"`;
    demotes any other currently-active version to `"deprecated"` (kept on
    disk, not archived, so `rollback` can find it). This is the actual
    storage write P3's `promotion_gate.promote()` never performs itself —
    only called by `validation/promotion.py` after both gates pass.
  - `rollback(skill_id) -> Skill` — reverts to the most recent `"deprecated"`
    version (`max()` of matching version numbers); demotes the current
    active version to `"deprecated"` in turn (so rolling back is itself
    reversible). Raises `SkillLibraryError` if there's no deprecated version.
  - `_pin_marker(skill_id) -> Path`, `is_pinned(skill_id) -> bool`,
    `pin(skill_id)` (raises if unknown; touches the marker file),
    `unpin(skill_id)` (removes the marker if present, no error if absent).
  - `archive(skill_id)` — raises if pinned; sets every version's status to
    `"archived"`; moves the whole skill directory into `.archive` (replacing
    any existing archive entry for the same id).
  - `restore(skill_id)` — raises if not archived or if an active-tree entry
    already exists; moves the directory back; resets every version's status
    from `"archived"` to `"candidate"` (restored skills re-enter the P3/P4
    promotion gate rather than coming back pre-promoted).

### `skill_library/index.py`

Nearest-neighbor lookup over the library's activation+prerequisites text.
Used by `Librarian` for dedup (P2) and later reused unmodified as P5's
retrieval index — same artifact, no rework needed. Default embedding is a
dependency-free hashing-trick bag-of-words vectorizer — proven to find real
near-duplicates, explicitly **not** production retrieval quality; swap
`embed_fn` for a real embedding provider with no other code changes.

`EmbedFn = Callable[[list[str]], list[list[float]]]`, `_TOKEN_RE`,
`DEFAULT_DIMS = 256`.

- **`_tokenize(text) -> list[str]`** — lowercase alnum-token extraction.
- **`hashing_embed(texts, dims=DEFAULT_DIMS) -> list[list[float]]`** —
  deterministic (MD5-based, stable across processes unlike builtin `hash()`)
  bag-of-words hashing-trick vectorizer, L2-normalized so cosine similarity
  reduces to a plain dot product.
- **`_dot(a, b) -> float`** — plain dot product.
- **`EmbeddingIndex`** — in-memory, one vector per `skill_id` (upsert
  overwrites).
  - `__init__(embed_fn=hashing_embed)`.
  - `upsert(skill_id, text, metadata=None)` — embeds and stores.
  - `remove(skill_id)` — drops the vector and metadata.
  - `__contains__(skill_id) -> bool`, `__len__() -> int`.
  - `query(text, k=5, filter_fn=None) -> list[tuple[str, float]]` — embeds
    `text`, scores every stored vector by dot product (optionally filtered
    by `filter_fn(metadata)`), returns the top `k` `(skill_id, similarity)`
    pairs sorted descending.

### `skill_library/librarian.py`

`Librarian(dspy.Module)` decides CREATE/REVISE/MERGE/SPECIALIZE/REJECT for
a candidate draft against the library's nearest existing skills;
`apply_decision()` executes that decision against `SkillLibrary`. RETIRE is
a separate, non-LLM path (`sweep_retirements`). Scope note: this
`Librarian` only ever compares one incoming candidate against its nearest
existing skills (one-to-many) — MERGE here means "fold this candidate into
an existing skill," mechanically identical to REVISE; a library-internal
many-to-many consolidation sweep is a distinct, unbuilt operation.

`LibrarianAction = Literal["CREATE", "REVISE", "MERGE", "SPECIALIZE", "REJECT"]`.

- **`LibrarianDecide(dspy.Signature)`** — input: `candidate_skill` (rendered
  draft text), `nearest_existing` (rendered nearest skills, or `"(none)"`).
  Output: `action: LibrarianAction`, `target_skill_id: str` (the existing
  skill_id for REVISE/MERGE/SPECIALIZE; empty for CREATE/REJECT),
  `rationale: str`. Docstring spells out the criteria for each action in
  detail (CREATE: nothing covers this yet; REVISE: same underlying skill
  refined; MERGE: overlapping enough to fold in; SPECIALIZE: a general
  skill's procedure fails in a repo/language-specific way; REJECT:
  duplicate, router-only, or non-generalizable).
- **`LibrarianDecision` (dataclass)** — `action`, `target_skill_id: str | None`,
  `rationale`.
- **`_render_draft(draft: SkillDraft) -> str`**, **`_render_skill(skill: Skill) -> str`**
  — text renderings fed into the LM prompt.
- **`Librarian(dspy.Module)`**
  - `__init__(library, index, top_k=5)` — wraps
    `dspy.ChainOfThought(LibrarianDecide)` as `self.decide`.
  - `forward(draft, repo_context=None) -> LibrarianDecision` — builds a
    query from `draft.activation + prerequisites`, optionally filters the
    index query by `task_type` (skills with no `task_type` metadata always
    pass the filter), fetches `top_k` neighbors, reads each via
    `library.read` (skipping any whose index entry outlived the skill, e.g.
    archived), renders them, and calls `self.decide(...)`.
- **`apply_decision(library, index, draft, decision, repo_context=None) -> Skill | None`**
  — executes the decision: REJECT → `None`, nothing written; CREATE →
  `library.create`; REVISE/MERGE → `library.revise(decision.target_skill_id, draft)`
  (both require `target_skill_id`, raising `ValueError` if missing);
  SPECIALIZE → `library.specialize`. On any non-REJECT outcome, upserts the
  resulting skill into `index` with `{"task_type": repo_context.get("task_type")}`
  metadata. Raises `ValueError` for an unrecognized action.
- **`SkillMetrics` (dataclass)** — `utilization: int`, `success_rate: float`.
- **`MetricsFn = Callable[[str], SkillMetrics | None]`**
- **`_no_metrics(skill_id) -> None`** — the default `metrics_fn`; always
  returns `None` (library-health tracking doesn't exist yet — this is P4/P5
  territory — so by default `sweep_retirements` sweeps nothing).
- **`sweep_retirements(library, index, metrics_fn=_no_metrics, min_utilization=5, min_success_rate=0.4) -> list[str]`**
  — for every non-pinned skill_id, if `metrics_fn` returns a result with
  `utilization >= min_utilization` and `success_rate < min_success_rate`,
  archives the skill and removes it from the index. Returns the list of
  retired skill_ids.

---

<a id="evolution"></a>
## 4. `evolution/`

### `evolution/adapter.py`

`SkillGEPAAdapter` — the integration point between this system and GEPA's
optimization engine, duck-typed against `gepa.core.adapter.GEPAAdapter`'s
`evaluate()`/`make_reflective_dataset()` contract (no base class required).

`Candidate = dict[str, str]`,
`RunTaskFn = Callable[[str, dict[str, Any]], Trace]` — raises only for
*systemic* failures; a single task's own failure should come back as a
low-scoring `Trace`, not an exception.

- **`skill_to_candidate(skill: Skill) -> Candidate`** — the three
  GEPA-optimizable components (spec §5.3): `procedure` (as-is),
  `prerequisites`/`failure_recovery` (newline-joined). `activation`,
  `verification`, and provenance are deliberately absent — not optimized
  text.
- **`candidate_to_skill_fields(candidate: Candidate) -> dict[str, Any]`** —
  inverse: splits `prerequisites`/`failure_recovery` back into
  non-blank-line lists.
- **`SkillTrajectory` (dataclass)** — `task_id`, `trace: Trace | None` (`None`
  if `run_task_fn` raised and `evaluate()` caught it), `score: float`,
  `feedback_text: str`.
- **`SkillGEPAAdapter`**
  - `propose_new_texts = None` (class attribute) — **required**: GEPA's
    reflective-mutation proposer accesses `adapter.propose_new_texts`
    directly (not via `getattr` with a default) to decide between a custom
    proposer and the default `reflection_lm`-driven one. Omitting this
    attribute entirely crashes every iteration with `AttributeError` — a
    real bug found and fixed during P3's build. `None` means "use the
    default reflection_lm-driven proposal."
  - `__init__(base_skill: Skill, run_task_fn: RunTaskFn)`.
  - `evaluate(batch: list[str], candidate: Candidate, capture_traces=False) -> EvaluationBatch`
    — decodes `candidate` back to skill fields once, then for each `task_id`
    in `batch`: runs `run_task_fn(task_id, skill_fields)`, scores via
    `score_and_feedback`; catches any `Exception` and treats it as a
    single-example failure (`score=0.0`, trace=`None`, an explanatory
    feedback string) rather than letting it abort the whole batch — matching
    GEPAAdapter's documented contract. Returns `EvaluationBatch(outputs, scores, trajectories)`
    with `trajectories=None` unless `capture_traces`.
  - `make_reflective_dataset(candidate, eval_batch, components_to_update) -> dict[str, list[dict]]`
    — for every captured trajectory, builds `{"task_id", "score", "feedback"}`
    and appends it under every component name in `components_to_update`
    (all components see the same per-task feedback).

### `evolution/feedback.py`

Normalizes heterogeneous execution signals into `(score, feedback_text)`
pairs GEPA's reflection step consumes, and into the `EvaluationRecord`
shape.

- **`score_and_feedback(trace: Trace) -> tuple[float, str]`** — prefers
  `trace.outcome_signal["score"]`/`["detail"]` (Terminal-Bench's per-test
  pass/fail) when present; falls back to a coarse `1.0`/`0.0` from
  `trace.outcome == "success"` otherwise, noting the fallback in the
  detail text. Builds a multi-part feedback string: task/outcome/score,
  verification detail, every tool error encountered (`turn N: name failed: ...`,
  truncated to 300 chars each), and the final message (truncated to 500
  chars) if present.
- **`to_evaluation_record(trace, skill_id, skill_version, suite) -> EvaluationRecord`**
  — `suite` is `"in_domain_heldout"` or `"system_regression"`. Computes
  `cost = {"input_tokens", "output_tokens", "tool_calls"}` from
  `trace.total_usage` and a count of all tool calls across all steps.
  `task_id` is `trace.repo_context["repo"]` if present, else `trace.trace_id`.

### `evolution/gepa_runner.py`

Batch entrypoint: given a skill and a train/val task split, runs a real
`gepa.optimize()` with `SkillGEPAAdapter` to get a candidate revision.

- **`EvolutionResult` (dataclass)** — `candidate_skill_fields: dict`,
  `val_score: float`, `seed_val_score: float`, `total_evals: int`,
  `gepa_result: Any` (the raw `gepa.GEPAResult`, kept for inspection/audit).
- **`_total_evals(result) -> int`** — version-tolerant fallback chain,
  because `GEPAResult`'s eval-count attribute differs between the installed
  `gepa==0.1.4` (PyPI) and GitHub's `main` branch: tries `result.total_evals`
  first (not present on 0.1.4, but forward-compatible if it's added later),
  then `result.total_metric_calls`, then `sum(result.discovery_eval_counts)`.
  This was a real bug found and fixed during P3's build — the plan initially
  assumed `.total_evals` existed based on reading GitHub source, but the
  actually-installed release didn't have it.
- **`evolve_skill(skill, trainset, valset, run_task_fn, reflection_lm, max_metric_calls=None, max_reflection_cost=None, **optimize_kwargs) -> EvolutionResult`**
  — builds a `SkillGEPAAdapter(base_skill=skill, run_task_fn=run_task_fn)`,
  encodes the seed candidate via `skill_to_candidate`, calls
  `gepa.optimize(seed_candidate=..., trainset=..., valset=..., adapter=..., reflection_lm=..., max_metric_calls=..., max_reflection_cost=..., **optimize_kwargs)`.
  Reads `result.candidates[result.best_idx]` as the winner, decodes it back
  to skill fields, and returns an `EvolutionResult` with `seed_val_score = result.val_aggregate_scores[0]`
  (index 0 is always the seed candidate by GEPA's contract).

### `evolution/promotion_gate.py`

P3's *lite* validation gate — applied to a GEPA-proposed candidate before it
can replace an active skill. **Not** full P4: structural checks, a
bounded-edit-size check, and a contradiction check only — no cross-domain
regression suite, no shadow rollout (those are `validation/`'s job).

`MAX_CHANGED_LINES_RATIO = 0.9` — deliberately generous; catches a wholesale
rewrite from one bad batch, not normal editing.

- **`GateResult` (dataclass)** — `passed: bool`, `reason: str`.
- **`check_structure(candidate: Skill) -> GateResult`** — every required
  field (`activation`, `procedure`, `verification`) must be non-blank;
  `prerequisites` must be non-empty; `activation` must be ≤60 chars
  (the retrieval-index cap).
- **`_changed_lines_ratio(previous_text, new_text) -> float`** — via
  `difflib.SequenceMatcher` on line lists: `1 - (unchanged_lines / max(len(prev), len(new), 1))`.
- **`check_edit_size(previous, candidate, max_ratio=MAX_CHANGED_LINES_RATIO) -> GateResult`**
  — computed over `procedure + failure_recovery` text; fails if the changed-
  lines ratio exceeds `max_ratio`.
- **`ContradictionChecker(dspy.Module)`** — wraps
  `dspy.ChainOfThought(CheckContradiction)` as `self.check`.
  `forward(previous: Skill, candidate: Skill) -> GateResult` — renders both
  skills' procedure+failure_recovery text, calls the LM check, returns
  failure with the LM's explanation if `has_contradiction`.
- **`promote(previous, candidate, contradiction_checker=None) -> GateResult`**
  — runs every check in order, cheapest first, short-circuiting on the
  first failure: structural → edit-size → contradiction (an LLM call, only
  reached if the two free checks already passed).

---

<a id="validation"></a>
## 5. `validation/`

### `validation/eval_runner.py`

Runs a skill against a list of task_ids via a pluggable `run_task_fn`,
producing `EvaluationRecord`s. Shared by the in-domain held-out check and
the system regression suite.

- **`run_suite(skill, task_ids, run_task_fn, suite) -> list[EvaluationRecord]`**
  — `suite` is `"in_domain_heldout"` or `"system_regression"`. Encodes the
  skill through `skill_to_candidate` → `candidate_to_skill_fields` even
  though nothing here optimizes it — this guarantees byte-identical
  injected text between a skill scored via P3's GEPA adapter and one
  re-scored here, so there's no drift between the two scoring paths. Never
  raises for a single task's failure (records a `0.0`-score `EvaluationRecord`
  with an explanatory `feedback_text` instead), matching P3's adapter
  contract.
- **`aggregate_score(records) -> float`** — mean score, `0.0` if empty.
- **`aggregate_cost(records) -> dict[str, float]`** — mean of each cost key
  present across any record (missing keys treated as `0` for that record),
  `{}` if empty.

### `validation/regression.py`

System-level regression suite (spec §5.4): runs a fixed, cross-domain task
set against both the incumbent (active) skill and the candidate, checking
the candidate doesn't regress on aggregate score *or* cost — an "improved"
skill that succeeds by making the agent try much harder is a cost
regression even without failing outright.

`_COST_KEYS = ("input_tokens", "output_tokens", "tool_calls")`.

- **`RegressionResult` (dataclass)** — `passed`, `reason`,
  `candidate_score`, `baseline_score`, `candidate_cost: dict`,
  `baseline_cost: dict`, `candidate_records: list[EvaluationRecord]`,
  `baseline_records: list[EvaluationRecord]`.
- **`check_regression(candidate, baseline, regression_task_ids, run_task_fn, epsilon=0.05, cost_tolerance=0.25) -> RegressionResult`**
  — `baseline` is the currently active version (the caller decides what to
  do if none exists). Runs both skills through `run_suite` on the same
  task_ids. Fails if `candidate_score < baseline_score - epsilon`. Else,
  for each of `_COST_KEYS`, fails if the candidate's cost exceeds the
  baseline's by more than `cost_tolerance` (only checked when
  `baseline_cost > 0` for that key). Passes otherwise.

### `validation/promotion.py`

Combines the in-domain held-out gate and the system regression gate into a
single promote-or-reject decision, and on approval actually writes the
promotion via `SkillLibrary.promote()` — the real version of what P3's
`promotion_gate.promote()` (structural/contradiction only) never performs
itself. On rejection: the predecessor stays active, the candidate stays on
disk (`status="candidate"`) with the failure attached to its own
`provenance.validation_evidence` — visible to P3's next GEPA batch or a
human, not just silently dropped.

- **`PromotionDecision` (dataclass)** — `promoted: bool`, `reason: str`,
  `in_domain_score: float`, `predecessor_score: float | None`,
  `regression: RegressionResult | None`.
- **`_reject(library, candidate, reason, records, in_domain_score, predecessor_score, regression) -> PromotionDecision`**
  — extends `candidate.provenance.validation_evidence` with `records`,
  appends a note to `limitations`, calls `library.save(candidate)` (writing
  the evidence to disk without bumping version/status), returns a
  `promoted=False` decision.
- **`evaluate_promotion(library, skill_id, candidate_version, in_domain_task_ids, regression_task_ids, run_task_fn, margin=0.0, first_promotion_floor=0.5, epsilon=0.05, cost_tolerance=0.25) -> PromotionDecision`**
  — reads the candidate and the current `predecessor_version` (if any).
  Runs the in-domain held-out suite on the candidate and aggregates its
  score. **If no predecessor exists** (first promotion for this skill_id):
  requires clearing `first_promotion_floor` (an absolute floor, since
  there's nothing to regress against) — rejects or promotes accordingly.
  **If a predecessor exists**: also runs the held-out suite on the
  predecessor; rejects if the candidate doesn't clear
  `predecessor_score + margin` (margin is relative, deliberately distinct
  from the absolute first-promotion floor); otherwise runs
  `check_regression`, rejecting on regression. Only on passing every
  applicable gate does it call `library.promote(skill_id, candidate_version)`.

### `validation/shadow.py`

Shadow-mode rollout bookkeeping (spec §5.4): a candidate accumulates
observations from live tasks — logged, not authoritative (the active
version is still what's actually injected) — before it's eligible to spend
the full promotion gate on it. This module owns eligibility bookkeeping
only; it doesn't decide what "live" means or run anything — a production
loop calls `record_observation` as it goes.

- **`ShadowObservation` (dataclass)** — `task_id`, `score: float`.
- **`ShadowLedger`** — one JSON file per `skill_id@version` under `root`;
  not a database, fine at this scale.
  - `__init__(root)` — creates `root`.
  - `_path(skill_id, version) -> Path` — `root / f"{skill_id}@{version}.json"`.
  - `record_observation(skill_id, version, task_id, score)` — appends an
    observation and rewrites the file.
  - `_load(skill_id, version) -> list[ShadowObservation]` — reads (`[]` if
    the file doesn't exist).
  - `observations(skill_id, version) -> list[ShadowObservation]` — public
    read accessor.
  - `is_promotion_eligible(skill_id, version, min_observations=10, min_success_rate=0.5) -> bool`
    — requires at least `min_observations` recorded *and* an observed
    success rate ≥ `min_success_rate`. Documented as a concrete, simple
    proxy for the spec's "N tasks or until statistical confidence" — a real
    confidence-interval computation would tighten the check, not replace
    its shape.
  - `clear(skill_id, version)` — deletes the ledger file for that
    skill/version (call after a promotion decision, accepted or rejected,
    so stale evidence doesn't leak into a later candidate version).

### `validation/rollback.py`

Production rollback: reverts an active skill to its predecessor when *live*
monitoring (not the offline in-domain/regression suites, which already ran
pre-promotion) detects a regression. The actual swap is
`SkillLibrary.rollback()`; this module only decides whether to call it.

- **`rollback_if_regressed(library, skill_id, live_success_rate, baseline_success_rate, observed_count, min_observations=20, epsilon=0.1) -> Skill | None`**
  — returns `None` without acting if `observed_count < min_observations`
  (a rollback decided on 2 unlucky tasks is itself a reliability problem,
  not a fix for one). Otherwise calls `library.rollback(skill_id)` and
  returns the restored `Skill` if
  `live_success_rate < baseline_success_rate - epsilon`; else `None`.

---

<a id="retrieval"></a>
## 6. `retrieval/`

### `retrieval/retriever.py`

Hybrid retrieval over the active-status library: embedding similarity
(`skill_library.index.EmbeddingIndex` — the exact same artifact P2's dedup
lookup uses) plus a lexical fallback for exact tool/error-string matches
embeddings miss, re-ranked by blending similarity with each skill's track
record.

`_TOKEN_RE`.

- **`tokenize(text) -> set[str]`** — lowercase alnum tokens as a set.
- **`RetrievalCandidate` (dataclass)** — `skill_id`, `similarity: float`,
  `lexical_hit: bool`, `track_record: float`, `blended_score: float`.
- **`track_record_score(skill: Skill) -> float`** — mean score across
  `skill.provenance.validation_evidence`; `0.5` (a neutral prior) if there's
  no evidence yet — neither penalizing nor favoring an unproven skill.
- **`_lexical_hit(query_tokens, skill_text) -> bool`** — `True` if the
  token overlap between `query_tokens` and `tokenize(skill_text)` is ≥50% of
  `query_tokens`; `False` if `query_tokens` is empty.
- **`retrieve(library, index, query_text, repo_context=None, k=5, similarity_weight=0.6, track_record_weight=0.4) -> list[RetrievalCandidate]`**
  — restricted to `status == "active"` skills. Optionally filters by
  `task_type` from `repo_context`. Over-fetches `max(k*3, k)` neighbors from
  the index before re-ranking (the raw embedding top-k isn't necessarily the
  blended top-k). For each neighbor: reads the skill (skipping ones whose
  index entry outlived the skill, e.g. archived), computes `lexical_hit` and
  `track_record_score`, then blends: **the lexical-hit floor (`0.8`)
  applies only to the *similarity* component**, not the final blended score
  — `effective_similarity = max(similarity, 0.8) if lexical_hit else similarity`,
  then `blended = similarity_weight * effective_similarity + track_record_weight * track_record`.
  This is a real bug fix from P5's build: an earlier draft floored the
  *final* blended score, letting a skill with a proven 0% track record
  outrank a validated skill forever purely on an exact text match; flooring
  only the similarity component lets track record still pull a
  lexically-exact-but-bad skill back down. Returns the top `k` by
  `blended_score`, descending. An empty result is valid — callers must not
  force a weak match into context.

### `retrieval/activation.py`

Retrieval similarity alone is not permission to inject a skill —
`ApplicabilityChecker` verifies a retrieved candidate's prerequisites
actually hold for the current task/repo, with verdicts cached per
`(repo, task_type, skill_id, version)`. Also owns deterministic multi-skill
precedence ordering.

`_SEGMENT_PRECEDENCE = {"failure_recovery": 0, "tool_convention": 1, "procedure": 2, "plan": 3}`.

- **`CheckApplicability(dspy.Signature)`** — input: `task_description`,
  `repo_context` (rendered text, may be sparse), `skill_prerequisites`
  (one per line). Output: `applies: bool`, `reason: str`. Docstring: a
  prerequisite the current context can't confirm should fail the check
  rather than being assumed true.
- **`ActivationVerdict` (dataclass)** — `skill_id`, `version`, `applies: bool`,
  `reason: str`.
- **`ApplicabilityChecker(dspy.Module)`**
  - `__init__()` — wraps `dspy.ChainOfThought(CheckApplicability)` as
    `self.check`; `self._cache: dict[tuple, ActivationVerdict] = {}`.
  - `forward(skill, task_description, repo_context=None) -> ActivationVerdict`
    — cache key is `(repo, task_type, skill_id, version)`; returns the
    cached verdict on a hit, else calls the LM check, caches, and returns
    the new verdict.
- **`_dominant_kind(skill: Skill) -> str`** — approximates the skill's
  originating segment kind from content shape (`Skill` doesn't persist
  `Segment.kind`): `"failure_recovery"` if it has at least as many
  `failure_recovery` entries as `prerequisites`, else `"procedure"`.
  Documented as a coarse proxy, not ground truth.
- **`order_by_precedence(activated: list[tuple[Skill, ActivationVerdict]]) -> list[Skill]`**
  — drops any skill whose verdict didn't `apply`; sorts the rest by
  `(_SEGMENT_PRECEDENCE[_dominant_kind(skill)], -len(prerequisites))` —
  failure_recovery-oriented skills before procedure/plan-oriented ones,
  more-specific (narrower prerequisite lists) before broader, among skills
  that passed activation.

### `retrieval/adapter.py`

Rewrites an activated skill's generic procedure into task/repo-concrete
guidance for injection. Ephemeral — never written back to the library; only
genuinely new generalizable knowledge re-enters via a fresh trace through
P1. Explicitly distinct from `evolution/adapter.py`'s `SkillGEPAAdapter` —
an unrelated GEPA integration point that happens to share the word
"adapter."

- **`AdaptProcedure(dspy.Signature)`** — input: `task_description`,
  `generic_procedure`. Output: `adapted_procedure: str`. Docstring: never
  invent a specific not evidenced in `task_description`; if it doesn't name
  a path, keep the generic placeholder; preserve the original decision
  logic — this is a rewording, not a new procedure.
- **`AdaptedSkill` (dataclass)** — `skill_id`, `version`, `activation`,
  `adapted_procedure`, `verification`, `failure_recovery: list[str]`.
- **`Adapter(dspy.Module)`**
  - `__init__()` — wraps `dspy.ChainOfThought(AdaptProcedure)` as
    `self.adapt`.
  - `forward(skill, task_description) -> AdaptedSkill` — calls the LM
    rewrite, returns an `AdaptedSkill` combining the rewritten procedure
    with the skill's unmodified `activation`/`verification`/`failure_recovery`.

### `retrieval/attribution.py`

Closed-loop credit attribution: when multiple skills were co-active on one
task, attributes outcome credit per skill rather than crediting every
co-active skill equally. Heuristic, not ground truth: approximated via
lexical overlap between the trace's tool-call text and each injected
skill's adapted procedure — the skill whose wording the executed commands
most resemble gets more credit. Documented as an approximation on purpose,
matching the project's other heuristic-but-real building blocks (e.g.
`skill_library.index`'s hashing embedding).

`_TOKEN_RE`.

- **`_tokenize(text) -> set[str]`**, **`_trace_tokens(trace: Trace) -> set[str]`**
  — token set over every tool call's stringified input and output across
  every step.
- **`AttributedCredit` (dataclass)** — `skill_id`, `version`,
  `credit: float` (sums to 1.0 across all injected skills for one trace).
- **`attribute_credit(trace, injected: list[AdaptedSkill]) -> list[AttributedCredit]`**
  — `[]` if nothing was injected; `[full credit]` if exactly one skill was
  injected. For multiple: computes token overlap between `_trace_tokens(trace)`
  and each skill's `tokenize(adapted_procedure)`; if the total overlap is
  `0` (no detectable signal for any skill), splits credit evenly rather than
  fabricating a preference the trace doesn't evidence; otherwise splits
  proportionally to each skill's overlap count.

### `retrieval/bundles.py`

Pre-declared skill bundles (Hermes Agent's `skill_bundles.py` pattern): a
named set of skill_ids that reliably co-activate, injected as a unit
instead of relying on per-task dynamic precedence resolution for that
specific combination. A compactness/reliability win for known-good
combinations; `activation.order_by_precedence` remains the fallback for
combinations no one has pre-declared.

- **`Bundle` (dataclass)** — `name`, `skill_ids: list[str]`,
  `description: str = ""`.
- **`BundleStore`**
  - `__init__(root)` — creates `root`.
  - `_path(name) -> Path` — `root / f"{name}.json"`.
  - `save(bundle: Bundle)` — writes the bundle as JSON.
  - `load(name) -> Bundle | None` — reads it back, `None` if absent.
  - `all() -> list[Bundle]` — every stored bundle, sorted by filename
    (alphabetical, deterministic).
  - `matching(available_skill_ids: set[str]) -> Bundle | None` — the first
    stored bundle (in `all()`'s order) whose `skill_ids` is a *subset* of
    `available_skill_ids` — a known-good combination that's actually
    applicable here (every member was retrieved/activated), not a bundle
    naming skills that weren't.

### `retrieval/injection.py`

Ties retrieval + activation + adaptation + the safety gate into one call:
given a task, builds the system-prompt addendum `TraceCollector` should run
with. Supersedes `evolution/skill_injection.py`'s hardcoded single-skill
stub for real task execution; that module still exists for its own,
narrower job — P3's `evaluate()` deliberately injects one *fixed* candidate
under test, bypassing retrieval entirely, which is correct for
optimization (the point is scoring an exact candidate, not picking one).

`MAX_INJECTED_SKILLS = 3`, `MAX_TOTAL_CHARS = 6000` (a crude
character-count proxy for a token budget — good enough to cap injection
size without a tokenizer dependency).

- **`InjectionResult` (dataclass)** — `system_prompt_addendum: str | None`,
  `injected: list[AdaptedSkill]`, `activation_failures: list[tuple[str, str]]`
  (skill_id, reason), `safety_blocked: list[tuple[str, str]]`.
- **`build_injection(library, index, task_description, repo_context=None, applicability_checker=None, adapter=None, k=5, max_injected=MAX_INJECTED_SKILLS, max_total_chars=MAX_TOTAL_CHARS) -> InjectionResult`**
  — a `None` addendum is a valid, expected outcome (no candidates retrieved,
  none passed activation, or all adaptations were safety-blocked); the
  caller then runs unassisted. Pipeline: `retrieve(...)` → for each
  candidate, read the skill and run `ApplicabilityChecker`, splitting into
  `activated` vs `activation_failures` → `order_by_precedence(activated)` →
  for each ordered skill up to `max_injected`: adapt it, run
  `check_adapted_text` on the adapted procedure (skip + record to
  `safety_blocked` if unsafe), stop accumulating once adding it would
  exceed `max_total_chars`. Returns `InjectionResult(None, [], activation_failures, safety_blocked)`
  if nothing survived, else the rendered addendum plus everything injected.
- **`render_addendum(injected: list[AdaptedSkill]) -> str`** — tags each
  block with `[skill {skill_id}@{version}]` (per spec §5.5, so a resulting
  trace's provenance can record unambiguously which skill version was
  active — this is what `retrieval/attribution.py` keys credit on), followed
  by the skill's activation, adapted procedure, and verification text. Wraps
  all blocks with an "advisory, not binding" framing sentence.

### `retrieval/safety_gate.py`

Runtime safety gate: clearing P4's offline validation does not guarantee an
adapted, repo-concrete instruction is safe to execute in *this* live repo —
e.g. a shell command inert against the sandbox that validated it but
destructive here. Applied at injection time, independent of and in addition
to P3/P4's publish-time gates.

`_DANGEROUS_PATTERNS` — compiled regexes for: `rm -rf /`, `rm -rf ~`, a
shell fork bomb, `mkfs.*`, `dd ... of=/dev/sd|nvme|hd*`, `chmod -R 000`,
redirect-to-raw-disk (`> /dev/sd*`), and `curl|wget ... | (sudo) sh|bash`.

- **`SafetyVerdict` (dataclass)** — `safe: bool`, `reason: str`.
- **`check_adapted_text(adapted_procedure: str) -> SafetyVerdict`** —
  returns unsafe with the matched pattern on the first `_DANGEROUS_PATTERNS`
  match, else safe. Deliberately conservative and narrow: catches obviously
  destructive literal commands an adaptation introduced; explicitly **not**
  a sandbox and not a substitute for actually executing untrusted commands
  in isolation (that's `tool_handlers.py`/`container_tool_handlers.py`'s
  job) — it only stops a dangerous literal from reaching the agent's context
  as "recommended" guidance in the first place.

---

<a id="orchestrator"></a>
## 7. `orchestrator/`

### `orchestrator/driver.py`

The top-level driver tying P1–P5 into one loop — closing the gap flagged in
`IMPLEMENTATION_PLAN.md`: every phase was built and tested against its
immediate neighbor, but nothing before this called them in sequence against
a live task stream. Two entrypoints matching the spec's separation between
fast per-task live behavior and slow batched maintenance. Every external
effect (running the agent, running a GEPA/validation task, calling a
reflection LM) is injected as a callable — testable with fakes/`DummyLM`,
not something that only works against live infrastructure. **Not built
here**: live shadow-traffic mirroring (silently running an unpromoted
candidate alongside the active skill on real traffic) — `validation/shadow.py`'s
ledger exists and is ready for it, but the mirroring mechanism itself was
judged out of scope.

`CollectTraceFn = Callable[[str, "str | None"], Trace]` — `(task_description, system_prompt_addendum | None) -> Trace`,
wrapping whatever actually executes the agent (`TraceCollector.run()`
locally, or `tbench_adapter.run_tbench_task`'s container path). The caller
owns constructing the underlying collector; the driver only needs the
resulting `Trace`.

- **`TaskResult` (dataclass)** — `trace: Trace`, `injection: InjectionResult`,
  `credits: list[AttributedCredit]`, `drafts: list[SkillDraft]`,
  `decisions: list[tuple[SkillDraft, LibrarianDecision, Skill | None]]`.
- **`Orchestrator`** — holds the shared, long-lived state (library + index)
  both entrypoints read/write. One instance per running system, not per
  task.
  - `__init__(library, index, librarian=None)` — defaults `librarian` to
    `Librarian(library, index)`.
  - `handle_task(task_description, collect_trace_fn: CollectTraceFn, repo_context=None) -> TaskResult`
    — the live path, one task start to finish. Requires an LM already
    configured (same contract as `extract_skills`/`Librarian` — this method
    doesn't configure one). Pipeline: `build_injection(...)` (P5, active
    skills only) → `collect_trace_fn(task_description, injection.system_prompt_addendum)`
    to get a `Trace` → prefers whatever `collect_trace_fn` already
    determined for `repo_context` (e.g. a real Terminal-Bench task's own
    category) over the caller's possibly-generic one, filling in only if
    unset → `attribute_credit(trace, injection.injected)` → `extract_skills(trace)`
    (P1) → for each resulting draft, `self.librarian(draft, repo_context)`
    then `apply_decision(...)` (P2), collecting `(draft, decision, skill)`
    triples.
- **`EvolutionCycleResult` (dataclass)** — `skill_id`, `status: str`
  (`"no_improvement"` | `"lite_gate_rejected"` | `"promotion_rejected"` |
  `"promoted"`), `reason: str`, `evolution: EvolutionResult | None`,
  `candidate_skill: Skill | None`, `promotion: PromotionDecision | None`.
- **`run_evolution_cycle(library, index, skill_id, gepa_trainset, gepa_valset, gepa_run_task_fn, reflection_lm, in_domain_task_ids, regression_task_ids, promotion_run_task_fn, contradiction_checker=None, max_metric_calls=None, max_reflection_cost=None, margin=0.0, first_promotion_floor=0.5, epsilon=0.05, cost_tolerance=0.25, **gepa_optimize_kwargs) -> EvolutionCycleResult`**
  — the offline maintenance cycle for one skill: P3 `evolve_skill` → if the
  winning candidate doesn't beat the seed on valset, returns
  `"no_improvement"` immediately. Otherwise applies the candidate's fields
  onto the seed skill (`apply_candidate_fields`), runs the lite gate
  (`run_lite_gate` = `evolution.promotion_gate.promote`); on failure returns
  `"lite_gate_rejected"` with its reason. On success, writes the updated
  fields as a new candidate version via `library.revise(...)`, then calls
  P4's real `evaluate_promotion(...)`. Returns `"promoted"` or
  `"promotion_rejected"` accordingly, carrying the full `EvolutionResult`,
  the new `candidate_skill`, and the `PromotionDecision`. Every stage's
  rejection is a legitimate, logged outcome, not an exception.

### `orchestrator/cli.py`

CLI entrypoint tying the orchestrator to real Terminal-Bench task
execution. Requires `terminal-bench` (Python ≥3.12) and a running Docker
daemon for real execution — imported lazily inside each command's own
function so this module and its argument parsing stay importable/testable
without either.

Deliberately **two separate model flags**, not one: `--model` is
Anthropic-SDK-style (`"claude-opus-5"` — what `TraceCollector`/
`run_tbench_task` need) while `--reflection-model` is litellm-style
(`"anthropic/claude-opus-5"` — what `gepa.optimize()`'s `reflection_lm`
expects). Collapsing these into one flag would silently break whichever
path got the other format. `dspy_modules.lm_config`'s own model (used by
`extract_skills`/`Librarian`/`ApplicabilityChecker`/`Adapter`) is
deliberately *not* exposed as a flag here — it already reads
`SKILLGEN_DSPY_MODEL` from the environment if an override is needed.

- **`rebuild_index(library: SkillLibrary) -> EmbeddingIndex`** —
  `EmbeddingIndex` is in-memory only, so a fresh CLI invocation rebuilds it
  from every active skill's `activation + prerequisites` text on start.
  `O(library size)`; fine at this scale, would need revisiting only if the
  library got much larger.
- **`load_task_instruction(tasks_dir, task_id) -> tuple[str, dict[str, Any]]`**
  — reads a Terminal-Bench task's own instruction + category from
  `task.yaml` via `TrialHandler`, without starting its container. Returns
  `(instruction, {"repo": task_id, "language": None, "task_type": category})`.
- **`make_tbench_collect_trace_fn(tasks_dir, task_id, model) -> CollectTraceFn`**
  — returns a closure matching `orchestrator.driver.CollectTraceFn`'s
  signature; the `task_description` argument is accepted but **ignored** —
  a Terminal-Bench task's actual instruction always comes from its own
  `task.yaml` (`load_task_instruction`), not from whatever string
  `Orchestrator.handle_task` was given for retrieval purposes. Calls
  `run_tbench_task(tasks_dir, task_id, model=model, system_prompt_addendum=system_prompt_addendum)`.
- **`make_tbench_run_task_fn(tasks_dir, base_skill, model) -> RunTaskFn`** —
  shared by `run_evolution_cycle`'s P3 (`gepa_run_task_fn`) and P4
  (`promotion_run_task_fn`) calls, both of which need the same "decode
  candidate fields onto `base_skill`, render, inject" step against different
  task batches. Returns a closure: `apply_candidate_fields(base_skill, skill_fields)` →
  `build_system_prompt_addendum(skill)` → `run_tbench_task(tasks_dir, task_id, model=model, system_prompt_addendum=addendum)`.
- **`_build_parser() -> argparse.ArgumentParser`** — two subcommands:
  - `handle-task`: `--library-dir`, `--tasks-dir`, `--task-id` (all
    required), `--model` (default `DEFAULT_COLLECTOR_MODEL`), `--out-dir`
    (default `./traces`).
  - `evolve`: `--library-dir`, `--tasks-dir`, `--skill-id`, `--gepa-train`,
    `--gepa-val`, `--in-domain`, `--regression` (all required,
    comma-separated task-id lists for the four task groups), `--model`,
    `--reflection-model` (required), `--max-metric-calls` (default 60),
    `--margin` (default 0.0), `--first-promotion-floor` (default 0.5).
- **`main(argv=None) -> int`** — parses args, opens the `SkillLibrary`,
  rebuilds the index, calls `configure_lm()` (dspy_modules' own model —
  `SKILLGEN_DSPY_MODEL` overrides if needed). For `handle-task`: loads the
  task instruction, builds an `Orchestrator`, runs `handle_task`, saves the
  trace via `trace_collection.collector.save_trace`, prints the trace path
  + outcome, injected skill ids, per-skill credit, and every extraction
  decision. For `evolve`: reads the base skill, builds the shared
  `run_task_fn`, calls `run_evolution_cycle(...)` with the parsed
  comma-separated task lists, prints the cycle's status/reason plus the
  candidate skill id and in-domain/predecessor scores if produced.

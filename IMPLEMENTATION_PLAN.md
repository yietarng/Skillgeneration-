# Implementation Plan: P1 → P2 → P3

Status: draft v0.1 · converts `PROJECT_SPEC.md` into a sequenced, buildable plan
Scope for this plan: **P1 (Trajectory Segmentation and Extraction) → P2 (Skill-Library Maintenance) → P3 (Weight-Free Skill Evolution via GEPA)**. P4 (full validation/regression harness) and P5 (retrieval/activation/adaptation) are **not** built out here beyond the minimal stubs P3 needs to close its own loop — see §6.

## 1. Trace corpus: Terminal-Bench

Trace collection needs real, verifiably-scored tasks to be worth anything to P1/P3. [Terminal-Bench](https://github.com/laude-institute/terminal-bench) is the source for this plan. **Status: built (`container_tool_handlers.py`, `tbench_adapter.py`), grounded in the real cloned source and import-verified against the actual installed package, but never run against a live container — see the caveats below before trusting it for real trace collection.**

Confirmed against source, correcting this section's earlier (search-engine-sourced) guesses:
- Each task directory (e.g. `original-tasks/<task-id>/`) has `task.yaml` (an `instruction:` field, not a separate `instruction.md`, plus `category`/`difficulty`/`tags`/`parser_name`/timeouts), `Dockerfile` + `docker-compose.yaml`, `run-tests.sh`, `tests/`, and `solution.sh`.
- Verification is `run-tests.sh` executed inside the container, with its output parsed by a `parser_name`-selected parser (`terminal_bench.parsers.ParserFactory`; defaults to pytest, other parsers exist for swebench/mlebench/etc.) into a `dict[str, UnitTestStatus]` — this *is* `Trace.outcome_signal` (§4.1 of the spec): `{"kind": "test", "score": pass_fraction, "detail": "<test>=<passed|failed>, ..."}`. No LLM-judge fallback needed for this corpus.
- `task.category`/`task.tags` supply `repo_context` (task domain: system-administration, security, etc.) for free.
- Terminal-Bench's own agents interact with the container through a **tmux session** (`terminal_bench.terminal.tmux_session.TmuxSession`: send keystrokes, block until a `tmux wait` sentinel, capture the pane), not a plain `docker exec` per command. This is genuinely a better match for Claude's `bash_20250124` tool than `tool_handlers.SandboxedToolRunner`'s per-call `subprocess.run` — a tmux session, like the real bash tool, persists env vars/cwd/background jobs across calls; the existing local runner doesn't.

**Built as:**

```
trace_collection/
  tool_handlers.py            # unchanged: SandboxedToolRunner (local dir) — kept for fast dev iteration
  container_tool_handlers.py    # NEW: ContainerToolRunner. bash tool -> TmuxSession.send_keys(block=True)
                                  # + get_incremental_output(); text-editor tool -> session.container.exec_run
                                  # / session.copy_to_container directly (bypasses tmux -- structured file
                                  # I/O doesn't need to appear in the recorded terminal transcript). Only
                                  # imports terminal_bench under TYPE_CHECKING, so it has no hard runtime
                                  # dependency on the package and stays unit-testable (tests/test_
                                  # container_tool_handlers.py, a fake-session double, no Docker needed) in
                                  # the same environment as everything else.
  tbench_adapter.py               # NEW: run_tbench_task(tasks_dir, task_id, ...) -- loads the task via
                                    # terminal_bench.handlers.trial_handler.TrialHandler, starts its
                                    # container via terminal_bench.terminal.terminal.spin_up_terminal, runs
                                    # TraceCollector against a ContainerToolRunner-backed session, then
                                    # reuses Terminal-Bench's own test-copy + run-tests.sh invocation +
                                    # ParserFactory-selected parser for verification (not a reimplementation)
                                    # to fill Trace.outcome_signal. Has a real terminal_bench runtime
                                    # dependency, unlike container_tool_handlers.py.
```

**Correction to this plan's earlier assumption:** it said "`collector.py`'s `TraceCollector` doesn't need to change." Checking the actual code showed it hardcoded `self.tools = SandboxedToolRunner(workdir)` in `__init__` — there was no way to inject a different runner. Fixed by adding optional `tool_runner`/`tool_defs` constructor params (default behavior for the existing `collect` CLI path is unchanged; verified via the existing test suite plus a manual construction check). `tbench_adapter.py` passes `tool_runner=ContainerToolRunner(...)` and a descriptive `workdir="container:<name>"` string (used only for `Trace.workdir` reporting, since there's no local directory).

**New environment constraint, discovered while building this, not anticipated in the original plan:** the `terminal-bench` PyPI package requires **Python ≥3.12** (confirmed: `pip install terminal-bench` is flatly rejected under 3.11 — "Requires-Python >=3.12"). This repo's core `requirements.txt` (dspy/gepa/anthropic) has no such constraint. Keep them decoupled — `requirements-tbench.txt` is a separate, optional file with this spelled out, meant to be installed into its own 3.12+ virtualenv rather than forcing the whole project onto a newer interpreter. `container_tool_handlers.py`'s `TYPE_CHECKING`-only import of `TmuxSession` is what keeps it usable outside that venv.

**What was actually verified, and what wasn't (no Docker daemon in the environment that built this):**
- ✅ `container_tool_handlers.py`'s tool logic (path resolution, `view`/`create`/`str_replace`/`insert`, bash command wrapping) — unit-tested against a fake session double, 14/14 passing.
- ✅ Both new modules' imports, and every `terminal_bench` class/method signature they call (`TrialHandler.__init__`, `spin_up_terminal`, `TmuxSession.send_keys`/`copy_to_container`/`capture_pane`/`get_incremental_output`/`clear_history`, `DockerComposeManager.CONTAINER_TEST_DIR`) — checked by reading the real cloned source, then confirmed by installing the actual `terminal-bench` package into a Python 3.12 venv and importing both modules against it successfully.
- ❌ An actual container run: starting a real task container, running the agent loop through a live `TmuxSession`, copying in `tests/`, running `run-tests.sh`, and parsing real results. **Nothing here has executed against Docker.** Treat `tbench_adapter.py` as spec-accurate, not battle-tested — run it against one real task before trusting its output for anything.

**Pinned subset, not the full suite — still to do, needs a live run to curate responsibly:** the original plan called for a fixed, versioned list of ~15–20 Terminal-Bench task ids in `eval/tbench_task_ids.txt`, spanning 2–3 domains, chosen so P1/P2 see *repeated* structurally-similar tasks. That selection wasn't made — picking specific task ids without ever having run one, in an environment that can't run one, would be guessing, not curating. Do this once Docker + Python 3.12 are available: run a handful of candidate tasks, confirm they complete and produce sensible traces, then pin the list.

## 2. Dependencies

`requirements.txt` now includes:
```
anthropic>=0.70
dspy>=3.2
gepa>=0.1
```
`dspy` for the P1/P2 modules (`Segmenter`, `AbstractionJudge`, `Abstractor`, `Librarian`); `gepa` (the standalone package, not only `dspy.GEPA`) for P3 — see §5 for why.

A `dspy.LM` wrapper around the existing Anthropic client setup goes in `dspy_modules/lm_config.py` so all DSPy modules share one configured backend; this is also the seam where swapping backend models later (per the spec's model-agnostic requirement) happens in one place.

## 3. Phase P1 — Segmentation and Extraction

**New module:** `dspy_modules/`
```
dspy_modules/
  __init__.py
  lm_config.py         # dspy.LM(...) configured from ANTHROPIC_API_KEY; dspy.configure(lm=...)
  signatures.py         # dspy.Signature classes for every module below
  p1_extraction.py       # Segmenter, AbstractionJudge, Abstractor + extract_skills(trace) orchestrator
```

**Signatures (in `signatures.py`):**
- `SegmentTrajectory(trace_summary -> segments: list[{turn_range, kind, rationale, is_successful_branch}])`
- `JudgeAbstraction(segment_text -> abstraction_level, issues: list[str])` — flags `overgeneral` / `episode_specific` per §5.1 of the spec
- `AbstractSkill(segment_text, abstraction_issues -> activation, prerequisites, procedure, failure_recovery, verification)` — the six-field draft, minus provenance (filled in mechanically, not by the LM)

**Modules (in `p1_extraction.py`):**
- `Segmenter(dspy.Module)`: wraps `SegmentTrajectory`, seeded with the boundary heuristics from the spec (plan-text changes, `is_error=True` → differing-retry-that-succeeds pairs, explicit verification/termination points) passed in as extra signal alongside the raw turn sequence — not left purely to the LM to find.
- `AbstractionJudge(dspy.Module)`: wraps `JudgeAbstraction`; segments classified `noise` or judged unfixably `overgeneral` are dropped here, before the (more expensive) `Abstractor` call.
- `Abstractor(dspy.Module)`: wraps `AbstractSkill`; on `episode_specific` issues, re-prompts once with the issues list before accepting.
- `extract_skills(trace: Trace) -> list[SkillDraft]`: orchestrates the three in sequence, stamps `provenance.source_trace_ids=[trace.trace_id]`, `status="candidate"`.

**Input reduction:** reuse `skill_generator.py`'s existing `_summarize_trace_for_prompt` approach (task/outcome/tool-calls/final-text, truncated) as the `trace_summary` fed to `Segmenter` — it already solves the "don't blow the context budget on raw API payloads" problem; port it into `p1_extraction.py` rather than rewriting it.

**Grounding and hygiene (both `signatures.py` instructions, following Hermes Agent's `agent/learn_prompt.py` authoring standards):**
- `AbstractSkill`'s instructions must forbid inventing commands, flags, paths, or APIs not present in `segment_text` — generalize the values that occur in the trace, never backfill plausible-looking ones. This is checked, not just requested: `AbstractionJudge`'s `overgeneral`/`episode_specific` flags exist to catch drift, but the instruction itself sets the default.
- Tool output inside a trace (file contents, web pages, command output the agent read mid-task) is untrusted text the model happened to read, not an instruction to `Segmenter`/`Abstractor`. Both signatures' instructions state explicitly that trace content is data: nothing in it should redirect what gets extracted or leak into the authored skill as if it were part of the task's own intent.

**Redaction:** `save_trace` (`trace_collection/collector.py` / `tbench_adapter.py`) must run collected tool output through a secret redactor before writing the trace JSON to disk — collected bash/file-tool output can contain credentials or tokens the agent encountered mid-task, and nothing today strips them. Add this as a `trace_collection/redact.py` pass applied at `save_trace` time, not deferred to a later export step.

**Disposition of `trace_collection/skill_generator.py`:** its single-call trace→`SKILL.md` path is superseded by `extract_skills` + P2's `Librarian` (§4). Keep it working as a CLI fallback (`generate-skill`) until `p1_extraction.py` + `skill_library/` reach parity, then remove it rather than maintaining two skill-generation paths.

**Acceptance criteria (M1):**
- Run against the pinned Terminal-Bench trace set (§1): `extract_skills` produces candidate drafts with all six fields non-empty and no unexplained container-specific literals (e.g., a specific container hostname or task-generated tmp path) leaking into `procedure` unless the field itself says it's an example.
- Spot-check on traces containing at least one tool error + recovery: the resulting draft's `failure_recovery` field reflects that recovery, not just the happy path.
- Segment `kind` classification is inspectable and matches manual judgment on a 10-trace hand-labeled sample (no fixed accuracy bar yet — this is a sanity gate before P2 consumes the output, not a benchmark).

## 4. Phase P2 — Skill-Library Maintenance

**New module:** `skill_library/`
```
skill_library/
  __init__.py
  storage.py     # Skill / Provenance dataclasses (spec §4.3); read/write SKILL.md + metadata.json per
                   # skill; archive()/restore() move a skill to/from .archive/ (never delete); pin()/unpin()
  index.py         # embedding index over `activation` + `prerequisites` text (used for dedup lookup here;
                     # becomes the P5 retrieval index later — same artifact, no rework needed); excludes
                     # archived skills
  librarian.py       # Librarian(dspy.Module) + apply_decision() that executes it against storage.py;
                       # sweep_retirements() (calls storage.archive(), never a hard delete)
```

**Signature:** `LibrarianDecide(candidate_skill, nearest_existing: list[Skill] -> action: Literal[CREATE,REVISE,MERGE,SPECIALIZE,REJECT], target_skill_id: str | None, rationale: str)`.

**`Librarian(dspy.Module)` flow:** embed the candidate's `activation`+`prerequisites` via `index.py`, pull top-k nearest existing skills (filtered by `repo_context`/`task_type` when set), call `LibrarianDecide`. `RETIRE` (§5.2 of the spec) is not an LLM decision — it's a scheduled, metrics-driven sweep (`librarian.py::sweep_retirements()`) over utilization×success-rate, run separately from the per-candidate decision path.

**Invariants (`storage.py`/`librarian.py`), matching Hermes Agent's `agent/curator.py` since these are the difference between a maintenance system and a liability:**
- `sweep_retirements()` archives, it never deletes: an archived skill's files move to `skill_library/.archive/<skill_id>/` (out of `index.py`'s retrieval set) with `status="archived"`, and `storage.py` exposes a `restore(skill_id)` that moves it back.
- Every decision path (`Librarian` calls and `sweep_retirements()` alike) skips any skill with `pinned=True` before doing anything else — no `REVISE`/`MERGE`/`SPECIALIZE`/`RETIRE` ever touches a pinned skill. `storage.py` needs a `pin`/`unpin` entrypoint; nothing automated ever flips that flag.
- `Librarian` only ever proposes actions against skills with non-empty `provenance.source_trace_ids` (i.e., skills `extract_skills` produced). If the library is later seeded with any hand-authored skill, it has no `source_trace_ids` and is invisible to `Librarian`'s decision path by construction — it's a human's file to edit directly.

**No auto-write to `active`:** every `CREATE`/`REVISE`/`MERGE`/`SPECIALIZE` output from `Librarian` is written with `status="candidate"`, never directly overwriting a `status="active"` skill. In this plan's scope (no full P4 yet), promotion to `active` uses the **lite gate** described in §6 — this is the one place P2 depends on something from P3/§6 rather than the reverse.

**Acceptance criteria (M2):**
- Feed drafts from 2+ Terminal-Bench traces that hit the *same* underlying pattern (e.g., two different tasks both requiring a similar diagnostic-then-fix procedure): `Librarian` proposes `MERGE` or `REVISE`, not two independent `CREATE`s — verified by hand on this constructed pair before trusting it on the full pinned set.
- Feed a draft with no existing overlap: `Librarian` proposes `CREATE` and it's `status="candidate"` on disk (`SKILL.md` + `metadata.json`) but the library's `active` set is unchanged until the lite gate (§6) runs.
- `index.py`'s nearest-neighbor lookup returns the constructed near-duplicate pair above as each other's top match (basic retrieval sanity check, reused later for P5).

## 5. Phase P3 — Weight-Free Evolution via GEPA

**Status: built (`evolution/`) and verified with a real, live `gepa.optimize()` run** — not mocked. A fake `reflection_lm` (a plain Python callable returning a fixed corrected instruction, per GEPA's `Signature.output_extractor` contract — a backtick-fenced block) and a fake `run_task_fn` (deterministic scoring, no Docker/LLM) drove an actual GEPA optimization loop through real candidate proposal, minibatch acceptance, and Pareto/best-candidate tracking. Result: seed scored 0.0, GEPA proposed a fix, evolved candidate scored 1.0 and was selected as best — genuine evidence the wiring works, not just that the code imports. This caught two real bugs invisible from source-reading alone:
- `GEPAAdapter`'s reflective-mutation proposer accesses `adapter.propose_new_texts` by direct attribute access, not `getattr(..., None)` — an adapter that simply omits the attribute crashes with `AttributeError` on every iteration. Fixed: `SkillGEPAAdapter.propose_new_texts = None` as an explicit class attribute (opts into the default `reflection_lm`-driven proposal).
- `GEPAResult.total_evals` — read from source on GitHub's `main` branch and assumed present — **does not exist on the actual installed package** (`pip install gepa` resolves to the PyPI release `0.1.4`, which predates that property; only `total_metric_calls`/`discovery_eval_counts` exist there). `gepa_runner.py`'s `_total_evals()` now replicates the newer version's fallback chain locally rather than depending on an attribute that may not be there. `requirements.txt` bumped to `gepa>=0.1.4` (the version this was actually verified against, not a guess).

Both bugs are the kind that only surface by actually running the code against the real installed package — reading cloned GitHub source (as done when writing this plan's earlier §5) gets the *shape* of the API right but can silently drift from what's on PyPI.

**Why the standalone `gepa` package, not `dspy.GEPA`:** confirmed against source (`dspy/teleprompt/gepa/gepa.py`) — `dspy.GEPA` is a `Teleprompter` that compiles a `dspy.Module`'s named *predictors* via a `GEPAFeedbackMetric`, calling the module directly. The system actually being evaluated here is `TraceCollector`'s manual agentic loop against the raw Anthropic API (bash + text-editor tools) — not a `dspy.Module`. `gepa`'s own `GEPAAdapter` protocol (`src/gepa/core/adapter.py`) is built for exactly this case: "optimize a textual component of *any* system," where the system is opaque to GEPA beyond the adapter's `evaluate`/`make_reflective_dataset`. Use `gepa.optimize()` + a custom `GEPAAdapter` directly, not `dspy.GEPA`. (If the agent loop is ever rewritten as a `dspy.Module` — noted as a possible future refactor in §5.3 of the spec — `dspy.GEPA` becomes the natural fit and this adapter can retire.)

**New module:** `evolution/`
```
evolution/
  __init__.py
  feedback.py        # normalize outcome_signal + pytest output + tool is_error strings into
                       # (score: float, feedback_text: str) — the EvaluationRecord shape (spec §4.4)
  skill_injection.py   # MINIMAL P5 stub (see below) — puts one skill's current text bundle into
                         # TraceCollector's SYSTEM_PROMPT for a run; no retrieval/ranking/adaptation
  adapter.py             # SkillGEPAAdapter(gepa.GEPAAdapter): evaluate() + make_reflective_dataset()
  gepa_runner.py            # batch entrypoint: given a skill_id and a task subset it applies to,
                              # run gepa.optimize(...) with SkillGEPAAdapter, get back a candidate
  promotion_gate.py           # lite validation gate — see §6
```

**Candidate encoding:** `gepa`'s `Candidate` type is `dict[str, str]` — values must be plain strings, but `prerequisites` and `failure_recovery` are `list[str]` in the Skill schema (§4.3 of the spec). `adapter.py` needs a `skill_to_candidate(skill) -> dict[str,str]` / `candidate_to_skill_fields(candidate) -> dict` pair that (de)serializes each list field to/from a single newline- or bullet-joined string component, so the three optimized components are literally `{"procedure": str, "prerequisites": str, "failure_recovery": str}`.

**`SkillGEPAAdapter.evaluate(self, batch, candidate, capture_traces=False)`** (argument order matches `GEPAAdapter.evaluate` in `src/gepa/core/adapter.py` — `batch` first, `candidate` second):
1. `candidate` is the skill's optimizable text bundle from the encoding above (`activation`, `verification`, and `provenance` are not GEPA-optimized text — they're not in the candidate dict at all).
2. For each task in `batch` (drawn from the pinned Terminal-Bench subset that this skill's `activation` covers): decode `candidate` back to skill fields, inject via `skill_injection.py`, run `TraceCollector.run()` against a fresh container for that task, collect the resulting `Trace`.
3. Run the task's pytest verification → `outcome_signal` → `feedback.py` → `(score, feedback_text)`.
4. Return an `EvaluationBatch(outputs=..., scores=..., trajectories=... if capture_traces else None)` — `len(outputs) == len(scores) == len(batch)` is a hard contract. **Never raise for a single task's failure** (a crashed container, a timeout): per `GEPAAdapter`'s documented contract, catch it and return that example a fallback score (`0.0`) with the failure captured in its trajectory instead — reserve real exceptions for systemic failures (e.g. the container runtime itself is down), which are controlled by `optimize()`'s `raise_on_exception` policy.

**`SkillGEPAAdapter.make_reflective_dataset(...)`:** surfaces, per example, what the injected skill said vs. what the agent actually did vs. what failed — this is what GEPA's reflective LM reads to propose the next candidate's `procedure`/`prerequisites`/`failure_recovery` text. This is the direct implementation of the spec's "convert heterogeneous execution feedback into controlled textual revisions."

**Held-out split and budget — use GEPA's own, don't rebuild them:** `gepa.optimize(seed_candidate, trainset, valset, adapter, reflection_lm, max_metric_calls, max_reflection_cost, run_dir, ...)` takes `trainset` and `valset` as separate arguments and already tracks `val_aggregate_scores` per candidate for Pareto/acceptance decisions — that *is* M3's "fresh held-out batch," so `gepa_runner.py` should split the skill's covered pinned-task list into `trainset`/`valset` and pass both, rather than building a separate held-out check downstream. Likewise, `max_metric_calls`/`max_reflection_cost` are real arguments on `optimize()` that cap cost during the run itself — pass them explicitly (from a per-skill budget) instead of only logging spend after the fact as originally planned. `gepa_runner.py` reads the winning version off the returned `GEPAResult`: `result.candidates[result.best_idx]` (equivalently `result.best_candidate`) and `result.val_aggregate_scores[result.best_idx]`.

**Drift/contradiction controls (§5.3 of the spec), concretely — applied to `result.best_candidate` after `gepa.optimize()` returns, as a downstream gate GEPA itself doesn't provide:**
- `promotion_gate.py` rejects the candidate outright if the diff against the previous version exceeds a changed-lines ratio threshold, or if any of the six required fields becomes empty/near-empty (after decoding the candidate dict back to skill fields).
- A cheap LLM-judge call (reuses `dspy_modules/signatures.py`, new `CheckContradiction` signature) compares old vs. new `procedure`/`failure_recovery` text; a flagged contradiction blocks auto-promotion and requires the candidate to be written to disk as `status="candidate"` for manual review instead.

**Acceptance criteria (M3): met, with one substitution.**
- ✅ End-to-end demonstration: seed skill v1 → `gepa_runner.evolve_skill()` with a real train/val split → `GEPAResult` whose best candidate's `val_aggregate_scores` (1.0) beats v1's (0.0) on `valset` → `promotion_gate.promote()` passes it (`tests/test_gepa_runner.py`). `valset` is scored by GEPA itself throughout the run, exactly as planned — no separate re-run constructed.
- ✅ `promotion_gate.py` demonstrably rejects deliberately-bad candidates in tests: empty `procedure`, empty `prerequisites`, `activation` over 60 chars, a wholesale rewrite exceeding the changed-lines cap, and (via a `DummyLM`-scripted `has_contradiction=True`) a flagged contradiction — all in `tests/test_promotion_gate.py`.
- **Substitution, not a gap:** "one skill from the pinned set" used a fake `run_task_fn` and fake `reflection_lm` instead of real Terminal-Bench tasks and a real LLM, because neither Docker nor a usable API key was available in the environment that built this (same constraint noted in §1). The GEPA *mechanics* (candidate proposal → evaluation → acceptance → Pareto tracking → promotion gate) are genuinely verified; the *quality* of real reflective proposals and real task scoring is not — that needs an environment with both, per §8's risks.

## 6. What's deliberately a stub here, and why

This plan's P3 needs *something* that puts a skill's text into a running agent to score it — that's unavoidably a sliver of P5 (retrieval/activation/adaptation) and a sliver of P4 (validation/promotion). Both are built **only to the minimum P3 needs**, not to the spec's full design:

| Full spec component | What this plan builds instead | Gap left for later |
|---|---|---|
| P5 retrieval (embedding+BM25+metadata ranking) | `skill_injection.py`: direct, hardcoded injection of one named skill — no ranking, no multi-skill composition, no `ApplicabilityChecker` | Real retrieval, activation-condition checking, adaptation to repo-concrete specifics, runtime safety gate (spec §5.5) |
| P4 validation (in-domain held-out + system regression suites, shadow rollout) | `promotion_gate.py`: structural checks + contradiction check + a single held-out batch within the same skill's task set | Cross-domain system regression suite, cost-regression checks, shadow-mode rollout, rollback mechanics (spec §5.4) |

Treat `skill_injection.py` and `promotion_gate.py` as throwaway-if-needed scaffolding: when P4/P5 are actually built out, they should absorb and generalize these, not sit alongside them as a permanent second path.

## 7. Sequencing and dependencies

```
P1 (extract_skills)
   │  candidate drafts
   ▼
P2 (Librarian, storage, index)
   │  status="candidate" skills on disk
   ▼
P3 (gepa_runner, promotion_gate)  ── needs skill_injection.py (P5 stub) to run evaluate()
   │  status="active" skills (lite-gated)
   ▼
(closed loop: active skill's injected runs produce new traces → back to P1)
```

Build strictly in this order — P2's `Librarian` needs real candidate drafts from P1 to have anything to decide on, and P3's `evaluate()` needs P2's storage format to read/write skill versions.

## 8. Risks specific to this plan

- **Terminal-Bench container cost/time:** each `evaluate()` call in P3 spins up a fresh Docker container per task in the batch; GEPA's iterative reflection loop multiplies this. Budget for it explicitly (small batches, capped GEPA iterations) before scaling the pinned task list.
- **Small-sample overfitting in the lite gate:** a ~15–20 task pinned subset, split further into optimization vs. held-out batches per skill, leaves very few tasks per skill. M3's "v2 beats v1 on a fresh batch" result should be read as a smoke test, not evidence of real generalization — that's exactly what full P4 (§6 gap) exists to fix later.
- **`ContainerToolRunner` security surface:** `docker exec`-ing model-generated bash into a container is still executing untrusted output; confirm Terminal-Bench's container isolation (network egress, resource limits) is sufficient for this use before pointing it at anything beyond the pinned task containers.
- **GEPA cost accounting:** `gepa_runner.py` must pass explicit `max_metric_calls`/`max_reflection_cost` to `gepa.optimize()` per skill (native caps, not something to reconstruct after the fact) and log `GEPAResult.total_evals` alongside the run — if evolving one skill costs more than the task-completion gains it produces are worth, that's a §7 (spec) cost-metric regression the system should be able to see, not just a hidden expense.

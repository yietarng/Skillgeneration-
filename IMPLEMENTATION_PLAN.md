# Implementation Plan: P1 → P2 → P3 → P4 → P5 → Orchestrator → CLI

Status: converts `PROJECT_SPEC.md` into a sequenced, buildable plan; **P1-P5 all built, tested, and documented**, §10 ties them into a single driver, and §11 gives that driver a real command-line entrypoint against actual Terminal-Bench task execution. What remains is not a phase but running the whole thing continuously against real infrastructure: a real Terminal-Bench task corpus (§1's pinned list is still unpicked) and sustained access to Docker + a live LLM key. See §9/§10/§11's own verification notes for exactly what is and isn't confirmed.
Scope for this plan: **P1 (Trajectory Segmentation and Extraction) → P2 (Skill-Library Maintenance) → P3 (Weight-Free Skill Evolution via GEPA) → P4 (Safe Validation and Regression Control) → P5 (Skill Retrieval, Activation, and Adaptation) → an orchestrator (§10) tying all five into one loop → a CLI (§11) driving that orchestrator against real task execution**.

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

**No auto-write to `active`:** every `CREATE`/`REVISE`/`MERGE`/`SPECIALIZE` output from `Librarian` is written with `status="candidate"`, never directly overwriting a `status="active"` skill. Promotion to `active` goes through P4's real gate (§6: `validation/promotion.py`'s `evaluate_promotion()`, which is what actually calls `SkillLibrary.promote()`) — this is the one place P2 depends on something from P3/P4 rather than the reverse.

**Acceptance criteria (M2):**
- Feed drafts from 2+ Terminal-Bench traces that hit the *same* underlying pattern (e.g., two different tasks both requiring a similar diagnostic-then-fix procedure): `Librarian` proposes `MERGE` or `REVISE`, not two independent `CREATE`s — verified by hand on this constructed pair before trusting it on the full pinned set.
- Feed a draft with no existing overlap: `Librarian` proposes `CREATE` and it's `status="candidate"` on disk (`SKILL.md` + `metadata.json`) but the library's `active` set is unchanged until P4's gate (§6) runs.
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
  promotion_gate.py           # cheap structural/edit-size/contradiction pre-filter, run before
                                # the real P4 gate (§6) spends a full in-domain + regression suite
                                # run on a candidate — see §7
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
- **Substitution, not a gap:** "one skill from the pinned set" used a fake `run_task_fn` and fake `reflection_lm` instead of real Terminal-Bench tasks and a real LLM, because neither Docker nor a usable API key was available in the environment that built this (same constraint noted in §1). The GEPA *mechanics* (candidate proposal → evaluation → acceptance → Pareto tracking → promotion gate) are genuinely verified; the *quality* of real reflective proposals and real task scoring is not — that needs an environment with both, per §9's risks.

## 6. Phase P4 — Safe Validation and Regression Control

**Status: built (`validation/`) and verified against real P1→P2→P3 output** — `tests/test_p3_to_p4_integration.py` runs an actual `gepa.optimize()` result through `SkillLibrary.revise()` and into `evaluate_promotion()`, which genuinely promotes it (flips `status: candidate → active`, demotes the incumbent to `deprecated`). This closes a real gap the P1-P3 plan left open: P3's `promotion_gate.py` (§5) only ever returned a pass/fail `GateResult` — nothing in P1-P3 ever actually wrote a promotion to storage. `SkillLibrary` gained the two methods that make that real:

- `promote(skill_id, version)` — flips a version to `active`, demotes any other currently-active version of the same `skill_id` to `deprecated` (kept on disk, not archived, so it's the target `rollback()` finds).
- `rollback(skill_id)` — swaps the active version back to the most recent `deprecated` one, demoting the current active in turn. A swap, not a delete: rolling back a rollback un-does it.

**New module:** `validation/`
```
validation/
  __init__.py
  eval_runner.py    # run_suite(skill, task_ids, run_task_fn, suite) -> list[EvaluationRecord],
                      # aggregate_score()/aggregate_cost(). Reuses evolution.adapter's
                      # skill_to_candidate/candidate_to_skill_fields encoding so a skill scored
                      # here and one scored inside P3's evaluate() see byte-identical injected
                      # text -- no drift between "how P3 scored a candidate" and "how P4 re-scores
                      # it". Never raises for a single task's failure, same contract as P3.
  regression.py       # check_regression(candidate, baseline, task_ids, run_task_fn, epsilon,
                        # cost_tolerance) -- runs BOTH the candidate and the current active
                        # version (the baseline) against a fixed cross-domain task set; fails on
                        # either a score drop beyond epsilon or a per-cost-key (input_tokens,
                        # output_tokens, tool_calls) increase beyond cost_tolerance. The spec's
                        # "a skill that improves success by making the agent try much harder is a
                        # regression on cost" is a literal per-key check, not a single blended score.
  promotion.py          # evaluate_promotion(library, skill_id, candidate_version,
                          # in_domain_task_ids, regression_task_ids, run_task_fn, margin,
                          # first_promotion_floor, epsilon, cost_tolerance) -> PromotionDecision.
                          # Runs the in-domain held-out check (candidate >= predecessor + margin)
                          # THEN the regression check, in that order (cheaper check first);
                          # library.promote() only on both passing. On rejection: the failing
                          # EvaluationRecords and a rejection note are attached to the candidate's
                          # own provenance via a new SkillLibrary.save() (metadata.json only,
                          # doesn't bump version) -- visible on disk for the next GEPA batch or a
                          # human, not just returned and discarded.
  shadow.py              # ShadowLedger: per-skill_id@version JSON files of (task_id, score)
                          # observations, is_promotion_eligible(min_observations, min_success_rate)
                          # -- the concrete proxy this plan uses for spec §5.4's "N tasks or until
                          # statistical confidence" (a real confidence-interval computation would
                          # tighten the threshold, not change the shape of the check). Gates
                          # whether evaluate_promotion() is even worth running, not a promotion
                          # decision itself.
  rollback.py             # rollback_if_regressed(library, skill_id, live_success_rate,
                           # baseline_success_rate, observed_count, min_observations, epsilon) --
                           # calls library.rollback() when LIVE (not offline-suite) monitoring
                           # detects a regression, gated by its own observation-count floor so a
                           # rollback isn't decided on a handful of unlucky tasks.
```

**First-promotion policy (a design decision the spec doesn't fully pin down):** the spec's promotion rule is phrased relative to "the predecessor," but a skill's very first candidate version has no predecessor to regress against. `evaluate_promotion()` handles this as a distinct branch: no active version yet → skip the regression suite entirely (nothing to compare) and require the candidate clear an absolute `first_promotion_floor` on the in-domain set instead of the relative `margin`. Kept as two separate parameters on purpose — one is a floor, one is a margin over a moving baseline, and conflating them would silently change semantics depending on whether a skill happens to have a predecessor yet.

**Acceptance, verified with fakes (no Docker/LLM in this environment — same constraint as §1/§5):**
- `tests/test_storage.py`: `promote()`/`rollback()` correctly flip status, demote in the right direction, and are swap-reversible.
- `tests/test_regression.py`: fails on a score drop beyond epsilon, fails on a per-cost-key blowup beyond tolerance, tolerates small variation within both.
- `tests/test_promotion.py`: first-promotion floor, margin-based rejection, and — the one that actually matters most — rejection **on regression even when the in-domain score improved**, proving the two gates are independent, not just the first one gating a rubber-stamped second check.
- `tests/test_shadow.py` / `tests/test_rollback.py`: eligibility thresholds and rollback's observation-count floor.
- `tests/test_p3_to_p4_integration.py`: the full chain, end to end, using the same fake-`reflection_lm` + fake-`run_task_fn` substitution documented in §5 — genuinely exercises the P1(schema)→P2(storage)→P3(evolve)→P4(promote) pipeline's wiring, not real-world validation quality.

## 7. Phase P5 — Skill Retrieval, Activation, and Adaptation

**Status: built (`retrieval/`) and verified against real P1-P4 output** — `tests/test_p5_integration.py` confirms `build_injection()`'s output actually reaches `TraceCollector`'s real `system_prompt_addendum` param (the injection point P3 added), not just that the pipeline's internal types line up. This retires `evolution/skill_injection.py`'s hardcoded single-skill stub as the production path — that module still exists and still does its own narrower job (P3's `evaluate()` deliberately injects one *fixed* candidate under test, bypassing retrieval entirely, which is correct for optimization).

**New module:** `retrieval/`
```
retrieval/
  __init__.py
  retriever.py     # retrieve(library, index, query_text, repo_context, k, similarity_weight,
                     # track_record_weight) -> list[RetrievalCandidate]. Reuses
                     # skill_library.index.EmbeddingIndex directly -- the exact artifact the spec
                     # anticipated P2's dedup index becoming. Restricted to status="active" skills.
                     # Blends embedding similarity with track_record_score() (average score across a
                     # skill's provenance.validation_evidence; 0.5 neutral prior with no evidence
                     # yet) plus a lexical-overlap fallback for exact-string matches. Lazy-loading is
                     # structural, not bolted on: the index only ever stores vectors + skill_id, so
                     # library.read() (full content) is called only for the top-k, not the library.
  activation.py      # ApplicabilityChecker(dspy.Module): verifies a candidate's prerequisites
                       # actually hold for this task/repo, not just topical similarity. Cached per
                       # (repo, task_type, skill_id, version). order_by_precedence(): deterministic
                       # multi-skill ordering when more than one candidate activates.
  adapter.py           # Adapter(dspy.Module): rewrites a skill's generic procedure into task-
                         # concrete guidance. Ephemeral -- never written back to the library. Not to
                         # be confused with evolution/adapter.py's SkillGEPAAdapter (unrelated).
  safety_gate.py         # check_adapted_text(): pattern-based guard against obviously destructive
                           # literal commands (rm -rf /, fork bombs, curl-pipe-to-shell, raw-disk
                           # writes) an adaptation might introduce. Not a sandbox -- a last check
                           # before adapted text reaches the agent's context, independent of and in
                           # addition to P3/P4's publish-time gates.
  injection.py            # build_injection(): the real thing evolution/skill_injection.py stood in
                            # for -- runs retrieve() -> activation -> order_by_precedence() ->
                            # adapt() -> safety gate -> render, capped by max_injected and a
                            # character-based token-budget proxy (max_total_chars). None
                            # system_prompt_addendum is a valid, expected outcome at any stage
                            # (nothing retrieved, nothing activated, everything safety-blocked) --
                            # never forces a weak match.
  bundles.py               # BundleStore (Hermes Agent's skill_bundles.py pattern): pre-declared
                             # named skill_id sets for known-good co-activating combinations, as a
                             # compactness alternative to per-task dynamic precedence.
  attribution.py             # attribute_credit(): splits outcome credit across co-active skills by
                               # lexical overlap between the trace's actual tool-call text and each
                               # skill's adapted procedure -- an explicit heuristic (no ground truth
                               # exists for "which skill's guidance was actually followed"),
                               # documented as such, in the same spirit as skill_library.index's
                               # hashing embedding.
```

**A real design bug the tests caught:** the first `retriever.py` draft floored a lexical hit onto the *final blended score* (`max(blended, 0.8)`), which meant a skill with an exact text match to the query could keep outranking a skill with a demonstrated 0% track record forever, no matter how bad its record got. Fixed to floor the *similarity component* before blending instead, so `track_record_weight` can still pull a lexically-exact-but-proven-bad skill back down — `tests/test_retriever.py::test_track_record_can_override_an_exact_lexical_match` pins this down.

**Two intentional approximations, not gaps hiding as gaps:**
- `order_by_precedence`'s "failure_recovery-oriented before procedure-oriented" ordering (spec §5.5) needs a skill's *originating* kind, but `Skill` doesn't persist the `Segment.kind` it came from — only the trace-level `Segment` does. Approximated from content shape (more `failure_recovery` entries than `prerequisites` reads as failure_recovery-oriented) rather than adding a schema field for one ordering heuristic.
- `attribution.attribute_credit`'s lexical-overlap credit split is explicitly a heuristic standing in for a real "did the agent follow this guidance" signal that doesn't exist. Both are documented in their own module docstrings, not just here.

**Acceptance, verified with fakes (`DummyLM` for every `dspy.Module`; no Docker/LLM, same constraint as §1/§5/§6):**
- `tests/test_retriever.py`: empty-library and non-active-skill exclusion, task_type filtering, ranking, and the track-record-override case above.
- `tests/test_activation.py`: caching (including cache-scoping per repo), and both precedence rules independently.
- `tests/test_retrieval_adapter.py`: adaptation rewrites only `procedure`, leaves `activation`/`verification`/`failure_recovery` untouched.
- `tests/test_safety_gate.py`: blocks the obvious destructive patterns, allows benign and scoped-destructive (`rm -rf ./build`) commands.
- `tests/test_bundles.py`, `tests/test_attribution.py`: storage round-trip / subset matching; credit favors the skill whose wording a trace's tool calls actually resemble, splits evenly on zero overlap.
- `tests/test_injection.py`: the full chain — happy path, empty retrieval, failed activation, safety-blocked adaptation, and the `max_injected` cap actually respecting precedence order — five paths, not just the happy one.
- `tests/test_p5_integration.py`: `build_injection()`'s result reaching `TraceCollector.system_prompt` for real, and a `None` result leaving the prompt byte-identical to the unassisted baseline.

## 8. Sequencing and dependencies

```
P1 (extract_skills)
   │  candidate drafts
   ▼
P2 (Librarian, storage, index)
   │  status="candidate" skills on disk; index also seeded for P5's reuse
   ▼
P3 (gepa_runner, promotion_gate)  ── uses evolution/skill_injection.py's fixed-candidate
   │  winning candidate, structurally sound + contradiction-free    injection to score itself
   ▼
P4 (validation/promotion.py, regression.py, shadow.py)  ── needs P2's promote()/rollback()
   │  status="active" skill (real gate: in-domain held-out + cross-domain regression + cost)
   ▼
P5 (retriever, activation, adapter, injection)  ── needs P2's active-status skills + index,
   │  system_prompt_addendum for a real task            and P4 to have produced any to retrieve
   ▼
(closed loop: TraceCollector runs with P5's injection → new trace →
 back to P1, tagged with which skill(s)/version were active via injection.render_addendum;
 attribution.attribute_credit() splits outcome credit across co-active skills;
 live monitoring can call validation/rollback.py independent of this offline loop)
```

Build strictly in this order — P2's `Librarian` needs real candidate drafts from P1 to have anything to decide on, P3's `evaluate()` needs P2's storage format to read/write skill versions, P4's `evaluate_promotion()` needs both P2's `promote()`/`active_version()` and P3's winning candidate to have something to gate, and P5's `retrieve()` needs P4 to have actually promoted something to `active` before there's anything in the library worth retrieving.

## 9. Risks specific to this plan

- **Terminal-Bench container cost/time:** each `evaluate()` call in P3 spins up a fresh Docker container per task in the batch; GEPA's iterative reflection loop multiplies this. Budget for it explicitly (small batches, capped GEPA iterations) before scaling the pinned task list.
- **Small-sample overfitting:** a ~15–20 task pinned subset, split further into GEPA's optimization batch, P4's in-domain held-out set, and P4's regression suite, leaves very few tasks per skill in each bucket. §6's `validation/promotion.py` now runs the real two-gate check the spec calls for, which is a genuine improvement over M3's original single-batch smoke test — but a real regression *suite* with only a handful of tasks in it is still a weak instrument for catching a rare cross-domain regression; more tasks per skill matters more than a fancier check on few of them. This is a data problem §1's still-unpicked pinned task list needs to solve, not something §6's code can fix on its own.
- **`ContainerToolRunner` security surface:** `docker exec`-ing model-generated bash into a container is still executing untrusted output; confirm Terminal-Bench's container isolation (network egress, resource limits) is sufficient for this use before pointing it at anything beyond the pinned task containers.
- **GEPA cost accounting:** `gepa_runner.py` must pass explicit `max_metric_calls`/`max_reflection_cost` to `gepa.optimize()` per skill (native caps, not something to reconstruct after the fact) and log `GEPAResult.total_evals` alongside the run — if evolving one skill costs more than the task-completion gains it produces are worth, that's a §7 (spec) cost-metric regression the system should be able to see, not just a hidden expense.
- **P5's retrieval quality is only as good as the hashing embedding it inherited from P2:** a bag-of-words hashing vectorizer can't capture true semantic similarity beyond shared vocabulary — a task phrased differently from a skill's `activation` text but semantically the same case will score low on the similarity component regardless of `track_record_weight`. Fine for proving the retrieval/ranking/activation/adaptation *mechanics* work (§7's tests do exactly that); a real embedding provider is a drop-in `embed_fn` swap (`skill_library/index.py`) whenever retrieval quality against real tasks needs to be assessed, not a design change.
- **`safety_gate.check_adapted_text`'s pattern list is necessarily incomplete:** it catches the obviously destructive literal commands this project's authors thought of, not an exhaustive taxonomy. It is a last check before injection, not a substitute for the actual execution sandbox (`tool_handlers.py`/`container_tool_handlers.py`) — treat a pattern-list gap as expected, not as a reason to trust adapted text more than the sandbox already does.
- **The orchestrator (§10) is tested with fakes, not a real continuously-running task stream.** `orchestrator/driver.py` now calls every phase in the right order and its branching logic is verified — but "verified the wiring and the decision logic" is not the same claim as "ran against a real, continuous task stream and improved over time." Populating §1's still-unpicked pinned Terminal-Bench task list and pointing the orchestrator's injected callables (`collect_trace_fn`, `gepa_run_task_fn`, `reflection_lm`, `promotion_run_task_fn`) at real infrastructure is what stands between "every phase and their sequencing work" (true, as of this revision) and "the system actually runs and improves over time in production" (not attempted, and not attemptable in an environment without Docker or a live LLM key).

## 10. Orchestrator: tying P1-P5 into one loop

**Status: built (`orchestrator/driver.py`) and verified** — 6/6 tests passing, including one fully real `gepa.optimize()` run through the orchestrator's own `run_evolution_cycle()` function (not called directly, the way `tests/test_p3_to_p4_integration.py` did it), proving the sequencing itself is correct, not just that each phase works in isolation.

**Two entrypoints, matching the spec's own split between live and offline cadence:**

- **`Orchestrator.handle_task(task_description, collect_trace_fn, repo_context=None) -> TaskResult`** — the live path, once per real task: `retrieval.build_injection()` (P5, active skills only) → `collect_trace_fn(task_description, system_prompt_addendum)` (caller-supplied — wraps `TraceCollector.run()` against whatever tool runner/workdir, or `tbench_adapter.run_tbench_task`) → `retrieval.attribution.attribute_credit()` on the resulting trace → `dspy_modules.p1_extraction.extract_skills()` (P1) → `skill_library.librarian.Librarian` + `apply_decision()` per draft (P2). Returns a `TaskResult` bundling the trace, the injection result, the credit split, the extracted drafts, and each draft's `(decision, resulting Skill | None)` — callers observe everything the cycle did, not just a final status.
- **`run_evolution_cycle(library, index, skill_id, ...) -> EvolutionCycleResult`** — the offline path, invoked periodically (by a scheduler, not per task, per spec §5.2's "auxiliary session" cadence): `evolution.gepa_runner.evolve_skill()` (P3) → `evolution.promotion_gate.promote()` (the lite structural/contradiction check) → `SkillLibrary.revise()` to store the winning candidate → `validation.promotion.evaluate_promotion()` (P4's real gate). Stops at whichever stage doesn't clear, returning a `status` of `"no_improvement"`, `"lite_gate_rejected"`, `"promotion_rejected"`, or `"promoted"` — every outcome is a legitimate, inspectable result, not an exception.

**Every external effect stays injected**, the same discipline every other phase in this plan followed: `collect_trace_fn`, `gepa_run_task_fn`, `reflection_lm`, and `promotion_run_task_fn` are all caller-supplied callables. The orchestrator itself never constructs a `TraceCollector`, never calls a live LLM, and never touches Docker — which is exactly what makes it testable with fakes here and swappable for real infrastructure elsewhere without changing a line of orchestration logic.

**Acceptance, verified with fakes (`DummyLM` + fake callables; no Docker/LLM, same constraint as every other phase):**
- `tests/test_orchestrator.py::test_handle_task_with_empty_library_extracts_and_creates` / `test_handle_task_injects_active_skill_and_attributes_credit` — the live path with and without an existing skill to retrieve, confirming the injected guidance actually reaches `collect_trace_fn` (mirroring `tests/test_p5_integration.py`'s check on `TraceCollector` directly) and that a single injected skill gets full attribution credit.
- `tests/test_orchestrator.py::test_run_evolution_cycle_promotes_via_real_gepa_optimize` — the full offline path through a real `gepa.optimize()` call, ending in an actual promotion.
- `tests/test_orchestrator.py::test_run_evolution_cycle_no_improvement_on_flat_scoring_landscape` — also a real `gepa.optimize()` call, against a deliberately flat-scoring task fn where "no improvement found" is the true, inevitable outcome, not a forced one.
- `tests/test_orchestrator.py::test_run_evolution_cycle_lite_gate_rejected` / `test_run_evolution_cycle_promotion_rejected` — these two use `monkeypatch` on `evolve_skill()` rather than a real GEPA search, because reliably forcing a real optimizer to land on a specific rejected candidate isn't practical; what's being tested here is `run_evolution_cycle`'s own branching (does a lite-gate rejection correctly avoid ever calling `evaluate_promotion`; does a promotion rejection correctly leave the library's active version unchanged while still keeping the rejected candidate on disk for inspection), not GEPA's search behavior, which is already covered by the two real runs above.

**Deliberately not built:** live shadow-traffic mirroring — silently running an unpromoted candidate alongside the active skill on a slice of real tasks to feed `validation.shadow.ShadowLedger` with live (not offline-suite) evidence. That ledger already exists from P4 and could be wired to a future canary-traffic feature; the mirroring mechanism itself would add real complexity (a second agent run per task, careful isolation so the shadow run can't affect the real one) without having been asked for, so it's flagged here rather than half-built.

## 11. Orchestrator CLI

**Status: built (`orchestrator/cli.py`) and verified against the real `terminal-bench` package and a real task fixture** — not just import-checked. `handle-task` and `evolve` wrap `Orchestrator.handle_task()`/`run_evolution_cycle()` (§10) with real Terminal-Bench task execution (`trace_collection.tbench_adapter.run_tbench_task`) as their `collect_trace_fn`/`run_task_fn`.

```
python -m orchestrator.cli handle-task \
    --library-dir ./skill_library_data --tasks-dir ./terminal-bench/original-tasks \
    --task-id acl-permissions-inheritance

python -m orchestrator.cli evolve \
    --library-dir ./skill_library_data --tasks-dir ./terminal-bench/original-tasks \
    --skill-id bash-missing-flag \
    --gepa-train task-a,task-b,task-c --gepa-val task-d,task-e \
    --in-domain task-f,task-g --regression task-h,task-i \
    --reflection-model anthropic/claude-opus-5
```

**Two separate model flags, not one, on purpose:** `--model` is Anthropic-SDK-style (`"claude-opus-5"`, `trace_collection.collector`'s convention — what `TraceCollector`/`run_tbench_task` need) while `--reflection-model` is litellm-style (`"anthropic/claude-opus-5"`, what `gepa.optimize()`'s `reflection_lm` expects). `dspy_modules.lm_config`'s own model (used by `extract_skills`/`Librarian`/`ApplicabilityChecker`/`Adapter`) isn't exposed as a third flag at all — it already reads `SKILLGEN_DSPY_MODEL` from the environment when an override is needed. Three model-ish flags in three string formats would only multiply the ways to get this wrong; the CLI's own docstring spells out why each one is scoped the way it is.

**`rebuild_index(library) -> EmbeddingIndex`:** fills a real gap this CLI's existence exposed — `skill_library.index.EmbeddingIndex` is in-memory only, so every fresh CLI invocation needs to reconstruct it from the library's `active`-status skills before retrieval can find anything. `main()` calls this once at startup for both subcommands.

**Two real bugs this build caught, fixed at the source rather than worked around:**
- `trace_collection/tbench_adapter.py`'s `run_tbench_task()` had no `system_prompt_addendum` parameter at all — meaning nothing built through §5-§10 could actually have injected retrieval/evolution guidance into a real Terminal-Bench run; every prior integration test used the local `SandboxedToolRunner` path or a fake `collect_trace_fn`/`run_task_fn`, so this gap was invisible until the CLI needed to wire a *real* execution path end to end. Fixed by threading the parameter through to `TraceCollector`.
- `evolution/skill_injection.py`'s `build_system_prompt_addendum()` didn't tag its output with `skill_id@version` the way `retrieval/injection.py`'s `render_addendum()` does — a real inconsistency between the P3/P4 injection path and the P5 one, caught by a test asserting the tag was present. Fixed at the source (both paths now tag consistently), not by changing the test's expectation to match the weaker behavior.

**Verified how:** since `orchestrator/cli.py`'s tbench-backed functions lazily import `trace_collection.tbench_adapter` (itself a hard dependency on `terminal_bench`, Python ≥3.12), they can't even be *called* in this repo's normal Python 3.11 dev/test environment — only imported (the lazy-import pattern keeps the module itself loadable). Verified for real in an isolated Python 3.12 venv with `terminal-bench` actually installed: `load_task_instruction()` against a real `task.yaml` (now committed as `tests/fixtures/tbench_tasks/sample-task/`, so this doesn't depend on an external clone), the two closures against a monkeypatched `run_tbench_task` (avoiding a need for live Docker while still exercising real code), and a full `main()` dispatch with only the two genuinely external calls (LM config, `run_tbench_task`) faked out — everything else in that run was real orchestration code.

**Acceptance:**
- `tests/test_orchestrator_cli.py` — 8 tests need nothing beyond this repo's normal environment (`argparse` structure, `rebuild_index()` against a real `SkillLibrary`/`EmbeddingIndex`) and always run; 3 more (`load_task_instruction`, both `make_tbench_*_fn` closures) are guarded with `pytest.importorskip("terminal_bench")` per-test (not at module level — an earlier draft put the skip at module level and it silently skipped every test in the file, including the ones needing no such thing) so they skip cleanly here and run for real in a proper 3.12+terminal-bench environment. All 11 pass in that environment, confirmed while building this.

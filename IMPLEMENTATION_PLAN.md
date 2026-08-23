# Implementation Plan: P1 → P2 → P3

Status: draft v0.1 · converts `PROJECT_SPEC.md` into a sequenced, buildable plan
Scope for this plan: **P1 (Trajectory Segmentation and Extraction) → P2 (Skill-Library Maintenance) → P3 (Weight-Free Skill Evolution via GEPA)**. P4 (full validation/regression harness) and P5 (retrieval/activation/adaptation) are **not** built out here beyond the minimal stubs P3 needs to close its own loop — see §6.

## 1. Trace corpus: Terminal-Bench

Trace collection needs real, verifiably-scored tasks to be worth anything to P1/P3. [Terminal-Bench](https://www.tbench.ai) is the source for this plan:

- Each task ships `instruction.md` (the task prompt), a `Dockerfile`/container environment, a reference solution, and a **pytest-based verification suite that checks final container state** (files, outputs, command effects) rather than the transcript.
- That verification suite *is* `Trace.outcome_signal` (§4.1 of the spec): `{"kind": "test", "score": pass_fraction, "detail": pytest_output}` — no LLM-judge fallback needed for this corpus.
- The Docker environment supplies `repo_context` (task domain: sysadmin, security, ML, scientific computing, etc., per Terminal-Bench's task taxonomy) for free.

**Integration change required:** `trace_collection/tool_handlers.py`'s `SandboxedToolRunner` currently runs bash via local `subprocess`, confined to a directory. Terminal-Bench tasks run inside per-task Docker containers. Add a sibling runner:

```
trace_collection/
  tool_handlers.py         # unchanged: SandboxedToolRunner (local dir) — kept for fast dev iteration
  container_tool_handlers.py   # NEW: ContainerToolRunner — same interface, execs into a running
                                 # Terminal-Bench task container (`docker exec`) instead of subprocess
  tbench_adapter.py            # NEW: given a Terminal-Bench task id, starts its container, runs
                                 # TraceCollector.run() against it via ContainerToolRunner, tears the
                                 # container down, and runs the task's pytest verification to fill
                                 # Trace.outcome_signal before saving
```

`collector.py`'s `TraceCollector` itself does not need to change — it already takes a tool runner and a task string; `tbench_adapter.py` just supplies a `ContainerToolRunner` and a task pulled from Terminal-Bench instead of an arbitrary CLI string.

**Pinned subset, not the full suite:** start with a fixed, versioned list of ~15–20 Terminal-Bench task ids spanning 2–3 domains (e.g., a handful of sysadmin + a handful of scripting/debugging tasks) rather than all ~89. Reasons: (a) each task is a fresh Docker container — full-suite runs are slow and costly to iterate on; (b) P1/P2 need *repeated* structurally-similar tasks to prove segmentation/dedup work, which a hand-picked, domain-clustered subset gives more reliably than a random full sweep. Record the pinned list in `eval/tbench_task_ids.txt` so later phases (P4's real held-out/regression split) can extend it without ambiguity about what's already been seen.

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

**Why the standalone `gepa` package, not `dspy.GEPA`:** `dspy.GEPA` is a teleprompter that optimizes the prompts of a `dspy.Module` pipeline it can call directly. The system actually being evaluated here is `TraceCollector`'s manual agentic loop against the raw Anthropic API (bash + text-editor tools) — not a DSPy pipeline. GEPA's underlying mechanism (`GEPAAdapter` with `evaluate()` + `make_reflective_dataset()`) is designed exactly for this: "optimize a textual component of *any* system," where the system is opaque to GEPA beyond the adapter. Use that directly.

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

**`SkillGEPAAdapter.evaluate(candidate, batch)`:**
1. `candidate` is the skill's optimizable text bundle: `{procedure, prerequisites, failure_recovery}` (§5.3 of the spec — `activation`, `verification`, and `provenance` are not GEPA-optimized text).
2. For each task in `batch` (drawn from the pinned Terminal-Bench subset that this skill's `activation` covers): inject the candidate text via `skill_injection.py`, run `TraceCollector.run()` against a fresh container for that task, collect the resulting `Trace`.
3. Run the task's pytest verification → `outcome_signal` → `feedback.py` → `(score, feedback_text)`.
4. Return scores (for GEPA's Pareto selection) and the raw traces (`capture_traces=True`) for `make_reflective_dataset`.

**`SkillGEPAAdapter.make_reflective_dataset(...)`:** surfaces, per example, what the injected skill said vs. what the agent actually did vs. what failed — this is what GEPA's reflective LM reads to propose the next candidate's `procedure`/`failure_recovery` text. This is the direct implementation of the spec's "convert heterogeneous execution feedback into controlled textual revisions."

**Drift/contradiction controls (§5.3 of the spec), concretely:**
- `promotion_gate.py` rejects a GEPA-proposed candidate outright if the diff against the previous version exceeds a changed-lines ratio threshold, or if any of the six required fields becomes empty/near-empty.
- A cheap LLM-judge call (reuses `dspy_modules/signatures.py`, new `CheckContradiction` signature) compares old vs. new `procedure`/`failure_recovery` text; a flagged contradiction blocks auto-promotion and requires the candidate to be written to disk as `status="candidate"` for manual review instead.

**Acceptance criteria (M3):**
- End-to-end demonstration on one skill from the pinned set: inject v1, run its covered tasks, observe some failures, run `gepa_runner.py`, get a v2 candidate, pass it through `promotion_gate.py`, and show v2 scores higher than v1 on a **fresh** batch of the same skill's covered tasks (held out from the batch GEPA optimized against, even within this lite loop — no held-out split at all would make M3 unfalsifiable).
- `promotion_gate.py` demonstrably rejects at least one deliberately-bad synthetic candidate (e.g., one with a blanked-out `failure_recovery` field) in a test, proving the gate isn't a no-op.

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
- **GEPA cost accounting:** `gepa_runner.py` should log its own token/call cost per skill optimization run — if evolving one skill costs more than the task-completion gains it produces are worth, that's a §7 (spec) cost-metric regression the system should be able to see, not just a hidden expense.

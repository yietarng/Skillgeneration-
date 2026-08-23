# Project Specification: Weight-Free Skill Generation for Coding Agents

Status: draft v0.1 · revises the raw problem statement into an implementable spec
Base stack: Python, [DSPy](https://github.com/stanfordnlp/dspy), [GEPA](https://github.com/gepa-ai/gepa); reference architecture: [Hermes Agent](https://github.com/nousresearch/hermes-agent)

## 0. Note on scope of this revision

The source problem statement's P5 heading ("Skill Retrieval, Activation, and Adaptatio...") was truncated before its body arrived. Section 5.5 below reconstructs P5 from the stated objective ("retrieve and adapt these skills for future tasks") and from its role as the closing stage of the P1–P5 pipeline (retrieval must feed from the library P2 maintains and the skills P1/P3 produce, and hand adapted guidance to the executing agent). Confirm or correct that reconstruction before treating P5 as final.

## 1. Objective

Coding agents discover useful procedures, diagnostics, tool-use conventions, and recovery methods while executing tasks, but this knowledge is normally lost at session end. Raw trajectories are unsuitable for reuse: too long, too repo-specific, too noisy, and contaminated with unsuccessful or accidental decisions.

This project builds a system that:

1. Extracts compact, transferable **skills** from raw execution trajectories.
2. Maintains a **skill library** that stays curated, not just append-only.
3. **Evolves** skill instructions from execution feedback (scores, errors, test failures, evaluator comments) without fine-tuning the underlying LLM.
4. **Validates** every change before it can affect production behavior, with regression control.
5. **Retrieves, activates, and adapts** skills for new tasks.

Success is measured against a baseline (no-skill) agent on: task-completion rate, test-pass rate, repeated-error rate, tool-call count, context-token count, and human-correction count.

## 2. Non-goals

- No fine-tuning or weight updates to the backend LLM at any stage. All improvement is textual (skill content, retrieval index, prompts).
- Not backend-model-specific: the architecture must work with any DSPy-compatible LM, though the current trace collector happens to target the Anthropic API directly.
- Not a replacement for existing static skill formats (e.g., Claude Code's `SKILL.md`) — this system should be able to *emit* into that format, not invent a new one gratuitously.

## 3. Pipeline overview

```
 Agent execution
       │
       ▼
 ┌─────────────┐   P1    ┌──────────────┐   P2    ┌───────────────┐
 │   Trace     │ ──────► │ Segmentation  │ ──────► │  Skill Library │
 │ Collection  │         │ & Extraction  │         │  Maintenance   │
 └─────────────┘         └──────────────┘         └───────┬────────┘
       ▲                                                    │
       │                                            P3      ▼
       │                                        ┌────────────────────┐
       │                                        │  Weight-Free Skill  │
       │                                        │  Evolution (GEPA)   │
       │                                        └─────────┬──────────┘
       │                                                    │
       │                                            P4      ▼
       │                                        ┌────────────────────┐
       │                                        │  Validation &       │
       │                                        │  Regression Control │
       │                                        └─────────┬──────────┘
       │                                                    │ (promoted skills)
       │                                            P5      ▼
       │                                        ┌────────────────────┐
       └─────────────────────────────────────── │ Retrieval,          │
         (new trace from skill-augmented run)    │ Activation,         │
                                                  │ Adaptation          │
                                                  └────────────────────┘
```

Everything downstream of trace collection runs **offline / asynchronously** relative to the agent's live task loop, except P5's retrieval + adaptation step, which must be low-latency enough to run inline before/at the start of a task.

## 4. Core data model

### 4.1 Trace (extends current `trace_collection/schema.py`)

The existing `Trace`/`StepRecord`/`ToolCallRecord` dataclasses already capture per-turn API responses, tool calls, and token usage. Extend with:

| Field | Type | Purpose |
|---|---|---|
| `outcome_signal` | `{"kind": "test"\|"exit_code"\|"llm_judge"\|"human", "score": float, "detail": str}` | Ground-truth-ish success signal used as P3/P4 feedback, not just `outcome` string |
| `segments` | `list[Segment]` | Filled in by P1, not at collection time |
| `repo_context` | `{"repo": str, "language": str, "task_type": str}` | Used for applicability metadata in P5 |

### 4.2 Segment (new)

```python
class Segment:
    trace_id: str
    turn_range: tuple[int, int]
    kind: Literal["plan", "procedure", "tool_convention", "failure_recovery", "noise"]
    abstraction_level: Literal["episode_specific", "task_class", "overgeneral"]  # P1 judgment
    rationale: str          # why this segment was selected / classified this way
    is_successful_branch: bool
```

### 4.3 Skill (the unit P2–P5 operate on)

Every skill has exactly six required fields, per the problem statement — this is the schema contract, not a style suggestion:

```python
class Skill:
    skill_id: str
    version: int
    name: str
    activation: str          # when this should trigger — retrieval + applicability check use this
    prerequisites: list[str] # applicability conditions; a failed check blocks activation
    procedure: str           # the reusable procedure / decision points
    failure_recovery: list[str]  # common failures and recovery strategies
    verification: str        # verification and termination criteria
    provenance: Provenance
    status: Literal["candidate", "active", "deprecated", "retired"]
```

```python
class Provenance:
    source_trace_ids: list[str]
    created_at: str
    revised_from: str | None        # previous skill_id/version, if a revision
    validation_evidence: list[EvaluationRecord]
    limitations: str                # explicitly stated scope boundaries / known failure modes
```

### 4.4 EvaluationRecord (P4 output, feeds P2/P3 decisions)

```python
class EvaluationRecord:
    skill_id: str
    skill_version: int
    suite: Literal["in_domain_heldout", "system_regression"]
    task_id: str
    score: float
    cost: dict            # tokens, tool_calls, wall_time
    feedback_text: str    # raw evaluator/compiler/test output — GEPA's reflection input
```

## 5. Component specs

### 5.1 P1 — Trajectory Segmentation and Extraction

**DSPy modules:**

- `Segmenter(dspy.Module)` — given a trace's turn sequence (tool calls, outcomes, plan text), proposes segment boundaries and classifies each into `plan | procedure | tool_convention | failure_recovery | noise`. Boundary heuristics seed the search: sub-goal changes in stated plans, error→recovery pairs (tool call with `is_error=True` followed by a differing retry that succeeds), and verification/termination points.
- `AbstractionJudge(dspy.Module)` — scores a candidate segment's abstraction level. Rejects `noise` and flags `overgeneral` (no actionable specifics) vs `episode_specific` (unreplaced literal paths/names/values) so the `Abstractor` knows what to fix.
- `Abstractor(dspy.Module)` — rewrites an accepted segment into a **candidate skill draft** populating all six required fields, generalizing literal values (paths, exact commands, repo names) into parameterized patterns while preserving the decision logic.

**Handling unsuccessful branches:** a `failure_recovery` segment is extracted *because* it failed and then recovered — the failed sub-branch is the point. `is_successful_branch=False` segments are not discarded; they are the primary source for the `failure_recovery` field. A trace that ends in overall failure can still yield a valid skill if it contains a genuine recovery, distinguished from segments that are simply dead ends (no recovery, no signal) via `AbstractionJudge`.

**Output:** zero or more candidate skill drafts per trace, each with `status="candidate"` and `provenance.source_trace_ids=[trace_id]`, handed to P2.

### 5.2 P2 — Skill-Library Maintenance Without Bloat

**Storage:** directory-of-`SKILL.md` (compatible with the Claude Code skill format) + a parallel `metadata.json`/vector index per skill, mirroring `trace_collection/skill_generator.py`'s existing `write_skill()` output shape but adding the structured metadata needed for retrieval and lifecycle decisions.

**`Librarian(dspy.Module)`:** given a candidate skill and its top-k nearest existing skills (embedding similarity over `activation` + `prerequisites`, filtered by `repo_context`/`task_type` metadata), decides one action:

| Action | Trigger condition |
|---|---|
| `CREATE` | No existing skill covers this activation condition / task class |
| `REVISE` | Same skill, new evidence — refines procedure/recovery/prerequisites in place, version bump |
| `MERGE` | Two+ skills cover overlapping conditions with compatible procedures — consolidate, retire the smaller |
| `SPECIALIZE` | An existing general skill's procedure fails in a repo/language-specific way that recurs — fork a scoped variant, keep the general skill for other contexts |
| `REJECT` | Duplicate with no new information, or fails a minimum-utility bar (e.g., only ever solved a single episode, no generalizable decision point) |
| `RETIRE` | Existing skill's rolling success rate degrades below threshold, or it's superseded by a merge/specialization and its unique coverage is now zero |

**Compactness controls:**
- Hard library size budget; `RETIRE` is triggered by a utilization × success-rate score falling out of the bottom percentile when the budget is exceeded (not just LRU).
- Semantic dedup gate before `CREATE`: embedding similarity threshold, then an LLM-judge tie-breaker for borderline cases (`MERGE` vs `CREATE`).
- Every accepted `REVISE`/`MERGE`/`SPECIALIZE` must pass P4 validation before it replaces what's live (see 5.4) — the Librarian proposes, it doesn't deploy.

### 5.3 P3 — Weight-Free Skill Evolution (GEPA)

Skill text fields (`procedure`, `failure_recovery`, `prerequisites`) are treated as **optimizable text components** in a DSPy program, evolved by [GEPA](https://github.com/gepa-ai/gepa) rather than by gradient updates — this is the direct translation of "weight-free" into an existing, sample-efficient method built for exactly this: reflective mutation of text components from textual + scalar feedback, with Pareto-frontier candidate selection across multiple objectives.

**Feedback conversion:** a `feedback.py` module normalizes heterogeneous signals — compiler errors, failed test names/messages, tool `is_error` outputs, evaluator free-text comments — into the `(score, feedback_text)` pairs GEPA's reflection step consumes. This is the concrete answer to "convert heterogeneous execution feedback into controlled textual revisions": GEPA's reflective LM proposes a new candidate skill text conditioned on `feedback_text` for a batch of recent evaluations, not on scores alone.

**Drift/contradiction controls (GEPA gives sample efficiency, not safety — these are added on top):**
- Bounded edit size per generation (diff against previous version capped, e.g., by a max changed-lines ratio) to prevent wholesale rewrites from one bad batch.
- Structural validator: every GEPA-proposed candidate must still populate all six required fields non-trivially, or it's rejected before reaching P4.
- Contradiction check: an LLM-judge module compares new vs. previous version and flags direct contradictions (e.g., a recovery step that now says the opposite of what it said before) for human review rather than silent replacement.
- Run offline in batches over accumulated `EvaluationRecord`s per skill (or per skill cluster), not per-task online — evolution proposes new *candidate* versions; it never mutates the active skill directly (P4 gates that).

### 5.4 P4 — Safe Validation and Regression Control

Every skill revision produced by P2 (`REVISE`/`MERGE`/`SPECIALIZE`) or P3 (GEPA candidate) is `status="candidate"` until it clears both:

1. **In-domain held-out set:** a task batch of the same class the skill targets, disjoint from the batch that produced the revision. Must score `>= predecessor + margin` on task-completion/test-pass rate.
2. **System-level regression suite:** a fixed, versioned, cross-domain task sample run with the candidate installed in the library. Must not regress below `baseline - epsilon` on the aggregate metrics (§7), including cost metrics (tokens, tool calls) — a skill that "improves" success by making the agent try much harder is a regression on cost.

**Promotion rule:** replace the predecessor only if both gates pass. Otherwise: keep the predecessor `active`, keep the candidate for inspection (or discard if it also underperforms the reject bar), and log the failure back as an `EvaluationRecord` so P3's next GEPA batch sees *why* it failed.

**Rollout mechanics:** candidates run in **shadow mode** first — logged against live tasks but not authoritative — for N tasks or until statistical confidence is reached on the in-domain metric, before flipping `status: candidate → active`. Keep the immediate predecessor pinned and retrievable for fast rollback if production monitoring (not just the offline suites) detects regression.

### 5.5 P5 — Skill Retrieval, Activation, and Adaptation

- **Retrieval:** hybrid search over the `active`-status library only — embedding similarity on `activation` + `prerequisites` text (precomputed index, updated on promotion/retire), combined with metadata filters (`repo_context`, `task_type`, language) and a lexical (BM25) fallback for exact tool/error-string matches that embeddings miss. Candidates are re-ranked by blending similarity with each skill's rolling track record from `EvaluationRecord` (success rate, cost) — similarity alone can rank an unproven or historically weak skill above a validated one. Returns a ranked candidate set, not a single skill; an empty or below-threshold result set is a valid outcome, not an error — the task proceeds with the unassisted baseline agent rather than forcing a weak match into context.
- **Activation:** an `ApplicabilityChecker(dspy.Module)` evaluates each retrieved candidate's `prerequisites` against the current task + repo state before it's allowed into context — retrieval similarity alone is not sufficient permission to activate. Verdicts are cached per `(repo_context, skill_id, skill_version)` so this check isn't repaid in full on every task. Failed checks are logged (useful signal for P1/P2: a skill that's retrieved often but rarely passes activation may need narrower `prerequisites` or a `SPECIALIZE` split). When multiple skills pass activation for one task, apply an explicit precedence order (e.g., `failure_recovery`-triggered skills after `plan`/`procedure`-triggered ones, narrower `prerequisites` before broader) rather than injecting an unordered set — conflicting procedures from co-activated skills must resolve deterministically, not by injection order.
- **Adaptation:** an `Adapter(dspy.Module)` rewrites an activated skill's generic `procedure` into task/repo-concrete guidance (actual paths, actual tool names, actual test commands) for injection into the agent's context. Adaptation output is **ephemeral** — it is not written back into the library; only genuinely new generalizable knowledge re-enters the library, via a fresh trace through P1. Injected guidance is **advisory, not binding**: the agent may deviate from it, and a deviation is itself signal (worth capturing, not suppressing) that the skill's `procedure` or `prerequisites` may need revision.
- **Runtime safety gate:** clearing P4's offline validation does not guarantee an adapted, repo-concrete instruction is safe to execute in *this* live repo (e.g., a shell command that was inert against the sandbox that validated it but is destructive here). Apply a lightweight guard at injection time — independent of and in addition to the publish-time gate — before adapted procedure text reaches the tool-execution loop.
- **Injection:** adapted skill(s) are added to the agent's system/context (same mechanism `SKILL.md` injection already uses), tagged with `skill_id@version` so downstream provenance is unambiguous, under a token budget and a cap on the number of concurrently injected skills, so this step cannot itself become the token-cost regression P4 is supposed to catch.
- **Closed loop:** the resulting task trace is collected exactly like any other (§4.1's `provenance` records which skill(s) were active, at which version), and feeds back into P1 — this is how the system observes whether adapted guidance actually helped, independent of the offline P4 suites. When multiple skills were co-active on one task, attribute outcome credit per skill (e.g., by which skill's guidance the accepted actions actually followed, per the deviation signal above) rather than crediting all co-active skills equally — this attribution is what makes §7's retrieval-hit-rate and activation-pass-rate metrics meaningful per skill instead of only in aggregate.

## 6. Why DSPy + GEPA specifically

- **DSPy** gives every module above (`Segmenter`, `AbstractionJudge`, `Abstractor`, `Librarian`, `ApplicabilityChecker`, `Adapter`) a declarative `Signature` instead of a hand-tuned prompt string, and composes them as `dspy.Module`s against a swappable backend LM — this is what makes "different backend models" (stated requirement) a configuration change, not a rewrite.
- **GEPA** is the P3 engine specifically because it optimizes *text* components from *reflective, textual* feedback (not just scalar reward), with Pareto-based multi-objective candidate selection — which maps directly onto "improve from heterogeneous feedback without weight updates" and onto needing to balance task success against cost/regression rather than a single scalar. It's invoked as an offline batch optimizer per skill (or skill cluster) over accumulated `EvaluationRecord`s, gated by P4 before anything it proposes goes live — GEPA proposes, P4 disposes.
- **Hermes Agent** is a reference for agentic tool-use loop structure and system-prompt conventions; the current `trace_collection/collector.py` already implements a comparable manual agentic loop directly against the Anthropic API (bash + text-editor tools) — reuse that as the execution substrate rather than re-deriving one.

## 7. Metrics

Measured for a skill-augmented agent vs. a no-skill baseline, on both the in-domain held-out set and the regression suite:

- Task-completion rate / test-pass rate
- Repeated-error rate (same failure signature recurring within or across tasks)
- Tool-call count per task
- Context tokens consumed per task
- Human-correction count (where available)
- Library health (secondary): size over time, retrieval hit-rate, activation pass-rate, retire/create ratio — signals for whether P2's bloat control is working

## 8. Relationship to the existing codebase

| Existing file | Current role | Target role |
|---|---|---|
| `trace_collection/schema.py` | `Trace`/`StepRecord`/`ToolCallRecord` | Base for §4.1; add `outcome_signal`, `segments`, `repo_context` |
| `trace_collection/collector.py` | Runs one task against the real Anthropic API, sandboxed bash + text-editor tools | Keep as-is as the execution substrate; add outcome-signal capture (test/exit-code hook) at trace close |
| `trace_collection/tool_handlers.py` | Sandboxed bash/text-editor execution | Reused unchanged; also becomes the execution substrate for P4's held-out/regression task runs |
| `trace_collection/skill_generator.py` | Single LLM call, trace → full `SKILL.md`, no library, no validation | Splits into P1's `Segmenter`/`AbstractionJudge`/`Abstractor` DSPy modules; loses direct-to-`SKILL.md` write, which becomes P2's job post-validation |
| `trace_collection/cli.py` | `collect`, `generate-skill` | Extend with `librarian`, `evolve`, `validate`, `retrieve` subcommands as those modules land |

**Not yet present, needed for P2–P5:** `skill_library/` (storage + index + `Librarian`), `evolution/` (GEPA runner + feedback conversion), `validation/` (held-out + regression suites, promotion logic), `retrieval/` (retriever + `ApplicabilityChecker` + `Adapter`), and a shared `dspy_modules/` (or similar) home for the `Signature`/`Module` definitions used across those.

## 9. Proposed target layout

```
trace_collection/     # existing — extended per §8
skill_library/         # P2: storage.py, index.py, librarian.py
evolution/              # P3: gepa_runner.py, feedback.py, metric.py
validation/             # P4: eval_sets/, regression_runner.py, promotion.py
retrieval/              # P5: retriever.py, activation.py, adapter.py
dspy_modules/           # shared Signatures/Modules used by the above
eval/benchmarks/        # versioned in-domain + regression task suites
```

## 10. Phased roadmap

| Phase | Delivers | Depends on |
|---|---|---|
| 0 (done) | Raw trace collection + naive single-shot skill distillation | — |
| 1 | Structured `Skill`/`Segment` schema; P1 as DSPy modules; batch trace ingestion | Phase 0 |
| 2 | `skill_library/` storage + P5 basic retrieval (no adaptation yet) + `Librarian` dedup/create/reject | Phase 1 |
| 3 | `validation/` harness: held-out + regression suites, promotion gate — nothing reaches `active` without it | Phase 2 |
| 4 | `evolution/`: GEPA loop over accumulated `EvaluationRecord`s, feeding `REVISE` candidates through Phase 3's gate | Phase 3 |
| 5 | Full closed loop: `ApplicabilityChecker` + `Adapter` (rest of P5), production shadow rollout, metrics dashboard | Phase 2–4 |

## 11. Open questions / risks

- **Ground-truth signal availability:** not every task has tests; need an LLM-judge fallback for `outcome_signal`, which itself needs calibration against real test-backed tasks to avoid judge drift.
- **GEPA loop cost:** reflective evolution adds LLM calls beyond task execution; needs a batching/cadence policy (e.g., weekly per active skill cluster) rather than per-trace.
- **Domain boundary definition:** in-domain held-out vs. system regression only works if "domain" (task class) is well-defined per skill — ill-defined domains make P4's overfitting check meaningless.
- **General skill vs. repo-specific fork:** the `SPECIALIZE` action needs a concrete threshold (how many repo-specific failures before forking) to avoid fragmenting the library.
- **Sandbox parity between collection and validation:** P4's held-out/regression runs must execute in the same sandboxing model as `tool_handlers.py` to be a fair comparison — container/VM isolation, not just directory confinement, for anything beyond trusted internal use.

## 12. Skill markdown template (six required fields)

```markdown
---
skill_id: <slug>
version: <int>
status: candidate|active|deprecated|retired
---

# <Skill name>

## Activation
<when this should trigger>

## Prerequisites
- <applicability condition>

## Procedure
<reusable procedure and decision points>

## Failure Recovery
- <common failure> → <recovery strategy>

## Verification & Termination
<how to know the skill's goal was met / when to stop>

## Provenance
- Source traces: <ids>
- Revised from: <previous skill_id/version or none>
- Validation evidence: <in-domain + regression results summary>
- Limitations: <explicit scope boundaries, known failure modes>
```

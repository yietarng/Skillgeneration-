"""The top-level driver tying P1-P5 into one loop -- the gap
IMPLEMENTATION_PLAN.md §9 flagged: every phase is built and tested against
its immediate neighbor, but nothing before this called them in sequence
against a live task stream.

Two entrypoints, matching the spec's own separation between fast, per-task
live behavior and slow, batched maintenance:

- handle_task(): the live path. One real task: retrieve+inject (P5,
  active skills only) -> run the agent -> attribute credit -> extract
  candidate skills (P1) -> Librarian decisions (P2). Runs once per task.
- run_evolution_cycle(): the offline path. Evolve one skill via GEPA (P3),
  gate the result through the lite structural/contradiction check and
  then P4's real promotion gate. Invoked periodically by a scheduler, not
  per task -- spec §5.2's "auxiliary session" cadence, not live-task
  cadence.

Every external effect (running the agent, running a GEPA/validation task,
calling a reflection LM) is injected as a callable, matching every other
module in this project. This is real orchestration logic, testable with
fakes/DummyLM, not something that only works against live infrastructure.

Not built here: live shadow-traffic mirroring (silently running an
unpromoted candidate alongside the active skill on a slice of real
traffic to feed validation.shadow.ShadowLedger). That ledger already
exists from P4 and could be wired to a future canary-traffic feature;
adding the mirroring mechanism itself was judged out of scope for "tie
the phases together" and would add real complexity without having been
asked for.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

from dspy_modules.p1_extraction import SkillDraft, extract_skills
from evolution.adapter import RunTaskFn
from evolution.gepa_runner import EvolutionResult, evolve_skill
from evolution.promotion_gate import ContradictionChecker
from evolution.promotion_gate import promote as run_lite_gate
from evolution.skill_injection import apply_candidate_fields
from retrieval.attribution import AttributedCredit, attribute_credit
from retrieval.injection import InjectionResult, build_injection
from skill_library.index import EmbeddingIndex
from skill_library.librarian import Librarian, LibrarianDecision, apply_decision
from skill_library.storage import Skill, SkillLibrary
from trace_collection.schema import Trace
from validation.promotion import PromotionDecision, evaluate_promotion

# (task_description, system_prompt_addendum | None) -> Trace. Wraps
# whatever actually executes the agent -- TraceCollector.run() against a
# local sandbox, or tbench_adapter.run_tbench_task's container path. The
# caller owns constructing TraceCollector (workdir/tool_runner choice);
# the driver only needs the resulting Trace.
CollectTraceFn = Callable[[str, "str | None"], Trace]


@dataclasses.dataclass
class TaskResult:
    trace: Trace
    injection: InjectionResult
    credits: list[AttributedCredit]
    drafts: list[SkillDraft]
    decisions: list[tuple[SkillDraft, LibrarianDecision, "Skill | None"]]


class Orchestrator:
    """Holds the shared, long-lived state (library + index) that both
    entrypoints read and write. One instance per running system, not per
    task."""

    def __init__(self, library: SkillLibrary, index: EmbeddingIndex, librarian: Librarian | None = None):
        self.library = library
        self.index = index
        self.librarian = librarian or Librarian(library, index)

    def handle_task(
        self,
        task_description: str,
        collect_trace_fn: CollectTraceFn,
        repo_context: dict | None = None,
    ) -> TaskResult:
        """One live task, start to finish. Requires an LM already
        configured for extract_skills()/self.librarian (dspy_modules.
        lm_config.configure_lm() or a DummyLM for tests) -- this method
        doesn't configure one itself, same contract as extract_skills()."""
        injection = build_injection(self.library, self.index, task_description, repo_context=repo_context)

        trace = collect_trace_fn(task_description, injection.system_prompt_addendum)
        # Prefer whatever collect_trace_fn already determined (e.g.
        # tbench_adapter knows the real task's category) over the caller's
        # possibly-generic repo_context.
        trace.repo_context = trace.repo_context or repo_context

        credits = attribute_credit(trace, injection.injected)

        drafts = extract_skills(trace)

        decisions: list[tuple[SkillDraft, LibrarianDecision, "Skill | None"]] = []
        for draft in drafts:
            decision = self.librarian(draft, repo_context)
            skill = apply_decision(self.library, self.index, draft, decision, repo_context)
            decisions.append((draft, decision, skill))

        return TaskResult(trace=trace, injection=injection, credits=credits, drafts=drafts, decisions=decisions)


@dataclasses.dataclass
class EvolutionCycleResult:
    skill_id: str
    # "no_improvement" (GEPA's best didn't beat the seed on valset) |
    # "lite_gate_rejected" | "promoted" | "promotion_rejected"
    status: str
    reason: str
    evolution: EvolutionResult | None = None
    candidate_skill: Skill | None = None
    promotion: PromotionDecision | None = None


def run_evolution_cycle(
    library: SkillLibrary,
    index: EmbeddingIndex,
    skill_id: str,
    gepa_trainset: list[str],
    gepa_valset: list[str],
    gepa_run_task_fn: RunTaskFn,
    reflection_lm: Any,
    in_domain_task_ids: list[str],
    regression_task_ids: list[str],
    promotion_run_task_fn: RunTaskFn,
    contradiction_checker: ContradictionChecker | None = None,
    max_metric_calls: int | None = None,
    max_reflection_cost: float | None = None,
    margin: float = 0.0,
    first_promotion_floor: float = 0.5,
    epsilon: float = 0.05,
    cost_tolerance: float = 0.25,
    **gepa_optimize_kwargs: Any,
) -> EvolutionCycleResult:
    """The offline maintenance cycle for one skill: P3 evolve -> lite gate
    (structural/edit-size/contradiction) -> stored as a new candidate
    version -> P4's real promotion gate. Stops at whichever stage rejects
    -- "GEPA found nothing better," "lite gate rejected," and "promotion
    gate rejected" are all legitimate, logged outcomes, not exceptions.
    """
    seed_skill = library.read(skill_id)

    evolution_result = evolve_skill(
        skill=seed_skill,
        trainset=gepa_trainset,
        valset=gepa_valset,
        run_task_fn=gepa_run_task_fn,
        reflection_lm=reflection_lm,
        max_metric_calls=max_metric_calls,
        max_reflection_cost=max_reflection_cost,
        **gepa_optimize_kwargs,
    )

    if evolution_result.val_score <= evolution_result.seed_val_score:
        return EvolutionCycleResult(
            skill_id, "no_improvement", "GEPA found no candidate better than the seed on valset", evolution_result
        )

    updated = apply_candidate_fields(seed_skill, evolution_result.candidate_skill_fields)
    lite_gate = run_lite_gate(previous=seed_skill, candidate=updated, contradiction_checker=contradiction_checker)
    if not lite_gate.passed:
        return EvolutionCycleResult(
            skill_id, "lite_gate_rejected", lite_gate.reason, evolution_result
        )

    candidate_draft = SkillDraft(
        activation=updated.activation,
        prerequisites=updated.prerequisites,
        procedure=updated.procedure,
        failure_recovery=updated.failure_recovery,
        verification=updated.verification,
        source_trace_ids=["gepa_evolution"],
    )
    candidate_skill = library.revise(skill_id, candidate_draft)

    promotion = evaluate_promotion(
        library,
        skill_id,
        candidate_skill.version,
        in_domain_task_ids,
        regression_task_ids,
        promotion_run_task_fn,
        margin=margin,
        first_promotion_floor=first_promotion_floor,
        epsilon=epsilon,
        cost_tolerance=cost_tolerance,
    )

    return EvolutionCycleResult(
        skill_id,
        "promoted" if promotion.promoted else "promotion_rejected",
        promotion.reason,
        evolution_result,
        candidate_skill,
        promotion,
    )

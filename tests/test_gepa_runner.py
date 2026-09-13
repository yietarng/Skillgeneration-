"""Runs a REAL gepa.optimize() (not mocked) with a fake, deterministic
reflection_lm and a fake, deterministic run_task_fn -- no live LLM or
Docker needed, but this exercises actual GEPA candidate proposal,
acceptance, and Pareto/best-candidate tracking, not just our own adapter
code in isolation. This is what caught two real bugs while building P3:
SkillGEPAAdapter needing an explicit propose_new_texts = None (GEPA
accesses it directly, not via getattr with a default) and GEPAResult's
total_evals not existing on the actually-installed gepa==0.1.4 (it's on
GitHub's main branch, not yet released) -- see gepa_runner.py's
_total_evals().
"""

from __future__ import annotations

import dspy
from dspy.utils import DummyLM

from evolution.gepa_runner import evolve_skill
from evolution.promotion_gate import promote
from evolution.skill_injection import apply_candidate_fields
from skill_library.storage import Provenance, Skill
from trace_collection.schema import Trace

_MARKER = "FIXED"


def _skill(**overrides) -> Skill:
    fields = dict(
        skill_id="bash-missing-flag",
        version=1,
        name="Bash command fails on missing flag",
        activation="Bash command fails citing a missing flag.",
        prerequisites=["A bash tool is available"],
        procedure="OLD: do nothing special.",
        failure_recovery=["OLD: give up"],
        verification="Retried command exits zero.",
        provenance=Provenance(source_trace_ids=["trace_1"], created_at="now"),
    )
    fields.update(overrides)
    return Skill(**fields)


def _run_task_fn(task_id: str, skill_fields: dict) -> Trace:
    """Deterministic stand-in for a real Terminal-Bench run: scores 1.0 iff
    the marker text made it into the candidate, 0.0 otherwise. No LLM, no
    Docker -- this is what run_task_fn would normally do via
    tbench_adapter.run_tbench_task with the candidate injected."""
    text = " ".join(
        [skill_fields["procedure"], *skill_fields["prerequisites"], *skill_fields["failure_recovery"]]
    )
    score = 1.0 if _MARKER in text else 0.0
    return Trace(
        trace_id=f"trace_{task_id}",
        task=task_id,
        model="m",
        workdir="/w",
        started_at="now",
        outcome="success" if score >= 1.0 else "max_turns",
        outcome_signal={"kind": "test", "score": score, "detail": f"deterministic fixture for {task_id}"},
    )


def _fake_reflection_lm(prompt) -> str:
    """A real reflection LM would read the feedback and propose a fix; this
    fake always proposes the same corrected text regardless of prompt
    content -- sufficient to prove GEPA's proposal/acceptance/Pareto
    machinery actually improves the seed, without needing a live model."""
    return f"```\n{_MARKER}: run the command, add the missing flag on failure, and retry once.\n```"


def test_evolve_skill_improves_on_the_seed_using_real_gepa_optimize():
    seed = _skill()

    result = evolve_skill(
        skill=seed,
        trainset=["t1", "t2", "t3"],
        valset=["v1", "v2"],
        run_task_fn=_run_task_fn,
        reflection_lm=_fake_reflection_lm,
        max_metric_calls=60,
        display_progress_bar=False,
    )

    assert result.seed_val_score == 0.0  # seed never contains the marker
    assert result.val_score > result.seed_val_score
    assert result.val_score == 1.0
    assert _MARKER in result.candidate_skill_fields["procedure"]
    assert result.total_evals > 0


def test_evolve_skill_result_flows_through_skill_injection_and_promotion_gate():
    """The whole P3 chain: gepa_runner's winning candidate -> a concrete
    Skill via skill_injection -> promotion_gate's structural/edit-size/
    contradiction checks."""
    seed = _skill()

    result = evolve_skill(
        skill=seed,
        trainset=["t1", "t2", "t3"],
        valset=["v1", "v2"],
        run_task_fn=_run_task_fn,
        reflection_lm=_fake_reflection_lm,
        max_metric_calls=60,
        display_progress_bar=False,
    )

    candidate_skill = apply_candidate_fields(seed, result.candidate_skill_fields)
    assert candidate_skill.skill_id == seed.skill_id  # identity preserved, only text changed
    assert candidate_skill.activation == seed.activation  # not GEPA-optimized

    dspy.configure(
        lm=DummyLM(
            [
                {
                    "reasoning": "The revision replaces a no-op procedure with a concrete one; no conflict.",
                    "has_contradiction": False,
                    "explanation": "",
                }
            ]
        )
    )
    gate_result = promote(previous=seed, candidate=candidate_skill)
    assert gate_result.passed is True

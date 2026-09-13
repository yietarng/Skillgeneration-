"""orchestrator.driver tests.

test_handle_task_* cover the live path (P5 -> agent -> P1 -> P2) with
DummyLM and a fake collect_trace_fn -- no live agent needed.

test_run_evolution_cycle_promotes_via_real_gepa_optimize is one fully
real gepa.optimize() run (mirrors tests/test_p3_to_p4_integration.py),
proving the "promoted" path end to end through the orchestrator's own
function, not just through evolve_skill/evaluate_promotion directly.

The no_improvement branch is also driven through a real gepa.optimize()
call (a flat-scoring run_task_fn makes "no improvement found" the true,
inevitable outcome). lite_gate_rejected and promotion_rejected are tested
via monkeypatched evolve_skill() results -- forcing those exact outcomes
through a real GEPA search isn't reliably controllable, and it's this
function's branching logic being tested, not GEPA's search behavior.
"""

from __future__ import annotations

import dspy
from dspy.utils import DummyLM

from dspy_modules.p1_extraction import SkillDraft
from evolution.gepa_runner import EvolutionResult
from orchestrator.driver import Orchestrator, run_evolution_cycle
from skill_library.index import EmbeddingIndex
from skill_library.storage import SkillLibrary
from trace_collection.schema import StepRecord, ToolCallRecord, Trace


def _draft(activation: str = "Bash command fails on missing flag.", **overrides) -> SkillDraft:
    fields = dict(
        activation=activation, prerequisites=["A bash tool is available"],
        procedure="proc", failure_recovery=["r"], verification="v", source_trace_ids=["t"],
    )
    fields.update(overrides)
    return SkillDraft(**fields)


def _sample_trace() -> Trace:
    trace = Trace(
        trace_id="trace_orchestrator001", task="Run the linter and fix any issues",
        model="claude-opus-5", workdir="/sandbox", started_at="2026-01-01T00:00:00Z", outcome="success",
    )
    trace.steps = [
        StepRecord(
            turn=0, timestamp="t0", stop_reason="tool_use", usage={}, response={"content": []},
            tool_calls=[
                ToolCallRecord(tool_use_id="tu0", name="bash", input={"command": "ruff check ."},
                                output="error: missing --fix flag", is_error=True, duration_ms=5.0)
            ],
        ),
        StepRecord(
            turn=1, timestamp="t1", stop_reason="tool_use", usage={}, response={"content": []},
            tool_calls=[
                ToolCallRecord(tool_use_id="tu1", name="bash", input={"command": "ruff check . --fix"},
                                output="All checks passed!", is_error=False, duration_ms=5.0)
            ],
        ),
    ]
    return trace


_P1_ANSWERS = [
    {
        "reasoning": "One failure/recovery segment across the two turns.",
        "segments": [
            {
                "start_turn": 0, "end_turn": 1, "kind": "failure_recovery",
                "rationale": "bash command failed citing a missing flag, retried with the flag added",
                "is_successful_branch": False,
            }
        ],
    },
    {"reasoning": "Concrete, generalizable; no leftover literals.", "keep": True, "abstraction_level": "task_class", "issues": []},
    {
        "reasoning": "Generalizing the retried bash invocation.",
        "activation": "Bash command fails citing a missing flag.",
        "prerequisites": ["A bash tool is available"],
        "procedure": "1. Run the command. 2. Add the flag on failure and retry once.",
        "failure_recovery": ["missing flag -> add it and retry once"],
        "verification": "Retried command exits zero.",
    },
    {"reasoning": "Draft is clean.", "keep": True, "abstraction_level": "task_class", "issues": []},
]


# -- handle_task -------------------------------------------------------


def test_handle_task_with_empty_library_extracts_and_creates(tmp_path):
    library = SkillLibrary(tmp_path)
    index = EmbeddingIndex()
    orchestrator = Orchestrator(library, index)

    dspy.configure(
        lm=DummyLM(
            _P1_ANSWERS
            + [{"reasoning": "no existing skill covers this", "action": "CREATE", "target_skill_id": "", "rationale": "novel"}]
        )
    )

    collected = []

    def collect_trace_fn(task_description, system_prompt_addendum):
        collected.append((task_description, system_prompt_addendum))
        return _sample_trace()

    result = orchestrator.handle_task("Run the linter and fix any issues", collect_trace_fn)

    assert collected == [("Run the linter and fix any issues", None)]
    assert result.injection.system_prompt_addendum is None
    assert result.credits == []
    assert len(result.drafts) == 1
    assert len(result.decisions) == 1
    draft, decision, skill = result.decisions[0]
    assert decision.action == "CREATE"
    assert skill is not None
    assert library.exists(skill.skill_id)


def test_handle_task_injects_active_skill_and_attributes_credit(tmp_path):
    library = SkillLibrary(tmp_path)
    index = EmbeddingIndex()
    task_description = "Run the linter and fix any issues"

    library.create(_draft(task_description), skill_id="lint-fix")
    library.promote("lint-fix", 1)
    seeded = library.read("lint-fix")
    index.upsert("lint-fix", f"{seeded.activation} {' '.join(seeded.prerequisites)}")

    orchestrator = Orchestrator(library, index)

    dspy.configure(
        lm=DummyLM(
            [
                {"reasoning": "task names a bash command", "applies": True, "reason": "confirmed"},
                {"reasoning": "adapting", "adapted_procedure": "adapted guidance for this exact task"},
            ]
            + _P1_ANSWERS
            + [
                {
                    "reasoning": "sharper evidence for the same skill",
                    "action": "REVISE", "target_skill_id": "lint-fix", "rationale": "same underlying skill",
                }
            ]
        )
    )

    def collect_trace_fn(task_description, system_prompt_addendum):
        assert system_prompt_addendum is not None
        assert "lint-fix@1" in system_prompt_addendum
        assert "adapted guidance for this exact task" in system_prompt_addendum
        return _sample_trace()

    result = orchestrator.handle_task(task_description, collect_trace_fn)

    assert result.injection.system_prompt_addendum is not None
    assert len(result.injection.injected) == 1
    assert len(result.credits) == 1
    assert result.credits[0].skill_id == "lint-fix"
    assert result.credits[0].credit == 1.0

    draft, decision, skill = result.decisions[0]
    assert decision.action == "REVISE"
    assert skill.skill_id == "lint-fix"
    assert skill.version == 2


# -- run_evolution_cycle -------------------------------------------------

_MARKER = "FIXED"


def _marker_run_task_fn(task_id: str, skill_fields: dict) -> Trace:
    text = " ".join([skill_fields["procedure"], *skill_fields["prerequisites"], *skill_fields["failure_recovery"]])
    score = 1.0 if _MARKER in text else 0.0
    return Trace(
        trace_id=f"trace_{task_id}", task=task_id, model="m", workdir="/w", started_at="now",
        outcome="success" if score >= 1.0 else "max_turns",
        outcome_signal={"kind": "test", "score": score, "detail": task_id},
    )


def _fixing_reflection_lm(prompt) -> str:
    return f"```\n{_MARKER}: run the command, add the missing flag on failure, and retry once.\n```"


def test_run_evolution_cycle_promotes_via_real_gepa_optimize(tmp_path):
    library = SkillLibrary(tmp_path)
    index = EmbeddingIndex()
    library.create(
        _draft("Bash command fails citing a missing flag.", procedure="OLD: do nothing special.",
               failure_recovery=["OLD: give up"]),
        skill_id="bash-missing-flag",
    )
    library.promote("bash-missing-flag", 1)

    dspy.configure(
        lm=DummyLM(
            [
                {
                    "reasoning": "Replaces a no-op procedure with a concrete one; no conflict.",
                    "has_contradiction": False, "explanation": "",
                }
            ]
        )
    )

    result = run_evolution_cycle(
        library, index, "bash-missing-flag",
        gepa_trainset=["t1", "t2", "t3"], gepa_valset=["v1", "v2"],
        gepa_run_task_fn=_marker_run_task_fn, reflection_lm=_fixing_reflection_lm,
        in_domain_task_ids=["in-1", "in-2"], regression_task_ids=["reg-1", "reg-2"],
        promotion_run_task_fn=_marker_run_task_fn,
        max_metric_calls=60, margin=0.1, display_progress_bar=False,
    )

    assert result.status == "promoted"
    assert result.candidate_skill.version == 2
    assert library.active_version("bash-missing-flag") == 2
    assert library.read("bash-missing-flag", version=1).status == "deprecated"


def test_run_evolution_cycle_no_improvement_on_flat_scoring_landscape(tmp_path):
    library = SkillLibrary(tmp_path)
    index = EmbeddingIndex()
    library.create(_draft("Bash command fails.", procedure="OLD proc"), skill_id="s1")
    library.promote("s1", 1)

    def flat_run_task_fn(task_id, skill_fields):
        return Trace(
            trace_id=f"t_{task_id}", task=task_id, model="m", workdir="/w", started_at="now",
            outcome="success", outcome_signal={"kind": "test", "score": 0.5, "detail": ""},
        )

    result = run_evolution_cycle(
        library, index, "s1",
        gepa_trainset=["t1", "t2"], gepa_valset=["v1"],
        gepa_run_task_fn=flat_run_task_fn, reflection_lm=lambda prompt: "```\nsomething else\n```",
        in_domain_task_ids=["i1"], regression_task_ids=["r1"], promotion_run_task_fn=flat_run_task_fn,
        max_metric_calls=20, display_progress_bar=False,
    )

    assert result.status == "no_improvement"
    assert result.candidate_skill is None
    assert library.active_version("s1") == 1  # unchanged


def test_run_evolution_cycle_lite_gate_rejected(tmp_path, monkeypatch):
    library = SkillLibrary(tmp_path)
    index = EmbeddingIndex()
    library.create(
        _draft("Bash command fails.", procedure="OLD proc", failure_recovery=["OLD recovery"]), skill_id="s1"
    )
    library.promote("s1", 1)

    fake_result = EvolutionResult(
        candidate_skill_fields={"procedure": "", "prerequisites": ["p"], "failure_recovery": ["r"]},  # empty -> fails check_structure
        val_score=1.0, seed_val_score=0.0, total_evals=5, gepa_result=None,
    )
    monkeypatch.setattr("orchestrator.driver.evolve_skill", lambda **kwargs: fake_result)

    result = run_evolution_cycle(
        library, index, "s1",
        gepa_trainset=["t1"], gepa_valset=["v1"],
        gepa_run_task_fn=lambda task_id, fields: None, reflection_lm=lambda prompt: "",
        in_domain_task_ids=["i1"], regression_task_ids=["r1"], promotion_run_task_fn=lambda task_id, fields: None,
    )

    assert result.status == "lite_gate_rejected"
    assert "procedure is empty" in result.reason
    assert result.candidate_skill is None
    assert library.active_version("s1") == 1  # unchanged, and no v2 was ever written
    assert library.list_versions("s1") == [1]


def test_run_evolution_cycle_promotion_rejected(tmp_path, monkeypatch):
    library = SkillLibrary(tmp_path)
    index = EmbeddingIndex()
    library.create(
        _draft("Bash command fails.", procedure="OLD proc", failure_recovery=["OLD recovery"]), skill_id="s1"
    )
    library.promote("s1", 1)

    # Small enough edit to clear the lite gate's edit-size cap (one line
    # unchanged), but the candidate text itself scores worse below.
    fake_result = EvolutionResult(
        candidate_skill_fields={
            "procedure": "OLD proc extended with a new step",
            "prerequisites": ["p"],
            "failure_recovery": ["OLD recovery"],
        },
        val_score=1.0, seed_val_score=0.0, total_evals=5, gepa_result=None,
    )
    monkeypatch.setattr("orchestrator.driver.evolve_skill", lambda **kwargs: fake_result)

    def scoring_run_task_fn(task_id, skill_fields):
        score = 1.0 if skill_fields["procedure"] == "OLD proc" else 0.0  # exact match only
        return Trace(
            trace_id=f"t_{task_id}", task=task_id, model="m", workdir="/w", started_at="now",
            outcome="success" if score >= 1.0 else "max_turns",
            outcome_signal={"kind": "test", "score": score, "detail": ""},
        )

    dspy.configure(lm=DummyLM([{"reasoning": "no conflict", "has_contradiction": False, "explanation": ""}]))

    result = run_evolution_cycle(
        library, index, "s1",
        gepa_trainset=["t1"], gepa_valset=["v1"],
        gepa_run_task_fn=lambda task_id, fields: None, reflection_lm=lambda prompt: "",
        in_domain_task_ids=["i1"], regression_task_ids=["r1"], promotion_run_task_fn=scoring_run_task_fn,
    )

    assert result.status == "promotion_rejected"
    assert library.active_version("s1") == 1  # unchanged
    assert result.candidate_skill is not None  # v2 was written as a candidate...
    assert result.candidate_skill.version == 2
    assert library.read("s1", version=2).status == "candidate"  # ...but never promoted

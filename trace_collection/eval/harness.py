"""Minimal evaluation harness measuring the effect of environment-probing
curation, analogous in spirit to the paper's CLBench comparison: same
scenarios run with probing off vs on, scoring whether the generated skill
states the grounded fact instead of repeating the trace's stale/wrong claim,
and counting API calls per condition as a cheap proxy for the paper's
per-question cost/query numbers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import anthropic

from ..skill_generator import DEFAULT_MODEL, generate_skill
from .fixtures import ALL_SCENARIOS, Scenario


class _CountingMessages:
    def __init__(self, messages, counter: "_CallCounter"):
        self._messages = messages
        self._counter = counter

    def create(self, *args, **kwargs):
        self._counter.calls += 1
        return self._messages.create(*args, **kwargs)


class _CallCounter:
    """Wraps a real anthropic.Anthropic client to count messages.create
    calls, as a stand-in for the paper's per-question query/cost metric."""

    def __init__(self, client: anthropic.Anthropic):
        self.calls = 0
        self.messages = _CountingMessages(client.messages, self)


def run_scenario(scenario: Scenario, model: str, enable_probing: bool, tmp_root: Path) -> dict:
    workdir = tmp_root / scenario.name / ("probed" if enable_probing else "unprobed")
    scenario.build_workdir(workdir)
    trace = scenario.trace_factory(str(workdir))
    trace_path = workdir.parent / f"{'probed' if enable_probing else 'unprobed'}.trace.json"
    trace_path.write_text(json.dumps(trace))

    counter = _CallCounter(anthropic.Anthropic())
    skill = generate_skill([str(trace_path)], model=model, client=counter, enable_probing=enable_probing)

    passed = skill["action"] == "commit" and scenario.check(skill["markdown"])
    return {
        "scenario": scenario.name,
        "probing": enable_probing,
        "action": skill["action"],
        "passed": passed,
        "api_calls": counter.calls,
    }


def run_eval(model: str = DEFAULT_MODEL, scenarios: list[Scenario] | None = None) -> list[dict]:
    scenarios = scenarios if scenarios is not None else ALL_SCENARIOS
    results = []
    with tempfile.TemporaryDirectory(prefix="probe_eval_") as tmp:
        tmp_root = Path(tmp)
        for scenario in scenarios:
            for enable_probing in (False, True):
                results.append(run_scenario(scenario, model, enable_probing, tmp_root))
    return results


def format_report(results: list[dict]) -> str:
    lines = [
        f"{'scenario':<16} {'probing':<8} {'action':<8} {'passed':<7} {'api_calls':<9}",
        "-" * 52,
    ]
    for r in results:
        lines.append(
            f"{r['scenario']:<16} {str(r['probing']):<8} {r['action']:<8} "
            f"{str(r['passed']):<7} {r['api_calls']:<9}"
        )

    def _rate(probing: bool) -> str:
        subset = [r for r in results if r["probing"] == probing]
        if not subset:
            return "n/a"
        passed = sum(1 for r in subset if r["passed"])
        return f"{passed}/{len(subset)}"

    lines.append("")
    lines.append(f"pass rate without probing: {_rate(False)}")
    lines.append(f"pass rate with probing:    {_rate(True)}")
    return "\n".join(lines)

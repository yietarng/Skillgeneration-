"""P5 -- closed-loop credit attribution (spec §5.5): when multiple skills
were co-active on one task, attribute outcome credit per skill rather than
crediting every co-active skill equally. This is what makes the spec §7
retrieval-hit-rate / activation-pass-rate metrics meaningful per skill
instead of only in aggregate.

Heuristic, not ground truth: there's no direct signal for "which skill's
guidance the agent actually followed," so this approximates it via lexical
overlap between the trace's tool-call text and each injected skill's
adapted procedure -- the skill whose wording the executed commands most
resemble gets more credit. Documented as an approximation on purpose,
matching this project's other heuristic-but-real building blocks (e.g.
skill_library.index's hashing embedding).
"""

from __future__ import annotations

import dataclasses
import re

from trace_collection.schema import Trace

from .adapter import AdaptedSkill

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _trace_tokens(trace: Trace) -> set[str]:
    tokens: set[str] = set()
    for step in trace.steps:
        for call in step.tool_calls:
            tokens |= _tokenize(str(call.input))
            tokens |= _tokenize(call.output)
    return tokens


@dataclasses.dataclass
class AttributedCredit:
    skill_id: str
    version: int
    credit: float  # sums to 1.0 across all injected skills for one trace


def attribute_credit(trace: Trace, injected: list[AdaptedSkill]) -> list[AttributedCredit]:
    if not injected:
        return []
    if len(injected) == 1:
        skill = injected[0]
        return [AttributedCredit(skill.skill_id, skill.version, 1.0)]

    trace_tokens = _trace_tokens(trace)
    overlaps = [len(trace_tokens & _tokenize(skill.adapted_procedure)) for skill in injected]
    total = sum(overlaps)

    if total == 0:
        # No detectable overlap with any -- split evenly rather than
        # fabricating a preference the trace doesn't evidence.
        share = 1.0 / len(injected)
        return [AttributedCredit(s.skill_id, s.version, share) for s in injected]

    return [
        AttributedCredit(s.skill_id, s.version, overlap / total)
        for s, overlap in zip(injected, overlaps)
    ]

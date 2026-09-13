"""Shared DSPy LM configuration.

Every DSPy module in this package (Segmenter, AbstractionJudge, Abstractor,
and later skill_library.librarian.Librarian) reads its backend model through
here, via ``dspy.configure(lm=...)``, rather than constructing its own
``dspy.LM``. This is the one seam where swapping the backend model -- the
spec's model-agnostic requirement -- is a config change, not a rewrite.

Not called automatically by dspy_modules.p1_extraction: the caller (CLI
entrypoint, or a test using dspy.utils.DummyLM instead) is responsible for
configuring an LM before invoking extract_skills(), so the extraction
pipeline itself stays testable without live API credentials.
"""

from __future__ import annotations

import os

import dspy

# litellm-style "provider/model" id. claude-opus-5 matches
# trace_collection/skill_generator.py's DEFAULT_MODEL; adjust if litellm's
# supported model list uses a different slug at deploy time.
DEFAULT_MODEL = "anthropic/claude-opus-5"
DEFAULT_MAX_TOKENS = 8192


def configure_lm(model: str | None = None, **lm_kwargs) -> dspy.LM:
    """Configure and return the dspy.LM used by every module in this package.

    Reads ANTHROPIC_API_KEY from the environment (required -- there is no
    silent fallback). ``model`` defaults to DEFAULT_MODEL, overridable via
    the SKILLGEN_DSPY_MODEL env var or this argument.
    """
    model = model or os.environ.get("SKILLGEN_DSPY_MODEL", DEFAULT_MODEL)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. dspy_modules needs a real Anthropic "
            f"API key to call {model}. For tests without live credentials, "
            "configure dspy.utils.DummyLM instead of calling configure_lm()."
        )
    lm_kwargs.setdefault("max_tokens", DEFAULT_MAX_TOKENS)
    lm = dspy.LM(model, api_key=api_key, **lm_kwargs)
    dspy.configure(lm=lm)
    return lm

# Quickstart

Just want to run the code? Follow these steps in order. For what each file
does and how the pipeline fits together, see [README.md](README.md).

## 1. Install

```bash
git clone <this-repo-url>
cd Skillgeneration-
pip install -r requirements.txt
```

## 2. Set your API key

Only needed for steps 3 and 5 below (anything that calls Claude).
`fetch-public-traces` (step 4) does not need it.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## 3. See the effect of grounding, with zero setup

This runs the curator on small built-in test scenarios (probing forced off
vs on) and prints a pass/fail table — the fastest way to see the mechanism
work:

```bash
python -m trace_collection.cli eval
```

## 4. Get some real traces to work with

Pulls real public coding-agent trajectories from GitHub and converts them
into this project's format, with a real environment attached so the
curator has something to check:

```bash
python -m trace_collection.cli fetch-public-traces --out-dir ./sample_traces
```

(This repo already ships 3 pre-fetched examples under `sample_traces/`, so
you can skip straight to step 5 if you just want to try it.)

## 5. Turn a trace into a skill

```bash
python -m trace_collection.cli generate-skill \
    "./sample_traces/marshmallow-code__marshmallow-1867/trace.json" \
    --out-dir ./skills
```

Look at what got written:

```bash
cat ./skills/*/SKILL.md
cat ./skills/*/GROUNDING.md   # what was checked against the real environment, and why
```

Compare against skipping the grounding step entirely:

```bash
python -m trace_collection.cli generate-skill \
    "./sample_traces/marshmallow-code__marshmallow-1867/trace.json" \
    --out-dir ./skills-unprobed --no-probing
```

## 6. (Optional) Record your own trace

Instead of using a pre-fetched one, have Claude actually attempt a task in
a sandboxed folder and record everything it does:

```bash
python -m trace_collection.cli collect "Add a /health endpoint to app.py" \
    --workdir ./sandbox --out-dir ./traces
```

Then feed the result into step 5 the same way:

```bash
python -m trace_collection.cli generate-skill "./traces/*.json" --out-dir ./skills
```

## Troubleshooting

- **`ModuleNotFoundError: No module named 'anthropic'`** → run step 1 again.
- **`generate-skill`/`collect`/`eval` fail with `Could not resolve
  authentication method`** → `ANTHROPIC_API_KEY` isn't set (step 2).
- **`fetch-public-traces` errors on network calls** → it needs outbound
  access to `github.com` and `raw.githubusercontent.com`, plus a `git`
  binary on `PATH`.
- **"Curator skipped this skill"** → this is the curator working as
  intended: probing showed the draft would be misleading, so nothing was
  written. Not a bug.

# Contributing to CereScan

Thanks for helping improve CereScan.

## Development workflow

1. Create an issue describing the bug, experiment, or feature.
2. Create a focused branch from `main`.
3. Make the smallest coherent change.
4. Add or update tests for behavior you changed.
5. Update documentation when the public behavior or workflow changes.
6. Run the local checks before opening a pull request.
7. Open a pull request with a clear summary, testing evidence, and limitations.

## Local setup

Use Python 3.12 for the supported development environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quality checks

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

If you cannot run the ML stack locally because of hardware or dependency constraints, explain that clearly in the pull request.

## Pull requests

A good PR should include:

- what changed;
- why it changed;
- how it was tested;
- screenshots for UI changes when useful;
- dataset/model implications;
- known limitations.

Keep PRs focused. Do not commit datasets, model weights, secrets, generated reports, or local virtual environments.

## ML-specific rules

- Never fabricate accuracy or benchmark results.
- Record dataset version and preprocessing changes.
- Keep train/validation/test data separation intact.
- Do not claim clinical performance from a student dataset.
- Add reproducibility details for experiments.

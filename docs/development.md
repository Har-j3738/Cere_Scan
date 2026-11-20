# Development Guide

## Repository philosophy

CereScan follows a reproducible, reviewable workflow rather than treating the repository as a collection of scripts.

### Principles

- Keep application, training, evaluation, and data preparation responsibilities separate.
- Prefer small functions with explicit inputs and outputs.
- Add tests when behavior changes.
- Document assumptions that affect scientific validity.
- Never commit datasets, model checkpoints, secrets, or generated local artifacts.

## Branches

Use short-lived branches:

```text
main
  ├── feature/...
  ├── fix/...
  └── docs/...
```

Keep `main` releasable.

## Commit messages

Prefer imperative, focused messages:

- `Add four-class dataset validation`
- `Fix Grad-CAM hook cleanup`
- `Document training reproducibility`

Avoid vague messages such as `changes` or `final update`.

## Testing

Run:

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

## ML reproducibility

Record:

- dataset name and version;
- preprocessing;
- train/validation/test split;
- model architecture;
- pretrained weights;
- random seed;
- optimizer and learning rate;
- batch size;
- number of epochs;
- hardware;
- final test metrics.

Do not copy a metric into the README merely because it appeared in a paper or another repository. It must come from the CereScan run being reported.

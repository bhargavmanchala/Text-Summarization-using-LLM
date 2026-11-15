# Contributing

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Workflow

1. Branch from `main` using a prefix: `feat/`, `fix/`, `docs/`, `refactor/`, `chore/`.
2. Make small, focused commits.
3. Run `ruff check .` and `pytest` before pushing; CI runs the same checks.
4. Open a pull request into `main`. Do not push directly to `main`.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/): `type: short summary` in the
imperative mood, with a body explaining *why* when it is not obvious.

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `build`, `ci`, `chore`.

## Rules

- **Never commit secrets** (API keys, `.env`). Use environment variables.
- Do not commit model weights, vector indexes or generated files (see `.gitignore`).
- Add or update a test for behaviour changes. Tests must not download models or call the network.
- Put documents in `docs/report`, `docs/presentation` or `docs/media`.

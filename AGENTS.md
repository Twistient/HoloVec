# Repository Guidelines

## Project Structure & Module Organization
Core runtime code resides in `holovec/` with focused subpackages: `models/` for algebra primitives, `encoders/` for kernel-aware mappings, `retrieval/` and `spaces/` for cleanup/search utilities, and `backends/` to keep NumPy/PyTorch/JAX shims isolated. Shared helpers live in `utils/` and `constants.py`. Tests mirror this layout in `tests/` (`test_encoders_*`, `test_utils_*`, etc.) and should add fixtures beside the closest module. Documentation sources live under `docs/source`, while runnable walkthroughs belong in `examples/`; performance notebooks can go in `benchmarks/`. Coverage HTML drops into `htmlcov/`, so avoid checking it in.

## Build, Test, and Development Commands
- `uv pip install -e .[all]` (or `pip install -e .[all]`) sets up local development with optional torch/jax, linting, typing, and docs extras.
- `pytest -v --cov=holovec --cov-report=term-missing --cov-report=html` is the canonical test run; it matches the `pyproject` defaults and refreshes `htmlcov/`.
- `ruff check holovec tests` and `black holovec tests` keep formatting consistent; run them before pushing.
- `mypy holovec tests` enforces the strict typing gates (no implicit optional, disallow untyped defs).
- `sphinx-build -b html docs/source docs/_build/html` generates the ReadTheDocs build locally when editing docs.

## Coding Style & Naming Conventions
Python 3.9+ code must remain fully typed; prefer `typing.Protocol`/`TypedDict` over `Any`. Use 4-space indentation, 100-character lines, `snake_case` for functions/modules, `PascalCase` for public classes, and leading underscores for private helpers. Keep binding/encoder factories pure and deterministic—inject backends via parameters instead of globals.

## Testing Guidelines
Each new feature needs at least one `tests/test_<area>*.py` case that exercises NumPy plus any newly supported backends. Mirror the naming of the implementation module, e.g., `holovec/encoders/vector.py` → `tests/test_encoders_vector.py`. Maintain ≥90% coverage by extending existing parametrized suites instead of duplicating fixtures. When tests generate artefacts or temporary vectors, use `pytest` fixtures and mark backend-specific skips explicitly.

## Commit & Pull Request Guidelines
Commits follow short, imperative subjects with optional prefixes (`Fix:`, `Refactor`, `Add`). Write body text when behavior changes or migrations occur. Pull requests should include: (1) a high-level summary with links to issues/roadmap items, (2) test evidence (command output or coverage deltas), (3) screenshots for doc/demo updates, and (4) notes on backward compatibility or runtime impact. Reference `SECURITY.md` if a change touches cryptographic or trust boundaries, and ensure CI stays green before requesting review.

# AGENTS.md

This file provides guidance for AI coding agents working on the HoloVec codebase.

## Commands

- **Install**: `uv sync --extra dev` (use `uv pip install -e .[all]` only when you explicitly need every optional extra)
- **Test all**: `uv run --extra dev pytest`
- **Test single file**: `uv run --extra dev pytest tests/test_models.py -v`
- **Test single test**: `uv run --extra dev pytest tests/test_models.py::test_function_name -v`
- **Test with coverage**: `uv run --extra dev pytest --cov=holovec --cov-report=term-missing`
- **Test without coverage**: `uv run --extra dev pytest --no-cov`
- **Lint**: `uv run --extra dev ruff check holovec tests`
- **Format**: `uv run --extra dev black holovec tests`
- **Type check**: `uv run --extra dev mypy holovec`

## Code Style

- Python 3.11+, fully typed (no `Any`, use `Protocol`/`TypedDict`)
- 100-char lines, 4-space indent, Black formatting
- `snake_case` functions/modules, `PascalCase` classes, `_leading_underscore` for private
- Imports: stdlib → third-party → local (isort via ruff handles this)
- Keep encoders/factories pure and deterministic; inject backends via parameters

## Testing

- Test files mirror source: `holovec/encoders/vector.py` → `tests/test_encoders_vector.py`
- Extend existing parametrized suites; use pytest fixtures for temp artifacts
- Target ≥90% coverage; mark backend-specific skips explicitly
- Engine CI matches the local gate: `pytest`, `ruff check holovec tests`, and `mypy holovec`

## Commit Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <description>

Types:
- feat:     New feature (triggers minor version bump)
- fix:      Bug fix (triggers patch version bump)
- docs:     Documentation only
- test:     Adding/updating tests
- refactor: Code change that neither fixes a bug nor adds a feature
- chore:    Maintenance tasks (deps, configs, etc.)
- perf:     Performance improvement
- ci:       CI/CD changes

Examples:
- feat: Add dict-like interface to Codebook
- fix: Correct HRR unbind similarity calculation
- test: Improve GHRR model coverage (46% → 100%)
- docs: Update installation instructions
- chore: Bump minimum Python version to 3.11
```

## Release Process

### When to Release

Release a new version when:
- Breaking changes have accumulated (→ major bump)
- New features are ready for users (→ minor bump)
- Bug fixes need to reach users (→ patch bump)
- Significant test/doc improvements warrant visibility

### Pre-Release Checklist

1. **Ensure all tests pass**:
   ```bash
   uv run --extra dev pytest tests/ -q
   ```

2. **Review unreleased changes**:
   ```bash
   git log $(git describe --tags --abbrev=0)..HEAD --oneline
   ```

3. **Determine version bump** (see Semantic Versioning below)

### Creating a Release

1. **Update CHANGELOG.md**:
   - Move items from `[Unreleased]` to new version section
   - Add release date: `## [X.Y.Z] - YYYY-MM-DD`
   - Categorize changes: Added, Changed, Deprecated, Removed, Fixed, Security, Testing, Documentation
   - Update comparison links at bottom of file

2. **Update version numbers**:
   - `pyproject.toml`: `version = "X.Y.Z"`
   - `CITATION.cff`: `version: X.Y.Z` and `date-released: YYYY-MM-DD`

3. **Commit the release**:
   ```bash
   git add CHANGELOG.md pyproject.toml CITATION.cff
   git commit -m "chore: Release vX.Y.Z"
   ```

4. **Push and tag** (Option A - CLI):
   ```bash
   git push origin master
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin --tags
   ```

   **Or** (Option B - GitHub UI):
   ```bash
   git push origin master
   ```
   Then create the release on GitHub via Releases → "Draft a new release", which creates the tag automatically.

### Semantic Versioning

Follow [SemVer](https://semver.org/):

- **MAJOR** (X.0.0): Breaking changes
  - Removed public API
  - Changed function signatures
  - Dropped Python version support
  
- **MINOR** (0.X.0): New features (backwards compatible)
  - New models, encoders, or utilities
  - New optional parameters
  - New CLI commands
  
- **PATCH** (0.0.X): Bug fixes (backwards compatible)
  - Bug fixes
  - Documentation fixes
  - Test improvements (no API changes)

### CHANGELOG Format

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New features

### Changed
- Changes to existing functionality
- Use "BREAKING:" prefix for breaking changes

### Deprecated
- Features that will be removed

### Removed
- Removed features

### Fixed
- Bug fixes

### Security
- Security fixes

### Testing
- Test improvements

### Documentation
- Documentation updates
```

## Maintaining AGENTS.md

### When to Propose Updates

Update AGENTS.md when:

1. **Process changes**: New workflows, release procedures, or conventions are established
2. **Command changes**: Build/test/lint commands are added, modified, or deprecated
3. **Style changes**: Code style guidelines are updated (linter rules, formatting, etc.)
4. **Lessons learned**: Recurring issues reveal missing guidance
5. **Tool changes**: New tools are adopted or existing tools are replaced
6. **Version changes**: Python version requirements change

### How to Propose Updates

1. **Identify the gap**: Note what guidance was missing or unclear during your session
2. **Draft the addition**: Write clear, concise guidance following existing format
3. **Include examples**: Add concrete examples where helpful
4. **Keep it current**: Remove outdated information when adding new guidance
5. **Commit separately**: Use `docs: Update AGENTS.md with <topic>` commit message

### What NOT to Include

- Temporary workarounds (fix the root cause instead)
- Project-specific secrets or credentials
- Verbose explanations (keep it scannable)
- Duplicate information from other docs (link instead)

## Project-Specific Notes

### Git Repository

- **Default branch**: `master` (not `main`)
- **Remote**: `origin` points to `github.com:Twistient/HoloVec`

### Dependency Management

This project uses `uv` for dependency management.

**Keep `uv.lock` committed** - it ensures reproducible builds:
- Pins exact dependency versions for CI consistency
- Enables security auditing of specific versions
- Makes debugging dependency issues easier

```bash
# Install dependencies
uv pip install -e .[all]

# Update dependencies
uv lock --upgrade

# Update a specific package
uv lock --upgrade-package <package-name>
```

**Do NOT add `uv.lock` to `.gitignore`** - it should always be tracked.

### Backend Architecture

HoloVec supports multiple computational backends (NumPy, PyTorch, JAX). When writing code:
- Never import a specific backend at module level (except numpy for type stubs)
- Use `backend.method()` calls, not direct numpy/torch/jax calls
- Test across all available backends when possible
- Treat NumPy as the release-blocking backend. PyTorch and JAX are optional support paths unless
  their backend-specific tests are explicitly exercised.

### Test Coverage Targets

| Module | Target |
|--------|--------|
| models/ | ≥90% |
| encoders/ | ≥90% |
| retrieval/ | ≥90% |
| utils/ | ≥90% |
| backends/ | ≥80% |
| spaces/ | ≥80% |

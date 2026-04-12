# Contributing to HoloVec

Thank you for your interest in contributing to HoloVec! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to keep our community welcoming and inclusive.

## Getting Started

### Development Setup

1. **Fork and clone the repository**

   ```bash
   git clone https://github.com/twistient/HoloVec.git
   cd HoloVec
   ```

2. **Install development dependencies**

   ```bash
   uv sync --extra dev
   ```

4. **Install pre-commit hooks** (optional but recommended)

   ```bash
   pre-commit install
   ```

### Running Tests

```bash
# Run all tests
uv run --extra dev pytest

# Run with coverage
uv run --extra dev pytest --cov=holovec --cov-report=html

# Run specific test file
uv run --extra dev pytest tests/test_models.py

# Run tests for specific backend
uv run --extra dev pytest tests/test_models.py -k numpy
```

### Building Docs

```bash
# Sync the locked docs toolchain
uv sync --extra docs

# Build the documentation site locally
uv run --extra docs mkdocs build --strict
```

### Code Style

We use several tools to maintain code quality:

- **Black** - Code formatting (line length: 100)
- **Ruff** - Fast linting and import sorting
- **Mypy** - Type checking

Run these tools before committing:

```bash
# Format code
uv run --extra dev black holovec tests examples

# Lint code
uv run --extra dev ruff check holovec tests examples

# Type check
uv run --extra dev mypy holovec
```

Or use the pre-commit hooks to run automatically:

```bash
pre-commit run --all-files
```

## How to Contribute

### Reporting Bugs

Before creating a bug report, please check existing issues to avoid duplicates.

When filing a bug report, include:

- **Description**: Clear description of the issue
- **Reproduction**: Minimal code to reproduce the problem
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**: Python version, OS, backend (NumPy/PyTorch/JAX)
- **Version**: HoloVec version (`python -c "import holovec; print(holovec.__version__)"`)

### Suggesting Enhancements

Enhancement suggestions are welcome! Please include:

- **Use case**: What problem does this solve?
- **Proposed solution**: How should it work?
- **Alternatives considered**: Other approaches you've thought about
- **Implementation ideas**: Technical approach (if applicable)

### Pull Requests

1. **Create a feature branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, documented code
   - Add tests for new functionality
   - Update documentation as needed
   - Update `CHANGELOG.md` under `[Unreleased]` for any user-visible change
   - Follow existing code style

3. **Test thoroughly**

   ```bash
   # Run tests
   uv run --extra dev pytest

   # Check coverage
   uv run --extra dev pytest --cov=holovec --cov-report=term-missing

   # Run type checker
   uv run --extra dev mypy holovec

   # Format and lint
   uv run --extra dev black holovec tests examples
   uv run --extra dev ruff check holovec tests examples
   ```

4. **Update CHANGELOG.md**
   - Add entry under `[Unreleased]` section
   - Use appropriate category: Added, Changed, Deprecated, Removed, Fixed, Security, Testing, Documentation
   - Prefix breaking entries with `BREAKING:` and include a short migration note when needed
   - Do not bump package versions in feature PRs; version/date updates are release-only work

5. **Commit your changes**

   ```bash
   git add .
   git commit -m "fix: brief description"
   ```

   Follow Conventional Commits:
   - `feat: ...` for new features
   - `fix: ...` for bug fixes
   - `refactor: ...` for code improvements
   - `docs: ...` for documentation changes
   - `test: ...` for test additions/changes
   - `chore: ...` for maintenance

6. **Push and create pull request**

   ```bash
   git push origin feature/your-feature-name
   ```

   Then create a pull request on GitHub with:
   - Clear title summarizing the change
   - Description of what changed and why
   - Verification commands/results (`pytest`, `ruff`, `mypy`, and docs build if docs/examples changed)
   - Changelog/migration summary for any public behavior or persistence change
   - Reference to related issues (e.g., "Fixes #123")
   - Screenshots/examples if applicable

   For breaking or wide-ranging changes, open the PR as a **draft** first.

## Development Guidelines

### Adding New VSA Models

When adding a new VSA model:

1. Create model class in `holovec/models/`
2. Implement required operations (bundle, bind, permute, etc.)
3. Add comprehensive tests in `tests/test_models.py`
4. Add property-based tests with Hypothesis
5. Test with all backends (NumPy, PyTorch, JAX)
6. Document the model theory in docstrings
7. Add examples in `examples/`
8. Update README.md and docs

### Adding New Encoders

When adding a new encoder:

1. Create encoder class in `holovec/encoders/`
2. Inherit from appropriate base class
3. Implement `encode()` method
4. Add tests in `tests/test_encoders.py`
5. Include edge case testing
6. Test with multiple backends
7. Document parameters and usage
8. Add example usage in docstrings

### Adding Backend Support

When adding a new backend:

1. Create backend class in `holovec/backends/`
2. Inherit from `Backend` base class
3. Implement all required methods
4. Add capability probes (supports_gpu, etc.)
5. Add comprehensive tests
6. Update README.md with backend info
7. Add installation instructions

### Documentation Standards

- Use NumPy-style docstrings
- Include parameter types and descriptions
- Provide usage examples
- Document return types
- Note any important caveats or limitations
- Reference papers/theory where applicable

Example:

```python
def encode(self, value: float) -> Array:
    """Encode a scalar value into a hypervector.

    Parameters
    ----------
    value : float
        The scalar value to encode. Should be in range [0, 1].

    Returns
    -------
    Array
        Hypervector representation of the input value.
        Shape: (d,) where d is the dimensionality.

    Examples
    --------
    >>> encoder = FractionalPowerEncoder(dim=1000, exponent=2.0)
    >>> hv = encoder.encode(0.5)
    >>> hv.shape
    (1000,)

    Notes
    -----
    This encoder uses fractional binding as described in [1]_.

    References
    ----------
    .. [1] Kleyko et al. (2023). "Hyperdimensional Computing Survey"
    """
```

### Testing Standards

- Aim for >90% code coverage
- Test edge cases and error conditions
- Use property-based testing where appropriate
- Test with all supported backends
- Include both unit and integration tests
- Test numerical stability
- Verify dtype preservation

## Project Structure

```
holovec/
├── backends/          # Computational backends (NumPy, PyTorch, JAX)
├── models/           # VSA model implementations
├── encoders/         # Data encoding strategies
├── spaces/           # Vector space implementations
├── retrieval/        # Retrieval and search utilities
└── utils/            # Utility functions

tests/                # Test suite
examples/             # Usage examples
docs/                 # Documentation
```

## Questions?

If you have questions about contributing:

- Check the [documentation](https://holovec.readthedocs.io)
- Open a [discussion](https://github.com/Twistient/HoloVec/discussions)
- Ask in an [issue](https://github.com/Twistient/HoloVec/issues)

## License

By contributing to HoloVec, you agree that your contributions will be licensed under the Apache License 2.0.

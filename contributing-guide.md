# Contributing to Lost in OCR Translation

Thank you for your interest in contributing to this research project! We welcome contributions from the community.

## Getting Started

1. Fork the repository
2. Create a new branch for your feature/fix
3. Make your changes
4. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/lost-in-ocr-translation.git
cd lost-in-ocr-translation

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dev dependencies
pip install -r requirements-dev.txt
```

## Code Style

We use the following tools to maintain code quality:

- **Black** for Python formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run all checks:
```bash
make lint
```

## Testing

All new features should include tests:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_retrieval.py
```

## Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Add entry to CHANGELOG.md
4. Follow conventional commit format:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `test:` Test additions/changes
   - `refactor:` Code refactoring

## Reporting Issues

Please use GitHub Issues to report bugs or request features. Include:

- Python version
- OS information
- Complete error traceback
- Minimal reproducible example

## Research Contributions

If you're extending the research:

1. Document your experimental setup
2. Include evaluation metrics
3. Provide comparison with baseline
4. Update relevant sections in README

## Questions?

Feel free to open a discussion or reach out to the maintainers.
# Contributing to Opti-Oignon

Thank you for your interest in contributing! This guide will help you get started.

## Ways to Contribute

- **Report bugs** - Found something broken? Open an issue
- **Suggest features** - Have an idea? We'd love to hear it
- **Improve documentation** - Help others understand the project
- **Submit code** - Fix bugs or add features
- **Test and review** - Try new features and provide feedback

## Getting Started

### 1. Fork and Clone

```bash
# Fork via GitHub UI, then:
git clone https://github.com/YOUR-USERNAME/opti-oignon.git
cd opti-oignon
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

## Development Workflow

### Code Style

We follow PEP 8 with some flexibility. Key points:

- **Line length**: 100 characters max
- **Quotes**: Double quotes for strings
- **Imports**: Grouped (stdlib, third-party, local)
- **Type hints**: Encouraged for public functions

```python
# Good
def process_query(prompt: str, model: str = "default") -> dict:
    """Process a user query and return results."""
    result = {"status": "success", "data": None}
    return result

# Avoid
def process_query(prompt,model="default"):
    result = {'status':'success','data':None}
    return result
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=opti_oignon

# Run specific test file
pytest tests/test_analyzer.py

# Run with verbose output
pytest -v
```

### Code Formatting

```bash
# Format with black
black opti_oignon/

# Sort imports
isort opti_oignon/

# Check types (optional)
mypy opti_oignon/
```

## Submitting Changes

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add support for YAML chunking in RAG
fix: resolve timeout issue with large models
docs: update installation guide for Windows
refactor: simplify router logic
test: add tests for preset matching
```

Prefixes:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `refactor:` - Code change that neither fixes nor adds
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

### Pull Request Process

1. **Update documentation** if you changed behavior
2. **Add tests** for new features
3. **Run the test suite** and ensure it passes
4. **Update CHANGELOG.md** with your changes
5. **Create the PR** with a clear description

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Testing
How did you test these changes?

## Checklist
- [ ] Code follows project style
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] CHANGELOG updated
```

## Project Structure

```
opti-oignon/
├── opti_oignon/          # Main package
│   ├── agents/           # Multi-agent system
│   ├── config/           # Configuration files
│   ├── rag/              # RAG system
│   ├── routing/          # Model routing
│   ├── analyzer.py       # Task analysis
│   ├── executor.py       # Query execution
│   ├── router.py         # Model selection
│   └── ui.py             # Gradio interface
├── tests/                # Test suite
├── docs/                 # Documentation
└── examples/             # Usage examples
```

## Adding New Features

### New Preset

1. Add to `config/presets.yaml` or `data/user_presets.yaml`
2. Include keywords with weights
3. Test auto-detection
4. Document in README if significant

### New Agent

1. Create `agents/specialists/your_agent.py`
2. Inherit from `BaseAgent`
3. Register in `agents/__init__.py`
4. Add to `agents/config.yaml`
5. Write tests

### New Chunker (RAG)

1. Add function in `rag/chunkers.py`
2. Register in `CHUNKERS` dict
3. Add file extension mapping
4. Test with sample files

## Reporting Issues

### Bug Reports

Include:
- **Description**: What happened?
- **Expected behavior**: What should happen?
- **Steps to reproduce**: How can we see the bug?
- **Environment**: OS, Python version, Ollama version
- **Logs**: Error messages or tracebacks

### Feature Requests

Include:
- **Use case**: Why do you need this?
- **Proposed solution**: How should it work?
- **Alternatives**: Other approaches considered?

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions

## Questions?

- Open a [Discussion](https://github.com/AntsAreRad/opti-oignon/discussions)
- Check existing issues first
- Be patient - this is a side project!

---

Thank you for contributing! 🧅

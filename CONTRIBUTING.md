# Contributing to AI Medical Record Summarizer

Thank you for your interest in contributing to the AI Medical Record Summarizer project! We welcome contributions from the community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)
- [License](#license)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
   ```bash
   git clone https://github.com/your-username/AI_Medical_Record_Summarizer.git
   cd AI_Medical_Record_Summarizer
   ```
3. Set up the development environment (see README.md)

## Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-number-description
   ```

2. Make your changes and commit them with a descriptive message:
   ```bash
   git add .
   git commit -m "feat: add new feature"
   # or
   git commit -m "fix: resolve issue with xyz"
   ```

3. Push your changes to your fork:
   ```bash
   git push origin your-branch-name
   ```

4. Open a Pull Request against the main branch

## Code Style

We use the following tools to maintain code quality:

- **Python**: Black for formatting, isort for import sorting, flake8 for linting
- **JavaScript/TypeScript**: Prettier for formatting, ESLint for linting

Before committing, please run:

```bash
# Format Python code
black .
isort .

# Check for linting issues
flake8
```

## Testing

We use pytest for testing Python code. Please add tests for new features and ensure all tests pass before submitting a PR.

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=backend tests/
```

## Pull Request Process

1. Ensure all tests pass
2. Update the documentation if necessary
3. Ensure your code follows the style guidelines
4. Open a Pull Request with a clear description of the changes
5. Reference any related issues in your PR description

## Reporting Issues

When reporting issues, please include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs. actual behavior
- Any relevant error messages or logs
- Your environment (OS, Python version, etc.)

## License

By contributing, you agree that your contributions will be licensed under the project's [LICENSE](LICENSE) file.

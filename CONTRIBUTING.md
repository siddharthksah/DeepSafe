# Contributing to DeepSafe

First off, thanks for taking the time to contribute! 🎉

The following is a set of guidelines for contributing to DeepSafe. These are just guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Code of Conduct

This project and everyone participating in it is governed by the [DeepSafe Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

This section guides you through submitting a bug report for DeepSafe. Following these guidelines helps maintainers and the community understand your report, reproduce the behavior, and find related reports.

- **Use a clear and descriptive title** for the issue to identify the problem.
- **Describe the exact steps which reproduce the problem** in as many details as possible.
- **Provide specific examples** to demonstrate the steps.
- **Describe the behavior you observed after following the steps** and point out what exactly is the problem with that behavior.
- **Explain which behavior you expected to see instead and why.**

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for DeepSafe, including completely new features and minor improvements to existing functionality.

- **Use a clear and descriptive title** for the issue to identify the suggestion.
- **Provide a step-by-step description of the suggested enhancement** in as many details as possible.
- **Explain why this enhancement would be useful** to most DeepSafe users.

### Pull Requests

- Fill in the required template
- Do not include issue numbers in the PR title
- Include screenshots and animated GIFs in your pull request whenever possible.
- Follow the Python style guides (PEP 8).
- End all files with a newline.

## Development Setup

1.  **Clone the repo**:
    ```bash
    git clone https://github.com/yourusername/DeepSafe.git
    cd DeepSafe
    ```

2.  **Set up environment**:
    Copy the example environment file:
    ```bash
    cp .env.example .env
    ```

3.  **Run with Docker**:
    ```bash
    make start
    ```

4.  **Run Tests**:
    ```bash
    make test
    ```

## Styleguides

### Git Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

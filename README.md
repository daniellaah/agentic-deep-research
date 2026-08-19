# Agentic Deep Research

Agentic Deep Research is a Python project for building a reliable deep research agent with testable workflows and clear model and tool boundaries.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

## Setup

Clone the repository:

```bash
git clone https://github.com/daniellaah/agentic-deep-research.git
cd agentic-deep-research
```

Install the project and development dependencies:

```bash
uv sync --dev
```

Create the local environment file:

```bash
cp .env.example .env
```

Configure the environment variables required by the integrations you use:

```dotenv
OPENAI_API_KEY=
TAVILY_API_KEY=
MODEL_NAME=
```

Do not commit `.env` or API keys.

## Quality Checks

Run the test suite:

```bash
uv run pytest
```

Run the code-quality checks:

```bash
uv run ruff check src tests
```

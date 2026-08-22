# Agentic Deep Research

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-Responses_API-412991?logo=openai&logoColor=white)](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)
[![uv](https://img.shields.io/badge/Package_Manager-uv-DE5FE9?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![CI](https://github.com/daniellaah/agentic-deep-research/actions/workflows/ci.yml/badge.svg)](https://github.com/daniellaah/agentic-deep-research/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A learning-oriented deep-research CLI that adds model and Agent mechanisms one observable
release at a time. The current release exposes a deterministic write-critic-revise
workflow before research tools and an autonomous Agent loop are introduced.

## Core Features

- Accepts a research question through a single CLI command.
- Runs an explicit write-critic-revise workflow with three synchronous Responses API
  requests.
- Shows each fixed stage's progress and the final report in the terminal.
- Saves the draft, critique, and final report in a unique run directory.
- Loads credentials and model selection from environment variables.
- Keeps the complete runtime in one Python file.

## Quick Start

### Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key and access to the model you want to use

### Install

```bash
git clone https://github.com/daniellaah/agentic-deep-research.git
cd agentic-deep-research
uv sync --locked
```

Create your local configuration:

```bash
cp .env.example .env
```

Set both required values in `.env`:

```dotenv
OPENAI_API_KEY=your_api_key
MODEL_NAME=your_model_name
```

Keep `.env` private and never commit it.

### Run

```bash
uv run deep_research.py \
  "Explain how AI agents can support scientific research."
```

## Usage

Pass one quoted research question as the positional argument:

```bash
uv run deep_research.py "Your research question"
```

Show the CLI help:

```bash
uv run deep_research.py --help
```

## Generated Output

The command prints stage progress and the final report directly in the terminal:

```text
[1/3] Write started: <model-name>
[1/3] Write completed: runs/<run-id>/draft.md
[2/3] Critic started: <model-name>
[2/3] Critic completed: runs/<run-id>/critique.md
[3/3] Revise started: <model-name>
[3/3] Revise completed: runs/<run-id>/report.md

Final report:
<final report>

Report written: runs/<run-id>/report.md
Run completed.
```

Each successful run creates the following directory under the repository root:

```text
runs/<run-id>/
├── draft.md
├── critique.md
└── report.md
```

Generated run artifacts are local and ignored by Git.

## Configuration

| Variable | Required | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | Yes | API key used to authenticate with OpenAI. |
| `MODEL_NAME` | Yes | OpenAI model used to generate the report. |

The CLI automatically loads `.env` from the repository root. Values already present in
the process environment take precedence over values in `.env`.

## How It Works

```text
Research question
    → Write
    → Critic
    → Revise
    → terminal output
    → runs/<run-id>/{draft,critique,report}.md
```

The runtime validates the question and configuration, then makes three explicit model
requests. The writer receives the question, the critic receives the question and draft,
and the reviser receives the question, draft, and critique. Each request is stateless and
the application passes only the text needed by the next stage. API and artifact failures
stop the workflow, produce a concise error, and return a nonzero exit code.

## Project Structure

```text
.
├── deep_research.py     # Complete application runtime
├── specs/              # Release specifications
├── .env.example        # Required configuration template
├── pyproject.toml      # Project metadata and dependencies
├── uv.lock             # Locked dependency versions
└── runs/               # Generated reports; ignored by Git
```

## Documentation

- [Changelog](CHANGELOG.md)
- [Roadmap](ROADMAP.md)
- [Tech stack](TECH_STACK.md)
- [Release specifications](specs/README.md)

## License

This project is licensed under the [MIT License](LICENSE).

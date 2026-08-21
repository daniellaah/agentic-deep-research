# Agentic Deep Research

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-Responses_API-412991?logo=openai&logoColor=white)](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)
[![uv](https://img.shields.io/badge/Package_Manager-uv-DE5FE9?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![CI](https://github.com/daniellaah/agentic-deep-research/actions/workflows/ci.yml/badge.svg)](https://github.com/daniellaah/agentic-deep-research/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An observable deep-research agent that plans investigations, searches the web and
academic literature, verifies evidence and citations, and produces source-grounded
reports.

## Core Features

- Accepts a research question through a single CLI command.
- Generates a report with one synchronous OpenAI Responses API request.
- Shows request progress and the generated report in the terminal.
- Saves every successful report in a unique run directory.
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

The command prints progress and the generated report directly in the terminal:

```text
Model request started: <model-name>
Model response received.
<generated report>
Report written: runs/<run-id>/report.md
Run completed.
```

Each successful run creates:

```text
runs/<run-id>/
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
    → OpenAI Responses API
    → response.output_text
    → terminal output
    → runs/<run-id>/report.md
```

The runtime validates the question and configuration, makes one model request, reads the
response text, and writes it to a Markdown report. API and artifact failures produce a
concise error and a nonzero exit code.

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

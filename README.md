# Agentic Deep Research

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-Responses_API-412991?logo=openai&logoColor=white)](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)
[![uv](https://img.shields.io/badge/Package_Manager-uv-DE5FE9?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![CI](https://github.com/daniellaah/agentic-deep-research/actions/workflows/ci.yml/badge.svg)](https://github.com/daniellaah/agentic-deep-research/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A learning-oriented deep-research CLI that adds model and Agent mechanisms one observable
release at a time. The current release exposes a custom tool-using Agent loop, manual
Responses API history, and bounded research before the report enters an explicit
write-critic-revise workflow.

## Core Features

- Accepts a research question through a single CLI command.
- Lets the model select simple Tavily web search and arXiv paper search through two custom
  function tools.
- Keeps the complete ordered Responses API history in the application and enforces hard
  model-turn and tool-call limits.
- Shows Agent turns, tool selections, fixed report stages, and the final report in the
  terminal.
- Keeps research notes, draft, critique, and final report in memory without writing files.
- Loads OpenAI, Tavily Search, and model configuration from environment variables.
- Keeps the workflow in `deep_research.py` and tool details in `agent_tools.py`.

## Quick Start

### Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key and access to the model you want to use
- A [Tavily API](https://www.tavily.com/) key

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

Set all required values in `.env`:

```dotenv
OPENAI_API_KEY=your_api_key
MODEL_NAME=your_model_name
TAVILY_API_KEY=your_tavily_api_key
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
[1/4] Research started.
[Research 1/10] Model started (tools remaining: 8)
[Tool 1/8] tavily_search_tool started.
[Tool 1/8] tavily_search_tool completed.
...
[1/4] Research completed.
[2/4] Write started.
[2/4] Write completed.
[3/4] Critic started.
[3/4] Critic completed.
[4/4] Revise started.
[4/4] Revise completed.

Final report:
<final report>

Run completed.
```

Release 0.3.0 does not create a run directory or write application output files.
Structured tracing is also not part of this release.

## Configuration

| Variable | Required | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | Yes | API key used to authenticate with OpenAI. |
| `MODEL_NAME` | Yes | OpenAI model used to generate the report. |
| `TAVILY_API_KEY` | Yes | API key used for custom web search calls. |

The CLI automatically loads `.env` from the repository root. Values already present in
the process environment take precedence over values in `.env`.

## How It Works

```text
Research question
    → bounded Research Agent
        → tavily_search_tool
        → arxiv_search_tool
        → application-owned response history
    → research notes
    → Write
    → Critic
    → Revise
    → final report in the terminal
```

The Research Agent repeatedly calls the synchronous Responses API, appends every model
output item to an ordered history, executes requested functions, and appends each result
with its matching `call_id`. It stops when the model returns research notes or a hard limit
is reached. The writer receives the question and notes; the critic also receives the draft;
the reviser receives all explicit upstream text. Model failures stop the workflow, while
tool failures are returned to the Agent so it can adapt within the remaining budget.

## Project Structure

```text
.
├── deep_research.py     # Prompts, Agent loop, report workflow, and CLI
├── agent_tools.py       # Tool schemas, implementations, and dispatch
├── specs/              # Release specifications
├── .env.example        # Required configuration template
├── pyproject.toml      # Project metadata and dependencies
└── uv.lock             # Locked dependency versions
```

## Documentation

- [Changelog](CHANGELOG.md)
- [Roadmap](ROADMAP.md)
- [Tech stack](TECH_STACK.md)
- [Release specifications](specs/README.md)

## License

This project is licensed under the [MIT License](LICENSE).

# Agentic Deep Research

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-Responses_API-412991?logo=openai&logoColor=white)](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)
[![uv](https://img.shields.io/badge/Package_Manager-uv-DE5FE9?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![CI](https://github.com/daniellaah/agentic-deep-research/actions/workflows/ci.yml/badge.svg)](https://github.com/daniellaah/agentic-deep-research/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A learning-oriented deep-research CLI that adds model and Agent mechanisms one observable
release at a time. The current release turns an initial question into an explicitly
approved ResearchBrief, creates one validated static research plan from that contract,
executes its tasks sequentially with a custom tool-using Agent loop, and then runs an
explicit write-critic-revise workflow.

## Core Features

- Accepts a research question through a single CLI command.
- Detects material ambiguity and, when needed, asks at most three clarification questions
  in one terminal round.
- Creates and prints one Pydantic-validated ResearchBrief, requires explicit approval, and
  allows at most one revision before final approval.
- Creates and prints one bounded Pydantic-validated research plan with ordered tasks and
  completion criteria.
- Lets the model select simple Tavily web search and arXiv paper search through two custom
  function tools.
- Runs every plan task sequentially with fresh application-owned Responses API history and
  independent hard model-turn and tool-call limits.
- Shows Scope decisions, pending briefs, approval, the plan, task boundaries, Agent turns,
  tool selections, every intermediate result, report stages, and final report in
  consistently formatted terminal sections.
- Keeps scoping data, the approved brief, plan, research notes, draft, critique, and final
  report in memory without writing files.
- Loads OpenAI, Tavily Search, and model configuration from environment variables.
- Keeps the workflow in `deep_research.py`, detailed model instructions in
  `agent_instructions.py`, and tool details in `agent_tools.py`.

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

The command prints stage progress, every intermediate result, and the final report directly
in the terminal:

```text
[1/6] Scope | STARTED
  [Scope] Clarification assessment | COMPLETED | 2 questions

================================================================================
CLARIFICATION QUESTIONS
================================================================================
1. <targeted question>
2. <targeted question>
================================================================================

Answer 1/2: <answer>
Answer 2/2: <answer>

================================================================================
RESEARCH BRIEF | PENDING APPROVAL
================================================================================
## Objective
<resolved objective>
... remaining ResearchBrief fields ...
================================================================================

Action [approve/revise/cancel]: approve
[1/6] Scope | COMPLETED | research brief approved
[2/6] Plan | STARTED
[2/6] Plan | COMPLETED | 2 tasks

================================================================================
RESEARCH PLAN
================================================================================
1. <task title>
   Research question: <focused question>
   Completion criteria:
     - <observable evidence or coverage>
================================================================================

[3/6] Research | STARTED | 2 tasks
  [Task 1/2] <task title> | STARTED
    [Turn 1/10] Model | STARTED | 8 tools remaining
    [Tool 1/8] tavily_search_tool | STARTED
    [Tool 1/8] tavily_search_tool | COMPLETED
...
  [Task 1/2] <task title> | COMPLETED

================================================================================
RESEARCH NOTES | TASK 1/2 | <task title>
================================================================================
<task research notes>
================================================================================

... each remaining task, followed by COMBINED RESEARCH NOTES, DRAFT, and CRITIQUE blocks ...

[6/6] Revise | COMPLETED
================================================================================
FINAL REPORT
================================================================================
<final report>
================================================================================

[Run] Deep research | COMPLETED
```

Release 0.5.0 does not create a run directory or write application output files.
Structured tracing is also not part of this release.

## Configuration

| Variable | Required | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | Yes | API key used to authenticate with OpenAI. |
| `MODEL_NAME` | Yes | OpenAI model supporting Structured Outputs and custom function tools. |
| `TAVILY_API_KEY` | Yes | API key used for custom web search calls. |

The CLI automatically loads `.env` from the repository root. Values already present in
the process environment take precedence over values in `.env`.

## How It Works

```text
Research question
    → bounded clarification assessment
    → optional clarification answers
    → validated ResearchBrief
    → explicit approval or one revision and final approval
    → validated static research plan
    → sequential plan tasks
        → bounded Research Agent with fresh history
            → tavily_search_tool
            → arxiv_search_tool
        → task-scoped research notes
    → Write
    → Critic
    → Revise
    → final report in the terminal
```

Scope first makes one synchronous Structured Outputs request for a Pydantic
`ClarificationAssessment`. It prints and collects at most three questions in one round,
then makes a second request for a complete `ResearchBrief`. Invalid local input repeats
only the current prompt. The user must approve the brief, cancel, or request one replacement
brief and approve that revision. No planning or tool call begins before approval.

The planner receives only the approved brief and validates one Pydantic `ResearchPlan`.
The application prints that immutable plan, then invokes the same Research Agent loop once
per task in order. Each loop appends every model output item to fresh ordered history,
executes requested functions, and links every result with its matching `call_id`. Only the
approved brief and combined note text reach Write; Critic and Revise build on that same
report context without receiving Scope history, the plan, or task metadata. Research uses
automatic tool selection while budget remains and `tool_choice="none"` after the eighth
tool attempt. Model failures stop the workflow, while tool failures are returned to the
active Agent so it can adapt within its remaining budget.

## Project Structure

```text
.
├── deep_research.py       # Scoping, planning, Agent loop, report workflow, and CLI
├── agent_instructions.py  # Detailed instructions for every model stage
├── agent_tools.py         # Tool schemas, implementations, and dispatch
├── specs/                 # Release specifications
├── .env.example           # Required configuration template
├── pyproject.toml         # Project metadata and dependencies
└── uv.lock                # Locked dependency versions
```

## Documentation

- [Changelog](CHANGELOG.md)
- [Roadmap](ROADMAP.md)
- [Tech stack](TECH_STACK.md)
- [Release specifications](specs/README.md)

## License

This project is licensed under the [MIT License](LICENSE).

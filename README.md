# Agentic Deep Research

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-Responses_API-412991?logo=openai&logoColor=white)](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)
[![uv](https://img.shields.io/badge/Package_Manager-uv-DE5FE9?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![CI](https://github.com/daniellaah/agentic-deep-research/actions/workflows/ci.yml/badge.svg)](https://github.com/daniellaah/agentic-deep-research/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A learning-oriented deep-research CLI that adds model and Agent mechanisms one observable
release at a time. Release 0.7.0 turns an initial question into an explicitly approved
ResearchBrief, lets a Research Supervisor adaptively delegate bounded work to isolated Research
Workers, and returns every Worker through an explicit application-owned run contract before the
write-critic-revise workflow. The current runtime uses OpenAI-hosted models; the planned harness
sequence next adds deep source reading, an optional local open-weight Worker, long-horizon context
management, dependency-aware multi-agent scheduling, and batch rollouts before training work
begins.

## Core Features

- Accepts a research question through a single CLI command.
- Detects material ambiguity and, when needed, asks at most three clarification questions
  in one terminal round.
- Creates and prints one Pydantic-validated ResearchBrief, requires explicit approval, and
  allows at most one revision before final approval.
- Uses a Pydantic-validated Supervisor decision containing zero or one next task to select
  one Worker or finish from the current application-owned `ResearchState`.
- Bounds every delegated task to one primary subject or direct comparison with one through three
  evidence targets rather than report-sized completion requirements.
- Lets the model select simple Tavily web search and arXiv paper search through two custom
  function tools.
- Runs at most four Workers sequentially with fresh application-owned Responses API history
  and independent hard model-turn and tool-call limits.
- Represents every Worker as an immutable `AgentRunRequest`, private mutable `AgentRunState`, and
  immutable `AgentRunResult` with visible status, termination reason, and budget usage.
- Shows Scope decisions, pending briefs, approval, Supervisor decisions, Worker boundaries,
  Agent turns, tool selections, every intermediate result, report stages, and final report
  in consistently formatted terminal sections.
- Keeps scoping data, the approved brief, ResearchState, research notes, draft, critique,
  and final report in memory without writing files.
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
[1/5] Scope | STARTED
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
[1/5] Scope | COMPLETED | research brief approved
[2/5] Research | STARTED
  [Supervisor 1/4] Decision | STARTED
  [Supervisor 1/4] Decision | COMPLETED | next task selected

================================================================================
SUPERVISOR DECISION 1/4
================================================================================
Decision: Start one worker

Title: <task title>
Research question: <focused question>
Evidence targets:
- <small observable evidence requirement>
================================================================================

  [Worker 1/4] <task title> | STARTED
    [Turn 1/6] Model | STARTED | 5 tools remaining
    [Tool 1/5] tavily_search_tool | STARTED
    [Tool 1/5] tavily_search_tool | COMPLETED
...
  [Worker 1/4] <task title> | COMPLETED | completed

================================================================================
AGENT RUN RESULT | WORKER 1 | <task title>
================================================================================
Status: completed
Termination reason: completed
Model turns: <used>/6
Tool calls: <used>/5
================================================================================

================================================================================
RESEARCH NOTES | WORKER 1 | <task title>
================================================================================
<worker research notes>
================================================================================

... another Supervisor decision, or a visible finish decision ...
... followed by COMBINED RESEARCH NOTES, DRAFT, and CRITIQUE blocks ...

[5/5] Revise | COMPLETED
================================================================================
FINAL REPORT
================================================================================
<final report>
================================================================================

[Run] Deep research | COMPLETED
```

Release 0.7.0 does not create a run directory or write application output files.
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
    → Research Supervisor observes ResearchState
        → selects one next task or finishes
        → AgentRunRequest for one isolated Research Worker
        → private AgentRunState with fresh history
            → tavily_search_tool
            → arxiv_search_tool
        → immutable AgentRunResult
        → application merges only completed results into ResearchState
        → Supervisor observes the updated state
    → Write
    → Critic
    → Revise
    → final report in the terminal
```

Scope first makes one synchronous Structured Outputs request for a Pydantic
`ClarificationAssessment`. It prints and collects at most three questions in one round,
then makes a second request for a complete `ResearchBrief`. Invalid local input repeats
only the current prompt. The user must approve the brief, cancel, or request one replacement
brief and approve that revision. No Supervisor or research-tool call begins before approval.

The application keeps a small `ResearchState` containing `approved_brief`, ordered
`worker_results`, and `stop_reason`. Each stateless Supervisor call sees a deterministic
rendering of that state and returns a list containing zero or one next task. An empty list
finishes Research; one task starts one isolated Research Worker with fresh ordered
Responses API history. The Worker executes requested functions and links every result with
its matching `call_id`; every expected outcome becomes an `AgentRunResult` with terminal status,
bounded termination reason, notes or error, and used-versus-limit turn and tool accounting. Only
completed results enter ResearchState, and only the approved brief and combined Worker notes
reach Write. Research uses automatic tool selection while
budget remains and `tool_choice="none"` after the fifth tool attempt. Each search returns at
most three entries, long result text is bounded before entering history, and Supervisor requests
use a fixed 4,000-token output limit. Worker requests use the configured model's default output
limit because Responses API output limits include both reasoning tokens and visible notes. Model
failures stop the workflow, while tool failures return to the active Worker so it can adapt within
its remaining budget. The official SDK client disables automatic retries so every failed model
request reaches the visible application failure boundary without a hidden repeated attempt.

## Planned Release Direction

The released 0.7.0 implementation is the current runnable baseline. Planned releases evolve
the same explicit runtime in this order:

| Version | Primary mechanism | Intended outcome |
| --- | --- | --- |
| 0.8.0 | Deep retrieval Worker | A Worker can select and read bounded content from sources returned by search. |
| 0.9.0 | Local open-weight Worker | The Worker can optionally use a local vLLM Responses-compatible model while hosted OpenAI stages remain available. |
| 0.10.0 | Long-horizon context sessions | A long Worker run crosses explicit in-memory summary boundaries instead of replaying unbounded history. |
| 0.11.0 | Bounded task graph and parallel workers | The Supervisor creates dependency-aware tasks and the application runs independent ready work concurrently. |
| 0.12.0 | Failure-aware adaptive orchestration | Failed tasks remain visible so the Supervisor can replace, narrow, or abandon them within hard limits. |
| 0.13.0 | Batch rollout runner | JSON or JSONL tasks generate multiple rollouts through the same Worker harness and result contract. |

OpenAI-hosted models remain useful as optional teachers, baselines, and acceptance references.
The project does not begin SFT or RL merely by adding a local inference path. Persistence,
structured trajectories, evaluation, evidence reliability, SFT, and RL remain later directions
selected from failures observed in real local-model and batch-rollout runs. See the
[roadmap](ROADMAP.md) for release boundaries and explicit exclusions.

## Project Structure

```text
.
├── deep_research.py       # Scope, Supervisor and Worker loops, report workflow, and CLI
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

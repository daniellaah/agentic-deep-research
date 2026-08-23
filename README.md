# Agentic Deep Research

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-Responses_API-412991?logo=openai&logoColor=white)](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)
[![uv](https://img.shields.io/badge/Package_Manager-uv-DE5FE9?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![CI](https://github.com/daniellaah/agentic-deep-research/actions/workflows/ci.yml/badge.svg)](https://github.com/daniellaah/agentic-deep-research/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A learning-oriented deep-research CLI that adds model and Agent mechanisms one observable
release at a time. Release 0.9.0 turns an initial question into an explicitly approved
ResearchBrief, lets a hosted Research Supervisor adaptively delegate bounded work to isolated
Research Workers, and runs the same explicit Worker harness with either an OpenAI-hosted model or
a local open-weight model served through a vLLM Responses-compatible endpoint. Each Worker can
search, deliberately select, and read bounded source content before returning through an
application-owned run contract. The planned harness sequence next adds long-horizon context
management, dependency-aware multi-agent scheduling, and batch rollouts before training begins.

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
- Lets the model search the web and arXiv, then read bounded relevant content from an eligible
  primary URL selected through an application-issued source ID.
- Runs at most four Workers sequentially with fresh application-owned Responses API history
  and independent hard model-turn, tool-call, and selected-source read limits.
- Represents every Worker as an immutable `AgentRunRequest`, private mutable `AgentRunState`, and
  immutable `AgentRunResult` with visible model source, model name, status, termination reason,
  and turn, tool, and read usage.
- Selects one OpenAI-hosted or local vLLM Worker policy for the complete run while keeping Scope,
  Supervisor, Write, Critic, and Revise on the hosted OpenAI model.
- Shows Scope decisions, pending briefs, approval, Supervisor decisions, Worker boundaries,
  Agent turns, tool selections, every intermediate result, report stages, and final report
  in consistently formatted terminal sections.
- Keeps scoping data, the approved brief, ResearchState, research notes, draft, critique,
  and final report in memory without writing files.
- Loads hosted OpenAI, optional local Worker, Tavily Search and Extract, and model configuration
  from environment variables.
- Keeps the workflow in `deep_research.py`, detailed model instructions in
  `agent_instructions.py`, and tool details in `agent_tools.py`.

## Quick Start

### Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key and access to the model you want to use
- A [Tavily API](https://www.tavily.com/) key
- For the optional local Worker, an independently running vLLM 0.20.2-compatible server or the
  documented vLLM-Metal Apple Silicon profile

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
WORKER_BACKEND=openai
```

Keep `.env` private and never commit it.

To use the local compatibility reference, start vLLM outside this application:

```bash
vllm serve Qwen/Qwen3-1.7B \
  --reasoning-parser qwen3 \
  --structured-outputs-config.backend xgrammar \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --api-key local-worker
```

Then select it in `.env`:

```dotenv
WORKER_BACKEND=local
LOCAL_WORKER_BASE_URL=http://127.0.0.1:8000/v1
LOCAL_WORKER_API_KEY=local-worker
LOCAL_WORKER_MODEL_NAME=Qwen/Qwen3-1.7B
```

The application does not install, launch, stop, or probe the vLLM server. Other model and parser
combinations must satisfy the same complete Responses function-tool loop before use.

On Apple Silicon, release 0.9.0 has also completed its real local-service smoke scenario with
the operator-managed vLLM-Metal runtime and `mlx-community/Qwen3.5-9B-4bit`:

```bash
vllm serve mlx-community/Qwen3.5-9B-4bit \
  --served-model-name qwen3.5-9b-4bit \
  --host 127.0.0.1 \
  --port 8001 \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.5 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --api-key local-worker
```

```dotenv
WORKER_BACKEND=local
LOCAL_WORKER_BASE_URL=http://127.0.0.1:8001/v1
LOCAL_WORKER_API_KEY=local-worker
LOCAL_WORKER_MODEL_NAME=qwen3.5-9b-4bit
```

The exact tested runtime versions and acceptance result are recorded in the
[v0.9.0 release specification](specs/v0.9.0-local-open-weight-worker.md#apple-silicon-manual-verification-profile).

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
[Worker model] Research policy | CONFIGURED | <openai|local>, <model name>
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
    [Turn 1/6] Model | STARTED | 5 tools, 2 source reads remaining
    [Tool 1/5] tavily_search_tool | STARTED
      [Sources] Worker registry | UPDATED | 3 new, 3 available
    [Tool 1/5] tavily_search_tool | COMPLETED
    [Turn 2/6] Model | STARTED | 4 tools, 2 source reads remaining
    [Tool 2/5] read_source_tool | STARTED
      [Read 1/2] S1 (<source host>) | STARTED
      [Read 1/2] S1 (<source host>) | COMPLETED | <characters> characters, 1 read remaining
    [Tool 2/5] read_source_tool | COMPLETED
...
  [Worker 1/4] <task title> | COMPLETED | completed

================================================================================
AGENT RUN RESULT | WORKER 1 | <task title>
================================================================================
Model source: <openai|local>
Model: <model name>
Status: completed
Termination reason: completed
Model turns: <used>/6
Tool calls: <used>/5
Source reads: <used>/2
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

Release 0.9.0 does not create a run directory or write application output files.
Structured tracing is also not part of this release.

## Configuration

| Variable | Required | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | Always | Hosted Scope, Supervisor, report, and hosted Worker authentication. |
| `MODEL_NAME` | Always | Hosted model supporting Structured Outputs and custom function tools. |
| `TAVILY_API_KEY` | Always | API key used for Tavily web search and selected-source extraction. |
| `WORKER_BACKEND` | No | `openai` by default; use `local` for a vLLM Worker. |
| `LOCAL_WORKER_BASE_URL` | Local only | vLLM OpenAI-compatible `/v1` base URL. |
| `LOCAL_WORKER_API_KEY` | Local only | API key sent only to the local endpoint. |
| `LOCAL_WORKER_MODEL_NAME` | Local only | Exact model identifier served by vLLM. |

The CLI automatically loads `.env` from the repository root. Values already present in
the process environment take precedence over values in `.env`.

## How It Works

```text
Research question
    → select one Worker model source for the run
    → bounded clarification assessment
    → optional clarification answers
    → validated ResearchBrief
    → explicit approval or one revision and final approval
    → Research Supervisor observes ResearchState
        → selects one next task or finishes
        → AgentRunRequest with selected model source and model name
        → private AgentRunState with fresh history
            → OpenAI-hosted model or local vLLM Responses endpoint
            → tavily_search_tool
            → arxiv_search_tool
            → application-issued per-Worker source IDs
            → read_source_tool for one selected eligible source
        → immutable AgentRunResult with model source and model name
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

The application constructs one hosted OpenAI SDK client for every run. With
`WORKER_BACKEND=openai`, the Worker reuses that client and `MODEL_NAME`. With
`WORKER_BACKEND=local`, only Worker model turns use a second SDK client configured by
`LOCAL_WORKER_*`; the local and OpenAI API keys are never substituted for one another. The
backend is fixed before Scope and there is no automatic fallback, server management, or provider
adapter.

The application keeps a small `ResearchState` containing `approved_brief`, ordered
`worker_results`, and `stop_reason`. Each stateless Supervisor call sees a deterministic
rendering of that state and returns a list containing zero or one next task. An empty list
finishes Research; one task starts one isolated Research Worker with fresh ordered
Responses API history. The Worker executes requested functions and links every result with
its matching `call_id`; every expected outcome becomes an `AgentRunResult` with model source,
model name, terminal status, bounded termination reason, notes or error, and used-versus-limit
turn, tool, and source-read accounting. Search results receive source IDs only for eligible
primary HTTP(S) URLs in the active Worker. The read tool accepts an ID and focused query rather
than a model-supplied URL, sends one resolved destination to Tavily Extract, and bounds selected
content to 6,000 characters. Each Worker may attempt at most two reads, and every read also
consumes one of its five total tool calls. Only completed results enter ResearchState, and only
the approved brief and combined Worker notes reach Write. Research uses automatic tool selection
while budget remains and `tool_choice="none"` after the fifth tool attempt. Each search returns
at most three entries,
long result text is bounded before entering history, and Supervisor requests use a fixed
4,000-token output limit. Worker requests use the configured model's default output limit because
Responses API output limits include both reasoning tokens and visible notes. Model failures stop
the workflow, while tool failures return to the active Worker so it can adapt within its remaining
budget. Both official SDK clients disable automatic retries so every failed model request reaches
the visible application failure boundary without a hidden repeated attempt. vLLM incompatibility
is reported as the existing Worker model error; the application does not translate response
items or retry against OpenAI.

## Planned Release Direction

The 0.9.0 implementation is the current runnable baseline. Planned releases evolve
the same explicit runtime in this order:

| Version | Primary mechanism | Intended outcome |
| --- | --- | --- |
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

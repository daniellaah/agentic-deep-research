# Agentic Deep Research

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-Responses_API-412991?logo=openai&logoColor=white)](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)
[![uv](https://img.shields.io/badge/Package_Manager-uv-DE5FE9?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![CI](https://github.com/daniellaah/agentic-deep-research/actions/workflows/ci.yml/badge.svg)](https://github.com/daniellaah/agentic-deep-research/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A learning-oriented deep-research CLI that adds model and Agent mechanisms one observable
release at a time. The current runtime turns an initial question into an explicitly approved
ResearchBrief, lets a Research Supervisor adaptively delegate bounded work to isolated Research
Workers, and uses one run-level OpenAI or DeepSeek provider for every LLM call through the
Responses API. Each Worker can search, deliberately select, and read bounded source content
across as many as three bounded context sessions before returning through an application-owned
run contract. Release 0.10.0 makes long-horizon state replacement explicit before later
multi-agent scheduling and batch rollouts.

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
- Runs at most four Workers sequentially with independent hard model-turn, tool-call,
  selected-source read, context-session, and context-summary limits.
- Replaces an active Worker history at a visible 12,000-token projection boundary with a bounded,
  Pydantic-validated, application-rendered `ResearchStateSummary` while preserving run budgets and
  source identity.
- Represents every Worker as an immutable `AgentRunRequest`, private mutable `AgentRunState`, and
  immutable `AgentRunResult` with visible LLM provider, model name, status, termination reason,
  and turn, tool, and read usage.
- Selects one OpenAI or DeepSeek provider and one environment-configured model for every Scope,
  Supervisor, Worker, context-summary, Write, Critic, and Revise model call in the run.
- Shows Scope decisions, pending briefs, approval, Supervisor decisions, Worker boundaries,
  Agent turns, tool selections, every intermediate result, report stages, and final report
  in consistently formatted terminal sections.
- Keeps scoping data, the approved brief, ResearchState, research notes, draft, critique,
  and final report in memory without writing files.
- Loads provider selection, the selected provider's API key and model name, and Tavily Search and
  Extract configuration from environment variables.
- Keeps the workflow in `deep_research.py`, detailed model instructions in
  `agent_instructions.py`, and tool details in `agent_tools.py`.

## Quick Start

### Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- An API key and model name for the selected OpenAI or DeepSeek provider
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

For an OpenAI run, set:

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL_NAME=your_openai_model_name
TAVILY_API_KEY=your_tavily_api_key
```

Keep `.env` private and never commit it.

For a DeepSeek V4 Flash acceptance run, set:

```dotenv
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_MODEL_NAME=deepseek-v4-flash
TAVILY_API_KEY=your_tavily_api_key
```

The unselected provider's variables may remain empty. DeepSeek uses its official hosted base URL;
the model ID still comes from `DEEPSEEK_MODEL_NAME` rather than a source-code default.

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
[LLM] Provider and model | CONFIGURED | <openai|deepseek>, <model name>
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
    [Context 1/3] Session | STARTED | fresh task history
    [Turn 1/15] Model | STARTED | 10 tools, 4 source reads remaining
    [Tool 1/10] tavily_search_tool | STARTED
      [Sources] Worker registry | UPDATED | 3 new, 3 available
    [Tool 1/10] tavily_search_tool | COMPLETED
    [Turn 2/15] Model | STARTED | 9 tools, 4 source reads remaining
    [Tool 2/10] read_source_tool | STARTED
      [Read 1/4] S1 (<source host>) | STARTED
      [Read 1/4] S1 (<source host>) | COMPLETED | <characters> characters, 3 reads remaining
    [Tool 2/10] read_source_tool | COMPLETED
    [Context 1/3] Boundary | REQUIRED | projected <tokens>; trigger 12,000
    [Summary 1/2] Research state | STARTED

================================================================================
RESEARCH STATE SUMMARY | WORKER 1 | SESSION 1 -> 2
================================================================================
<validated completed work, resolved source URLs, gaps, and next actions>
================================================================================

    [Summary 1/2] Research state | COMPLETED | validated
    [Context 2/3] Session | STARTED | validated summary
...
  [Worker 1/4] <task title> | COMPLETED | completed

================================================================================
AGENT RUN RESULT | WORKER 1 | <task title>
================================================================================
LLM provider: <openai|deepseek>
Model: <model name>
Status: completed
Termination reason: completed
Model turns: <used>/15
Tool calls: <used>/10
Source reads: <used>/4
Context sessions: <used>/3
Context summaries: <used>/2
Peak observed input tokens: <count>
Peak projected input tokens: <count>/20000
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

The current development release does not create a run directory or write application output
files. Structured tracing is also not part of this release.

## Configuration

| Variable | Required | Description |
| --- | --- | --- |
| `LLM_PROVIDER` | Always | One global provider for the run; accepts exactly `openai` or `deepseek`. |
| `OPENAI_API_KEY` | OpenAI only | OpenAI authentication for every LLM stage in an OpenAI run. |
| `OPENAI_MODEL_NAME` | OpenAI only | OpenAI model used by every LLM stage in an OpenAI run. |
| `DEEPSEEK_API_KEY` | DeepSeek only | DeepSeek authentication for every LLM stage in a DeepSeek run. |
| `DEEPSEEK_MODEL_NAME` | DeepSeek only | DeepSeek model used by every LLM stage in a DeepSeek run. |
| `TAVILY_API_KEY` | Always | API key used for Tavily web search and selected-source extraction. |

The CLI automatically loads `.env` from the repository root. Values already present in
the process environment take precedence over values in `.env`.

## How It Works

```text
Research question
    → select one LLM provider and model for the complete run
    → bounded clarification assessment
    → optional clarification answers
    → validated ResearchBrief
    → explicit approval or one revision and final approval
    → Research Supervisor observes ResearchState
        → selects one next task or finishes
        → AgentRunRequest with selected LLM provider and model name
        → private AgentRunState with run-level budgets and source registry
            → context session with fresh Responses history
                → OpenAI-hosted model or hosted DeepSeek Responses endpoint
                → synchronous ordered search or selected-source read calls
                → projected next input reaches 12,000 tokens
                → validated application-owned ResearchStateSummary
            → replacement session with only task, brief, budgets, and summary
            → complete notes or cross one final summary boundary
        → immutable AgentRunResult with LLM provider and model name
        → application merges only completed results into ResearchState
        → Supervisor observes the updated state
    → Write
    → Critic
    → Revise
    → final report in the terminal
```

Scope first makes one synchronous Structured Outputs request for a Pydantic
`ClarificationAssessment`. It prints and collects at most three questions in one round,
then makes a second request for a complete `ResearchBrief`. Invalid terminal input repeats
only the current prompt. The user must approve the brief, cancel, or request one replacement
brief and approve that revision. No Supervisor or research-tool call begins before approval.

The application reads `LLM_PROVIDER` before Scope and constructs exactly one SDK client. An
OpenAI run uses `OPENAI_API_KEY` and `OPENAI_MODEL_NAME`; a DeepSeek run uses
`DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL_NAME`, and the official `https://api.deepseek.com` base URL.
That same client and model serve every LLM stage. The unselected provider's variables are not
required or substituted, and there is no per-stage routing, automatic fallback, or provider
adapter.

The application keeps a small `ResearchState` containing `approved_brief`, ordered
`worker_results`, and `stop_reason`. Each stateless Supervisor call sees a deterministic
rendering of that state and returns a list containing zero or one next task. An empty list
finishes Research; one task starts one isolated Research Worker with fresh ordered
Responses API history. The Worker executes requested functions and links every result with
its matching `call_id`; every expected outcome becomes an `AgentRunResult` with LLM provider,
model name, terminal status, bounded termination reason, notes or error, and run-level budget and
context accounting. Search results receive source IDs only for eligible
primary HTTP(S) URLs in the active Worker. The read tool accepts an ID and focused query rather
than a model-supplied URL, sends one resolved destination to Tavily Extract, and bounds selected
content to 6,000 characters. Each Worker may use 15 model turns, 10 tool attempts, four selected-
source read attempts, three sessions, and two summaries. Every ordinary Worker request disables
parallel tool calls as a provider hint. If an endpoint still returns multiple calls, the
application executes budget-permitted calls synchronously in response order and appends a linked
result for every call.

After each tool-call batch, the application projects the next input from exact response usage plus
a conservative byte estimate for every linked function output. A projection of at least 12,000
tokens requires a summary boundary; more than 20,000 tokens stops before summary. The same selected
model creates a bounded `ResearchStateSummary`. The application validates source IDs and evidence
levels, renders URLs from its own registry, then replaces the old history with one new input
containing the original task, approved brief, remaining budgets, and validated summary. Counters,
the source registry, and successfully read source IDs survive; the replaced history does not.

Only completed results enter ResearchState, and only the approved brief and combined Worker notes
reach Write. Research uses automatic tool selection while budget remains and
`tool_choice="none"` after the tenth tool attempt. Worker requests use the configured model's
default output limit; summary requests use an 8,000-token limit. Model failures stop the workflow,
while tool failures return to the active Worker so it can adapt within its remaining budget. The
selected SDK client disables automatic retries. DeepSeek incompatibility is reported at the
existing model-error boundary without response translation or fallback to OpenAI.

## Planned Release Direction

The v0.10.0 development runtime implements global OpenAI-or-DeepSeek provider selection and
application-owned long-horizon context sessions. Real endpoint acceptance remains pending before
the release can be marked verified. Later releases evolve the same explicit runtime in this order:

| Version | Primary mechanism | Intended outcome |
| --- | --- | --- |
| 0.10.0 | Long-horizon context sessions | A long Worker run crosses explicit in-memory summary boundaries instead of replaying unbounded history. |
| 0.11.0 | Bounded task graph and parallel workers | The Supervisor creates dependency-aware tasks and the application runs independent ready work concurrently. |
| 0.12.0 | Failure-aware adaptive orchestration | Failed tasks remain visible so the Supervisor can replace, narrow, or abandon them within hard limits. |
| 0.13.0 | Batch rollout runner | JSON or JSONL tasks generate multiple rollouts through the same Worker harness and result contract. |

OpenAI-hosted models remain useful as teachers, baselines, and acceptance references.
Persistence, structured trajectories, evaluation, evidence reliability, SFT, and RL remain later
directions selected from failures observed in real hosted-model and batch-rollout runs. See the
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

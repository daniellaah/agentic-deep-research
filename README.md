# Agentic Deep Researcher

A course-driven project for learning agentic AI by building a deep research
workflow from scratch with Python and the OpenAI Responses API.

The project intentionally avoids orchestration frameworks such as LangChain and
LangGraph. The goal is to make planning, tool calling, observations, execution
state, and reflection visible in ordinary Python code.

## Learning Path

| Version | Notebook | Focus | Status |
| --- | --- | --- | --- |
| V1 | [`01_v1_static_multi_agent_workflow.ipynb`](notebooks/01_v1_static_multi_agent_workflow.ipynb) | Structured planning, tool use, deterministic execution, and reflection | Stable ([`v0.1.0`](https://github.com/daniellaah/agentic-deepresearch/tree/v0.1.0)) |
| V2 | [`02_v2_adaptive_deep_research.ipynb`](notebooks/02_v2_adaptive_deep_research.ipynb) | Scoping, adaptive supervision, isolated workers, compression, and parallel research | In development |

Each notebook is independently runnable. V1 remains a stable teaching artifact,
while V2 starts from the same executable baseline and replaces the static
research workflow incrementally. V2 currently includes structured clarification,
human-in-the-loop conversation state, and research brief generation.

## V1 Architecture

The current workflow uses five responsibilities:

```text
User topic
   |
   v
Planner Agent -- creates a validated, structured TaskPlan
   |
   v
Executor Agent -- dispatches steps and records status/results/errors
   |
   +--> Research Agent -- searches arXiv and the web through tool calling
   +--> Writer Agent   -- drafts a source-backed Markdown report
   +--> Editor Agent   -- reviews the draft and requests concrete revisions
   +--> Writer Agent   -- produces the final publication-ready report
```

The default four-step route is:

```text
research_agent -> writer_agent -> editor_agent -> writer_agent
```

The Editor-to-Writer transition is the reflection pattern: the first draft is
observed and criticized before the final revision is produced.

## V1 Concepts

- OpenAI Responses API basics
- Draft-Critic-Revision reflection
- Function tool definitions and tool execution
- The tool-call/observation loop
- Structured planning with Pydantic schemas
- Deterministic multi-agent routing
- Execution status and error recording
- Final report validation and Markdown persistence

## Project Layout

```text
.
├── src/
│   └── agentic_deepresearch/
│       ├── __init__.py            # Public package interface
│       ├── schemas.py             # Shared Pydantic data contracts
│       └── tools.py               # arXiv and Tavily tools
├── notebooks/
│   ├── 01_v1_static_multi_agent_workflow.ipynb
│   └── 02_v2_adaptive_deep_research.ipynb
├── .env.example                   # Environment variable template
├── pyproject.toml                 # Python dependencies and tool configuration
└── final_report.md                # Generated after a successful run
```

Reusable code uses an installable `src` package. The notebooks therefore use
absolute imports that do not depend on the Jupyter working directory:

```python
from agentic_deepresearch import schemas
from agentic_deepresearch.tools import arxiv_search_tool
```

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key
- A Tavily API key for general web search

## Setup

Install the environment:

```bash
uv sync --dev
```

This installs both the dependencies and the local `agentic_deepresearch`
package into `.venv`.

Create the local environment file:

```bash
cp .env.example .env
```

Configure the required values:

```dotenv
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
MODEL_NAME=your_responses_api_model
```

Never commit `.env` or real API keys.

## Run the Notebook

Start JupyterLab with the project environment:

```bash
uv run jupyter lab
```

Choose a notebook from the learning path and run its cells in order. Start with
`01_v1_static_multi_agent_workflow.ipynb` to learn the complete fixed workflow,
then continue with `02_v2_adaptive_deep_research.ipynb` as the adaptive design is
developed.

The V1 entry point is:

```python
final_report = run_deep_research(
    topic="What are the recent developments in reliable LLM agent design?",
    output_path="final_report.md",
    max_steps=4,
    verbose=True,
)
```

This function:

1. creates and validates a structured plan;
2. executes each step with the assigned agent;
3. passes the configured report word limits to the Writer as runtime requirements;
4. records completed or failed status for every attempted step;
5. validates the final report;
6. saves `final_report.md` only after validation succeeds.

API and search calls can incur cost and take time. The notebook therefore keeps
individual agent test cells commented out by default.

## V1 Plan Invariants

The deterministic validator requires:

- consecutive, one-based step IDs;
- no more than the configured maximum number of steps;
- at least one research step before the initial draft;
- exactly one editor step;
- an initial Writer step before editorial review;
- the Editor immediately followed by the final Writer;
- no additional polishing or finalization step after the final revision.

These constraints prevent the Planner from creating an ambiguous route such as
two consecutive final Writer steps.

## Execution Records

The Executor returns an in-memory history. A successful record looks like:

```python
{
    "step_id": 1,
    "agent": "research_agent",
    "task": "...",
    "status": "completed",
    "result": "...",
    "error": None,
}
```

When a step fails, its status is `failed`, `result` is `None`, and `error`
contains the exception type and message. Execution stops because later steps
depend on earlier results.

## Final Report Quality Gate

Before a report is saved, the entry point checks that it:

- is not empty;
- contains a top-level Markdown title;
- contains a `References` or `Sources` section;
- includes source URLs;
- stays within the configured word range.

This validation is deterministic. It does not prove that every factual claim or
citation is correct. Conversational framing is controlled by the Writer prompt,
while source verification remains an important future extension.

## V1 Limitations

- Steps execute sequentially.
- The workflow stops at the first failed step.
- There is no replanning or retry policy yet.
- Sources are collected and cited but not independently fact-checked.
- Execution history is in memory only.
- Final report validation checks format and basic constraints, not factual truth.

## Roadmap

- Add minimal unit tests for schemas and deterministic validators.
- Add source deduplication and citation verification.
- Replace the static research plan with an observation-driven Supervisor loop.
- Add context-isolated Research Workers and evidence compression.
- Add bounded parallel research with `asyncio`.
- Reuse V1 reflection and deterministic quality gates in the V2 report pipeline.
- Compare the completed pure-Python architecture with a framework-based version.

# Agentic Deep Research

Agentic Deep Research is a Python project for building a reliable deep research agent with testable workflows and clear model and tool boundaries.

The current engine uses the OpenAI Responses API inside a small, explicit agent harness. An adaptive planner creates research questions, a deterministic supervisor enforces global limits, independent workers investigate questions in bounded parallel batches, and an evidence store carries only compact citation-linked context between batches.

The result contains the research plan, cited findings, normalized evidence, reported conflicts, full worker artifacts, observable control and web actions, stop reason, and token usage. Private model reasoning is never stored.

Evidence confidence and source quality are lightweight harness signals: matching claims from distinct URLs are treated as corroborated, and selected institutional domains receive a primary-source hint. They do not prove that a citation supports a claim; semantic citation verification is a later reliability layer.

## Architecture

The core flow is:

```text
ResearchRequest
  -> Adaptive Planner
  -> Supervisor (budget, scheduling, stop conditions, replanning)
  -> Independent Research Workers (Responses API + web_search)
  -> Evidence Store (deduplication, source quality, confidence, conflicts)
  -> Bounded Context Builder
  -> Cited ResearchResult + full artifacts
```

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
MODEL_NAME=
JUDGE_MODEL_NAME=
```

Do not commit `.env` or API keys.

## Run a Research Task

Research a topic with a global limit of eight web-tool calls, at most four worker runs, two concurrent workers, and 20,000 output tokens per model response:

```bash
uv run deep-research "What makes a research agent reliable?"
```

Choose the report language and inspect the observable search trace:

```bash
uv run deep-research \
  "目前主流 Research Agent 使用了哪些可靠性设计？" \
  --language Chinese \
  --max-tool-calls 8 \
  --max-output-tokens 20000 \
  --max-research-steps 4 \
  --max-parallel-workers 2 \
  --max-context-chars 8000 \
  --show-trace
```

The planner chooses the highest-value subquestions. The supervisor decides what runs next and stops at the global tool or research-step limits. `--max-context-chars` prevents previous findings from growing every later prompt; full outputs remain available as artifacts.

The same workflow is available from Python:

```python
from openai import OpenAI

from agentic_deep_research import ResearchBudget, ResearchRequest, run_research
from agentic_deep_research.planning import OpenAIAdaptivePlanner
from agentic_deep_research.runner import OpenAIAgentRunner

client = OpenAI()
runner = OpenAIAgentRunner(client=client, model="your-model")
planner = OpenAIAdaptivePlanner(client=client, model="your-model")
request = ResearchRequest(
    topic="What makes a research agent reliable?",
    budget=ResearchBudget(
        max_tool_calls=8,
        max_output_tokens=20_000,
        max_research_steps=4,
        max_parallel_workers=2,
        max_context_chars=8_000,
    ),
)
result = run_research(request, runner=runner, planner=planner)

print(result.report)
print(result.plan)
print(result.evidence)
print(result.sources)
print(result.trace)
```

Web search requests consume API tokens and built-in tool calls. Unit tests use fake responses and do not make paid API calls.

## Benchmarks

Install the isolated benchmark dependencies:

```bash
uv sync --group benchmark
```

Download the official FRAMES cases and run a fixed 10-case development slice:

```bash
uv run --group benchmark deep-research-eval download frames \
  --output .benchmarks/data/frames.jsonl

uv run --group benchmark deep-research-eval run frames \
  --data .benchmarks/data/frames.jsonl \
  --output .benchmarks/runs/frames-dev10.jsonl \
  --limit 10
```

Download only the first 10 encrypted BrowseComp-Plus cases through streaming and run the same development protocol:

```bash
uv run --group benchmark deep-research-eval download browsecomp-plus \
  --limit 10 \
  --output .benchmarks/data/browsecomp-plus-dev10.jsonl

uv run --group benchmark deep-research-eval run browsecomp-plus \
  --data .benchmarks/data/browsecomp-plus-dev10.jsonl \
  --output .benchmarks/runs/browsecomp-plus-dev10.jsonl \
  --limit 10
```

Each case is appended to JSONL immediately. Re-running the same command resumes by case ID, while changing model or budget configuration raises an error instead of mixing incomparable results. Dataset files and raw runs are Git-ignored.

These commands use the adaptive live-web development protocol. A leaderboard-comparable BrowseComp-Plus run must instead use its fixed corpus, retriever, document IDs, and official judge. See the [recorded pre-adaptive development baselines](docs/baselines.md) for historical results and limitations.

## Quality Checks

Run the test suite:

```bash
uv run pytest
```

Run the code-quality checks:

```bash
uv run ruff check src tests
```

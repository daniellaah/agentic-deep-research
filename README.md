# Agentic Deep Research

Agentic Deep Research is a Python project for building a reliable deep research agent with testable workflows and clear model and tool boundaries.

The current baseline uses the OpenAI Responses API to run a bounded agentic web-search loop. It returns a cited report together with the consulted sources, observable research actions, stop reason, and token usage.

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

Research a topic with the default limits of eight web-tool calls and 20,000 output tokens:

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
  --show-trace
```

The model decides when it has enough evidence, while `--max-tool-calls` and `--max-output-tokens` provide hard stopping boundaries. The trace records web actions exposed by the Responses API; private model reasoning is not stored.

The same workflow is available from Python:

```python
from openai import OpenAI

from agentic_deep_research import ResearchBudget, ResearchRequest, run_research
from agentic_deep_research.runner import OpenAIAgentRunner

runner = OpenAIAgentRunner(client=OpenAI(), model="your-model")
request = ResearchRequest(
    topic="What makes a research agent reliable?",
    budget=ResearchBudget(max_tool_calls=8, max_output_tokens=20_000),
)
result = run_research(request, runner=runner)

print(result.report)
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

These commands use a live-web development protocol. A leaderboard-comparable BrowseComp-Plus run must instead use its fixed corpus, retriever, document IDs, and official judge. See the [recorded development baselines](docs/baselines.md) for results and limitations.

## Quality Checks

Run the test suite:

```bash
uv run pytest
```

Run the code-quality checks:

```bash
uv run ruff check src tests
```

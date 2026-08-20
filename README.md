# Agentic Deep Research

Agentic Deep Research is a Python project for building a reliable deep research agent with testable workflows and clear model and tool boundaries.

The current engine uses the OpenAI Responses API inside a small, explicit agent harness. An adaptive planner creates research questions, a deterministic supervisor enforces global limits and evidence sufficiency, and independent workers investigate questions in bounded parallel batches. A source-diverse `ContextPack` carries only complete citation-linked records between batches instead of truncating arbitrary text. A writer then synthesizes a report from controlled source markers, while a critic and citation verifier drive bounded gap-search and revision rounds. A durable runtime checkpoints every provider-facing operation so an interrupted run can resume without repeating work that was already saved.

The result contains the research plan, cited findings, a versioned evidence ledger, reported conflicts, draft and revision artifacts, citation-support judgments, observable control and web actions, stop reason, and token usage. Private model reasoning is never stored.

Every canonical source and evidence record receives a stable ID. Tracking parameters and URL fragments are removed before source deduplication; evidence retains its research-question and artifact provenance. Matching claims from distinct sources are treated as corroborated, but a domain name never proves source quality. Citation-derived evidence also does not pretend that a generated claim is a verbatim source excerpt. The separate citation verifier reopens paired source URLs and returns a semantic `supported`, `unsupported`, or `uncertain` judgment.

## Architecture

The core flow is:

```text
ResearchRequest
  -> Durable Runtime (checkpoint, retry, approval, cancellation)
  -> Adaptive Planner
  -> Supervisor (budget, scheduling, deterministic sufficiency, replanning)
  -> Independent Research Workers (Responses API + web_search)
  -> Evidence Ledger (stable identity, provenance, deduplication, conflicts)
  -> ContextPack (whole evidence records, conflicts, selection metadata)
  -> Report Writer (evidence-bound [E1], [E2], ... markers)
  -> Critic + Citation Verifier
  -> Gap Search + Reviser (bounded quality loop)
  -> Cited ResearchResult + quality metadata + full artifacts
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

Research a topic with a global limit of eight research web-tool calls, at most four worker runs, two concurrent workers, two citation-verification web-tool calls, two report revisions, and 20,000 output tokens per model response:

```bash
uv run deep-research "What makes a research agent reliable?"
```

The command prints a generated `run_id` and saves its checkpoint in `.research-runs/checkpoints.sqlite3`. Give an important run a stable ID and require plan approval before the first web-research call:

```bash
uv run deep-research \
  "What makes a research agent reliable?" \
  --run-id reliability-study \
  --require-approval

uv run deep-research --approve reliability-study
```

Resume a failed run, reject a pending plan, or request cooperative cancellation:

```bash
uv run deep-research --resume reliability-study
uv run deep-research --reject reliability-study --reason "Plan is too broad"
uv run deep-research --cancel reliability-study --reason "No longer needed"
```

Completed planner, worker, writer, critic, verifier, and reviser calls are replayed from the checkpoint instead of being sent again. Transient connection, timeout, rate-limit, and server failures receive bounded exponential-backoff retries. An execution lease and heartbeat prevent two local processes from resuming the same run concurrently. Cancellation cannot forcibly abort an HTTP request already in flight; it saves that response and prevents the next external operation from starting.

There is one unavoidable exactly-once boundary: if a process stops after the provider accepted or completed a request but before the completed effect was committed, the checkpoint contains `started` and the outcome is ambiguous. Resume pauses by default instead of silently paying for a duplicate call. Only after confirming that the original process has stopped should you run:

```bash
uv run deep-research --resume reliability-study --retry-ambiguous
```

That explicit retry can repeat the provider call. A lease left by a force-killed process expires automatically; a normally exiting process releases it immediately.

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
  --max-verification-tool-calls 2 \
  --max-revision-rounds 2 \
  --show-trace
```

The planner chooses the highest-value subquestions. After each complete research round,
the supervisor deterministically checks that every active question has usable evidence
from a completed worker and that `min_sources` distinct canonical sources are actually
bound to evidence. Merely visiting a URL does not satisfy the source minimum. It then
either continues, stops at a global tool or research-step boundary, or reports a
specific stalled state such as `no_new_evidence` or `no_new_questions`.

`--max-context-chars` prevents previous findings from growing every later prompt.
Context selection favors the active question, verified or unverified usable evidence,
corroboration, and source diversity. Evidence and conflict entries are complete JSONL
records; an oversized record is omitted and counted instead of being cut in half. Full
worker outputs remain available as artifacts. The verifier has its own web-tool budget
because reopening cited pages is a separate reliability step. The revision limit
guarantees that critique, gap search, and rewriting cannot loop forever.

The same workflow is available from Python:

```python
from pathlib import Path

from openai import OpenAI

from agentic_deep_research import (
    ResearchBudget,
    ResearchRequest,
    ResearchRuntime,
    SQLiteCheckpointStore,
)
from agentic_deep_research.planning import OpenAIAdaptivePlanner
from agentic_deep_research.reporting import OpenAIReportAgent
from agentic_deep_research.runner import OpenAIAgentRunner

client = OpenAI(max_retries=0)
runner = OpenAIAgentRunner(client=client, model="your-model")
planner = OpenAIAdaptivePlanner(client=client, model="your-model")
report_agent = OpenAIReportAgent(client=client, model="your-model")
request = ResearchRequest(
    topic="What makes a research agent reliable?",
    budget=ResearchBudget(
        max_tool_calls=8,
        max_output_tokens=20_000,
        max_research_steps=4,
        max_parallel_workers=2,
        max_context_chars=8_000,
        max_verification_tool_calls=2,
        max_revision_rounds=2,
    ),
)
runtime = ResearchRuntime(
    store=SQLiteCheckpointStore(Path(".research-runs/checkpoints.sqlite3")),
    runner=runner,
    planner=planner,
    report_agent=report_agent,
)
outcome = runtime.start(request, run_id="reliability-study")
result = outcome.result

if result is not None:
    print(result.report)
    print(result.plan)
    print(result.evidence)
    print(result.ledger)
    print(result.sources)
    print(result.citation_checks)
    print(result.revision_count)
    print(result.trace)
```

`OpenAI(max_retries=0)` is intentional here: the durable runtime owns and records retries, avoiding a hidden SDK retry layer. The lower-level `run_research(...)` function remains available for short, stateless calls that do not need persistence or operational controls.

Checkpoints contain the complete report, evidence, source URLs, traces, and error messages. Treat the SQLite file as potentially sensitive application data. The local store creates it with owner-only file permissions, and `.gitignore` prevents accidental repository commits; deployment still needs normal backup, access-control, and retention policies.

Web search requests consume API tokens and built-in tool calls. Unit tests use fake responses and do not make paid API calls.

## Benchmarks

Install the isolated benchmark dependencies:

```bash
uv sync --group benchmark
```

Download the official FRAMES cases and start a named, resumable 10-case live-web
experiment:

```bash
uv run --group benchmark deep-research-eval download frames \
  --output .benchmarks/data/frames.jsonl

uv run --group benchmark deep-research-eval run frames \
  --data .benchmarks/data/frames.jsonl \
  --experiment-dir .benchmarks/experiments/frames-dev10 \
  --experiment-id frames-dev10-live-web-v3 \
  --limit 10
```

Download only the first 10 encrypted BrowseComp-Plus cases through streaming and
run the same development protocol:

```bash
uv run --group benchmark deep-research-eval download browsecomp-plus \
  --limit 10 \
  --output .benchmarks/data/browsecomp-plus-dev10.jsonl

uv run --group benchmark deep-research-eval run browsecomp-plus \
  --data .benchmarks/data/browsecomp-plus-dev10.jsonl \
  --experiment-dir .benchmarks/experiments/browsecomp-plus-dev10 \
  --experiment-id browsecomp-plus-dev10-live-web-v3 \
  --limit 10
```

Each experiment directory contains an immutable `manifest.json`, durable research
checkpoints, privacy-conscious case records, and an aggregate summary. The manifest
binds the selected case content, model, research budget, retry policy, search
protocol, citation policy, and grader version. Re-running the same command reuses a
completed case, resumes an interrupted research case, or retries only the judge when
research already completed. Changing a bound setting raises an error instead of
mixing incomparable results.

Replay already checkpointed research without running the research agent again:

```bash
uv run --group benchmark deep-research-eval replay frames \
  --data .benchmarks/data/frames.jsonl \
  --experiment-dir .benchmarks/experiments/frames-dev10
```

Replay of an experiment originally configured with `--judge exact` and no OpenAI
report judge is fully local. If the saved manifest uses an OpenAI answer judge or
report judge, replay can still make grader calls for cases whose judgment is missing;
it never makes research calls. Provider errors remain distinct from wrong answers,
and summaries report both total-slice accuracy and accuracy among completed
judgments.

For deterministic retrieval experiments, provide a JSONL corpus whose records contain
`id`, `title`, `text`, and an optional `url`:

```bash
uv run --group benchmark deep-research-eval run frames \
  --data .benchmarks/data/frames.jsonl \
  --experiment-dir .benchmarks/experiments/frames-fixed-dev10 \
  --experiment-id frames-dev10-fixed-corpus-v1 \
  --search-protocol fixed-corpus \
  --corpus .benchmarks/corpora/research-corpus.jsonl \
  --limit 10
```

Fixed-corpus mode fingerprints the complete corpus, uses deterministic lexical
search and reads, and does not grant the research or citation-verification stages Web
Search. This makes retrieval changes repeatable, but it is a local development
protocol—not automatically an official BrowseComp-Plus run. Official comparability
still requires the benchmark's released corpus, document mapping, retriever, and
grader protocol.

Optionally add a deterministic report-quality gate or a structured model rubric:

```bash
uv run --group benchmark deep-research-eval run frames \
  --data .benchmarks/data/frames.jsonl \
  --experiment-dir .benchmarks/experiments/frames-report-dev10 \
  --experiment-id frames-dev10-report-rubric-v1 \
  --report-judge openai \
  --report-judge-model "$JUDGE_MODEL_NAME" \
  --limit 10
```

The long-report rubric scores explicit criteria with strict structured output. Judge
failures are recorded as retryable evaluation errors, never silently converted into
a zero-quality report. Dataset files, corpora, checkpoints, and experiment records
are Git-ignored; they may still contain sensitive research data and should be handled
accordingly.

The default commands use the verified adaptive live-web development protocol. See
the [recorded pre-adaptive development baselines](docs/baselines.md) for historical
results and limitations.

## Quality Checks

Run the test suite:

```bash
uv run pytest
```

Run the code-quality checks:

```bash
uv run ruff check src tests
```

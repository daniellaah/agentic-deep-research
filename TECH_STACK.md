# Tech Stack

## Status

This document records current technical decisions. It is intentionally narrower than the
roadmap. A future capability does not belong here until the project has decided to adopt
it.

## Runtime

- **Language:** Python 3.12
- **Environment and dependency management:** uv
- **Workflow runtime:** `deep_research.py` at the repository root
- **Model instructions:** `agent_instructions.py` at the repository root
- **Custom tools:** `agent_tools.py` at the repository root
- **Execution model:** synchronous Python until concurrency is required
- **Default implementation preference:** Python standard library before new dependencies

The first releases use ordinary functions, dictionaries, lists, and local variables.
Classes, dataclasses, protocols, event hierarchies, and asynchronous execution are added
only when a concrete release becomes easier to understand with them.

## Model API

### Current release boundary

- **Hosted workflow:** OpenAI-hosted Scope, Supervisor, Write, Critic, and Revise
- **Research Worker:** selected once per run as OpenAI-hosted or local open-weight through vLLM
- **API:** Responses API on both Worker paths
- **SDK:** official OpenAI Python SDK
- **Clients:** one hosted `OpenAI` client and, only for a local Worker, one separate `OpenAI`
  client; both disable SDK retries through `max_retries=0`
- **Hosted configuration:** `OPENAI_API_KEY` and `MODEL_NAME`
- **Worker selection:** optional `WORKER_BACKEND`, defaulting to `openai`
- **Local Worker configuration:** `LOCAL_WORKER_BASE_URL`, `LOCAL_WORKER_API_KEY`, and
  `LOCAL_WORKER_MODEL_NAME`

Model names remain configuration rather than hard-coded project decisions. The local reference
model in the release specification is an acceptance baseline, not an application default.

### Local Worker compatibility decision

Release 0.9.0 lets only the Research Worker use either the current OpenAI endpoint or a local
model served through a vLLM Responses-compatible endpoint. Scope, Supervisor, and report stages
remain OpenAI-hosted. OpenAI-hosted Workers remain useful as baselines and acceptance references.

The local compatibility baseline is vLLM 0.20.2 with `Qwen/Qwen3-1.7B` and the reasoning,
structured-output, automatic-tool-choice, and Hermes parser flags documented by the official
[vLLM Responses tool example](https://docs.vllm.ai/en/v0.20.2/examples/online_serving/openai_responses_client_with_tools/).
The application neither depends on vLLM nor manages its server. It changes only the official
SDK client's base URL, API key, and request model for Worker calls.

The release specification also records one manually verified Apple Silicon operator profile:
vLLM-Metal with `mlx-community/Qwen3.5-9B-4bit`, the `qwen3` reasoning parser, and the
`qwen3_coder` tool parser. The official
[vLLM-Metal supported-model table](https://github.com/vllm-project/vllm-metal/blob/main/docs/supported_models.md)
lists the Qwen3.5 family as supported. This platform profile satisfies the same application
contract and does not replace the portable vLLM 0.20.2 reference or add a project dependency.

Both paths use the same strict tool definitions, `store=False`, complete ordered output-item
replay, linked function outputs, budgets, and result classification. vLLM 0.20.2 accepts local
reasoning-text items but rejects encrypted reasoning input; the selected local model does not
produce OpenAI-encrypted reasoning. Its Responses protocol represents output-token exhaustion as
`incomplete` but does not guarantee OpenAI refusal content items for the selected model. These
differences remain documented and visible as existing Worker outcomes instead of being hidden by
a provider gateway or fallback request. See the
[release specification](specs/v0.9.0-local-open-weight-worker.md) for the exact compatibility
contract and primary sources.

### Release 0.9.0 state

The runtime begins with a bounded Scope stage. Pydantic 2 defines
`ClarificationAssessment` and `ResearchBrief`, and the official SDK's
`client.responses.parse` validates both Structured Outputs boundaries. Scope always makes
one clarification-assessment request and one initial-brief request. It may ask at most
three questions in one round and may make one replacement-brief request after a user
revision. Standard-library terminal input collects answers and explicit approval; invalid
local input repeats only the current prompt without making another model request.

Research cannot begin until the user approves a validated brief. The application renders
that exact brief once and initializes a plain-dictionary `ResearchState`. A stateless
Research Supervisor repeatedly observes the brief, completed Worker results, and derived
remaining budget through a deterministic state rendering. Its Pydantic-validated
`SupervisorDecision` contains one `next_tasks` list constrained to zero or one
`ResearchTask`. An empty list finishes Research; one task starts one isolated Research
Worker. A task contains one focused question about one primary subject or direct comparison and
one through three `evidence_targets`; multiple examples are distributed across successive
Workers. The Supervisor instructions prohibit report-sized tasks and requirements for more than
three sources. The application merges only completed Worker results into state. The static
`ResearchPlan` and Plan stage are removed.

Each delegated Worker crosses one explicit application-owned run boundary. A frozen
`AgentRunRequest` contains its Worker number, rendered approved brief, validated task, bounded
model source, and model name. A
mutable `AgentRunState` privately retains the ordered Responses items, model-turn and custom-tool
usage, an active source registry, source-read usage, current status, termination reason, final
notes, and concise error while the Worker runs.
A frozen `AgentRunResult` copies task and model identity, terminal status, bounded termination
reason, notes or error, and used-versus-limit turn, tool, and read accounting. Standard-library
dataclasses and `StrEnum` represent this boundary because it is application state rather than
model-generated Structured Output.

Terminal statuses are `completed`, `failed`, and `cancelled`. Bounded termination reasons are
`completed`, `tool_limit`, `turn_limit`, `context_limit`, `refusal`, `model_error`, `tool_error`,
and `cancelled`. A successful forced final-notes turn after all five tool attempts returns
`completed`/`tool_limit`; using all six model attempts without valid notes returns
`failed`/`turn_limit`. `context_limit` is reserved for the later application-owned context budget;
the current hosted path classifies Responses request failures as `model_error` without parsing
exception messages. Every expected Worker outcome prints one result summary. Only completed
results enter `ResearchState`; failed or cancelled results remain fail-fast and stop before
another Supervisor or report request.

Every request remains stateless and uses `store=False`. The thin `llm_call` provider
primitive supports mutually exclusive custom tools or a Pydantic text format, accepts an
optional tool choice and output-token limit, and returns the official SDK `Response` without
wrapping it. Supervisor requests use a 4,000-token output limit. Worker requests omit an
application-set output limit because Responses API output limits include both reasoning tokens
and visible output, and real 4,000- and 8,000-token runs both ended before returning final notes.
Required non-Worker free-text extraction reports incomplete responses and refusals before
checking for empty text. The Worker harness classifies its own response and tool-loop outcomes
into the run result. Research uses
`tool_choice="auto"` while tool budget remains. Its
final request after budget exhaustion uses the same Research instructions and tools with
`tool_choice="none"`, plus one application-owned input directing the model to return notes
from existing evidence. `run_deep_research` keeps the top-level Scope, Research, and report
workflow visible. `run_scope_workflow` owns bounded intake and approval;
`run_research_supervisor_loop` owns Supervisor decisions, minimal state, dispatch, and result
merging; `run_research_worker` owns one visible Worker lifecycle;
`run_research_worker_loop` owns one mutable run state, its custom tool loop, and result
finalization; and `run_report_workflow` owns the explicit Write-Critic-Revise sequence. `main`
owns the CLI harness, backend selection, client construction, top-level error handling, and final
report output. The hosted client and optional local client use `max_retries=0` so transient API
failures remain visible at the first application-owned failure boundary instead of creating
hidden provider retries. The OpenAI Worker path reuses the hosted client; the local path constructs
one isolated client and never substitutes either path's API key.
`agent_instructions.py` owns the plain instruction and model-input constants for every
model-facing stage. `agent_tools.py` owns the tool schemas, functions, and dispatch.

Search results with eligible primary HTTP(S) URLs receive application-issued `S1`, `S2`, ...
identifiers scoped to one Worker. The application rejects credential-bearing, local-name,
non-global IP-literal, malformed, explicit PDF, and non-HTTP(S) destinations before registration.
`read_source_tool` accepts only one issued ID and a focused query; the application resolves the
URL and Tavily Extract returns bounded relevant Markdown content. Each read consumes one of two
read attempts and one of the existing five total tool attempts. The registry and extracted
content disappear with Worker history, while the immutable result retains only used-versus-limit
read accounting.

### Manual history retained from release 0.3.0

The explicit `run_research_worker_loop` invokes `llm_call` repeatedly and maintains one ordered
`input_items` list in application memory:

1. start with the user input;
2. append every item from each `openai_response.output` in its original order;
3. append application-owned `function_call_output` items; and
4. send the complete list as the next Responses API `input`.

The project does not use `previous_response_id` or the Conversations API.

Reasoning and assistant items needed for stateless continuation remain in the ordered
history. The application preserves provider-owned fields such as opaque encrypted content
or phase metadata when returned, replays those items without interpreting them, and does
not present them as private chain-of-thought.

The hosted and local Workers never share a history. The selected local Qwen profiles return
parser-produced reasoning text rather than encrypted reasoning. Any incompatible local
output or replay request fails at the existing `model_error` boundary without translation or
fallback.

The official OpenAI documentation states that manually managed history should preserve
prior user inputs and every response output item. Recheck the
[Responses API reference](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)
and [current model guidance](https://developers.openai.com/api/docs/guides/latest-model)
when specifying or implementing a multi-call release.

### Hosted capabilities

Custom function tools, application-owned history, and the sequential supervisor-worker
loop are implemented directly before built-in OpenAI tools, MCP, programmatic tool calling,
hosted deep-research models, or framework-owned multi-agent orchestration are adopted.
Hosted capabilities remain later comparison points.

## Research Tools

- **Web search:** synchronous Tavily Python SDK
- **Paper search:** arXiv query API with Atom XML parsing
- **Selected-source reading:** synchronous Tavily Extract through the same Python SDK
- **Tool execution:** synchronous custom function calls in response order

`TAVILY_API_KEY` authenticates general web search and selected-source extraction.
`tavily_search_tool` returns only title, content, and URL fields from Tavily results.
`arxiv_search_tool` uses `requests` to call the arXiv API and the standard library to parse title,
author, date, abstract URL, summary, and PDF URL fields from its Atom feed. `read_source_tool` uses
one application-resolved URL and a focused query to request at most five relevant chunks through
Tavily Extract. The application returns at most one selected-source result and truncates content
to 6,000 characters.

Each search returns at most three results. Tavily content and arXiv summaries are truncated
to 2,000 characters per result before they enter manually replayed Worker history. Each Worker
has at most six model turns, five total tool execution attempts, and two selected-source read
attempts. These deterministic boundaries keep the explicit history loop observable without
adding a compression stage.

All three JSON tool definitions and implementations live in `agent_tools.py` so
`deep_research.py` shows the Agent loop without network-client details.

Release 0.9.0 preserves discovery as separate from reading and does not expose arbitrary URL
fetching.
The model can read only application-issued IDs from its own searches; full papers, PDFs, direct
destination requests, browser behavior, citation allowlists, and output correction remain out of
scope. `tavily-python` and `requests` keep the three tool implementations short while the
application continues to own the Agent loop and destination allowlist.

## Command-Line Interface

The invocation shape is:

```text
uv run deep_research.py "Question"
```

The CLI has one initial question and emits concise live progress. The bounded Scope intake
retains optional clarification answers, explicit ResearchBrief approval or cancellation,
and at most one revision request followed by final approval. Release 0.6.0 adds visible
Supervisor decisions, isolated Worker boundaries, central result merging, and the final
Research stop reason. Release 0.7.0 adds one result summary for every started Worker; release
0.8.0 adds selected-source registration and read progress plus read usage to that summary.
Release 0.9.0 prints the selected Worker model source and name once before Scope and in every
Worker result. Local input validation may repeat the current prompt, but it cannot create another
model clarification round or brief revision.

The first release uses plain terminal text and the standard library. `python-dotenv` is
limited to loading local `.env` configuration. A terminal rendering dependency may be
added only when a release requires behavior that is difficult to express clearly without
it.

## Run Output

Release 0.1.0 writes one report to a unique directory under `runs/`:

```text
runs/<run-id>/
└── report.md
```

Release 0.2.0 adds `draft.md` and `critique.md` as visible intermediate workflow artifacts.
Release 0.3.0 removes file output to keep the first Agent release focused. Release 0.4.0
keeps the validated plan and report-stage values in memory. Release 0.5.0 also keeps the
clarification assessment, answers, pending and approved briefs, and approval state in
memory. Release 0.6.0 replaces the plan with in-memory `ResearchState`, Supervisor
decisions, and ordered Worker results. The CLI prints those boundaries, combined research
notes, draft, critique, and final Markdown report. Release 0.7.0 replaces each plain successful
Worker-result dictionary with an immutable `AgentRunResult` and also prints failed or cancelled
results before the existing fatal boundary. Release 0.8.0 keeps the source registry and selected
content only in the active Worker's memory. Release 0.9.0 adds model source and model name to the
request and result while keeping clients, endpoints, credentials, and output histories in memory.
It adds no output file or run directory.

Later releases may add artifacts only when their specifications require them. Historical
generated artifacts remain local and ignored by Git.

## Configuration

`.env.example` documents required variables. The runtime loads `.env` through
`python-dotenv`; values already present in the process environment take precedence. The
actual `.env` file remains local and must never be read into documentation, printed during
diagnostics, or committed.

Current variables:

- `OPENAI_API_KEY`
- `MODEL_NAME`
- `TAVILY_API_KEY`
- `WORKER_BACKEND` (optional, `openai` by default)
- `LOCAL_WORKER_BASE_URL` (required only for `local`)
- `LOCAL_WORKER_API_KEY` (required only for `local`)
- `LOCAL_WORKER_MODEL_NAME` (required only for `local`)

## Data Representation

Use plain dictionaries and lists for Responses API history and tool results while they remain
easy to understand. Release 0.7.0 introduces named standard-library dataclasses only for the
explicit Worker request, mutable run state, and terminal result that later local inference,
context sessions, and rollout consumers must share. Release 0.8.0 extends the mutable state with
one plain source dictionary and read counter and extends the result only with read usage and its
limit; it adds no source class or retrieval abstraction. Release 0.9.0 adds one bounded model
source enum and adds the selected source and model name to the existing request and result. It
does not store a client or endpoint in Agent state or introduce a model configuration class.

Pydantic 2 validates only model-generated application boundaries:
`ClarificationAssessment`, `ResearchBrief`, `SupervisorDecision`, and nested
`ResearchTask`. Bounded list sizes, required non-empty strings, and forbidden extra fields
are runtime invariants. Clarification answers, approval state, Responses API histories, and
`ResearchState` remain plain strings, lists, and dictionaries. The current `ResearchState` keys
are `approved_brief`, `worker_results`, and `stop_reason`; `worker_results` is an ordered list of
completed `AgentRunResult` values.

## Quality Checks

Release 0.9.0 uses Python compilation, CLI startup, and the manual acceptance scenarios in
its specification. Ruff, mypy, and an automated test suite remain intentionally omitted
while the learning runtime is kept minimal.

## Dependency Policy

A dependency is acceptable only when:

1. the current specification requires the capability;
2. the standard library would distract from the Agent mechanism being studied;
3. the dependency does not introduce a framework that owns the agent loop; and
4. its purpose is recorded in this document.

The project currently depends on the OpenAI SDK, Pydantic 2, `python-dotenv`,
`tavily-python`, and `requests` at runtime.

vLLM, model weights, parsers, accelerator runtimes, and their dependencies belong to the
operator-managed local server and are not project dependencies.

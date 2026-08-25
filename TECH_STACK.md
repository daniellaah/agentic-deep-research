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

- **LLM provider:** required once per run as `openai` or `deepseek`
- **Workflow routing:** the selected provider and model serve Scope, Supervisor, every Research
  Worker, context summary, Write, Critic, and Revise
- **API:** Responses API on both provider paths
- **SDK:** official OpenAI Python SDK
- **Clients:** exactly one `OpenAI` SDK client per run; it uses the SDK default OpenAI endpoint or
  the official `https://api.deepseek.com` base URL and disables SDK retries with `max_retries=0`
- **Provider selection:** required `LLM_PROVIDER`
- **OpenAI configuration:** `OPENAI_API_KEY` and `OPENAI_MODEL_NAME`
- **DeepSeek configuration:** `DEEPSEEK_API_KEY` and `DEEPSEEK_MODEL_NAME`
- **Shared tool configuration:** `TAVILY_API_KEY`

Both model names are operator-configured. DeepSeek V4 Flash is the release's real-endpoint
acceptance baseline, not a hard-coded runtime default.

### Global provider compatibility decision

Release 0.10.0 removes the release 0.9.0 local vLLM runtime path. `LLM_PROVIDER` now selects one
hosted provider before Scope, and every model call in that run uses the selected provider's single
client and configured model. The application does not route stages independently, substitute
credentials, or fall back to the other provider.

DeepSeek's official [Responses API guide](https://api-docs.deepseek.com/guides/responses_api/)
documents the current compatibility boundary: `deepseek-v4-flash` supports function tools,
JSON-schema `text.format`, `max_output_tokens`, manual input-item replay, and Responses-style usage.
The endpoint is stateless and does not support `previous_response_id`, Conversations, or
`context_management`, which matches the application's explicit application-owned context-session
mechanism.

DeepSeek documents that `parallel_tool_calls` is ignored and parallel tool calling is always
enabled. Release 0.10.0 therefore accepts multiple function calls in one response, executes every
budget-permitted call synchronously in response order, and appends one linked output for every
returned call. The application does not depend on the provider honoring the request hint.

Both provider paths use the same strict tool definitions, `store=False`, complete ordered output-item
replay, linked function outputs, budgets, and result classification. DeepSeek does not support
server-side storage on this surface and returns `store: false`. Compatibility differences remain
documented and visible as existing Worker outcomes instead of being hidden by a provider gateway
or fallback request. See the
[release specification](specs/v0.10.0-long-horizon-context-sessions.md#llm-provider-and-deepseek-compatibility-contract)
for the exact compatibility contract and primary sources. The released
[v0.9.0 specification](specs/v0.9.0-local-open-weight-worker.md) remains the historical record of
the removed local path.

### Current runnable state

The runtime begins with a bounded Scope stage. Pydantic 2 defines
`ClarificationAssessment` and `ResearchBrief`, and the official SDK's
`client.responses.parse` validates both Structured Outputs boundaries. Scope always makes
one clarification-assessment request and one initial-brief request. It may ask at most
three questions in one round and may make one replacement-brief request after a user
revision. Standard-library terminal input collects answers and explicit approval; invalid
terminal input repeats only the current prompt without making another model request.

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
LLM provider, and model name. A
mutable `AgentRunState` privately retains the ordered Responses items, model-turn and custom-tool
usage, an active source registry, successfully read source IDs, source-read usage, current context
session, latest validated summary, peak token accounting, current status, termination reason,
final notes, and concise error while the Worker runs.
A frozen `AgentRunResult` copies task and model identity, terminal status, bounded termination
reason, notes or error, and used-versus-limit turn, tool, read, session, summary, and context-token
accounting. Standard-library dataclasses and `StrEnum` represent this boundary because it is
application state rather than model-generated Structured Output.

Terminal statuses are `completed`, `failed`, and `cancelled`. Bounded termination reasons are
`completed`, `tool_limit`, `turn_limit`, `context_limit`, `refusal`, `model_error`, `tool_error`,
and `cancelled`. A successful forced final-notes turn after all ten tool attempts returns
`completed`/`tool_limit`; using all 15 model attempts without valid notes returns
`failed`/`turn_limit`. `context_limit` now represents the application-owned 20,000-token
pre-summary projection limit or five-session limit. Provider context-window errors remain
`model_error` because the application does not parse exception text. Every expected Worker outcome
prints one result summary. Only completed results enter `ResearchState`; failed or cancelled
results remain fail-fast and stop before another Supervisor or report request.

Every request remains stateless and uses `store=False`. The thin `llm_call` provider
primitive supports mutually exclusive custom tools or a Pydantic text format, accepts an
optional tool choice, parallel-tool setting, and output-token limit, and returns the official SDK
`Response` without wrapping it. Every request retains provider-default reasoning because real
DeepSeek probes with low or disabled reasoning returned fenced JSON that the official SDK correctly
rejected. Supervisor requests use an 8,000-token output limit because real
DeepSeek acceptance showed that its default reasoning could consume a 4,000-token limit before
returning one small validated decision. Ordinary
Worker requests omit an
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
owns the CLI harness, provider selection, single-client construction, top-level error handling,
and final report output. The selected client uses `max_retries=0` so transient API failures remain
visible at the first application-owned failure boundary instead of creating hidden provider
retries. Only the selected provider's API key and model name are required for routing.
`agent_instructions.py` owns the plain instruction and model-input constants for every
model-facing stage. `agent_tools.py` owns the tool schemas, functions, and dispatch.

Every ordinary Worker response must provide consistent integer token usage. For zero, one, or
multiple returned function calls, the application executes budget-permitted calls synchronously
in response order and links every result. It projects the next input from exact input and output
usage plus the UTF-8 byte length and a fixed 256-token allowance for each linked function output.
A projection at or above 12,000 tokens enters the explicit context boundary; a projection above
20,000 stops before summary. Otherwise the complete active-session history is replayed unchanged.

At a permitted boundary, the same selected model spends one of the 15 run-level turns on a
16,000-token-bounded `ResearchStateSummary` Structured Output. The application validates its source
IDs against the run registry and prevents `selected_source` claims for IDs without a successful
read. It renders URLs from the registry, discards the old history, and starts the next of at most
five sessions from only the task, approved brief, remaining budgets, and validated summary. Tool,
read, source, turn, and context counters remain run-level.

Search results with eligible primary HTTP(S) URLs receive application-issued `S1`, `S2`, ...
identifiers scoped to one Worker. The application rejects credential-bearing, local-name,
non-global IP-literal, malformed, explicit PDF, and non-HTTP(S) destinations before registration.
`read_source_tool` accepts only one issued ID and a focused query; the application resolves the
URL and Tavily Extract returns bounded relevant Markdown content. Each read consumes one of four
read attempts and one of the ten total tool attempts. The source registry and successfully read
source-ID set survive context replacement, while raw extracted content survives only when selected
into the bounded summary. The immutable result retains accounting rather than source content.

### Manual history retained from release 0.3.0

The explicit `run_research_worker_loop` invokes `llm_call` repeatedly and maintains one ordered
active-session `input_items` list in application memory:

1. start with the user input;
2. append every item from each `openai_response.output` in its original order;
3. append application-owned `function_call_output` items; and
4. send the complete list as the next Responses API `input`; and
5. after a validated context summary, replace the complete list with one resumed-state user item.

The project does not use `previous_response_id` or the Conversations API.

Reasoning and assistant items needed for stateless continuation remain in the ordered
history. The application preserves provider-owned fields such as opaque encrypted content
or phase metadata when returned, replays those items without interpreting them, and does
not present them as private chain-of-thought.

OpenAI and DeepSeek execute in separate runs and never share a client or history. Any incompatible
DeepSeek output or replay request fails at the existing `model_error` boundary without translation
or fallback.

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
has at most 15 model turns, ten total tool execution attempts, four selected-source read attempts,
five context sessions, and four context summaries. These deterministic boundaries keep explicit
history replay and replacement observable.

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
Worker result. Release 0.10.0 renames that identity to LLM provider and applies it to the complete
workflow. It also prints session starts, required projection boundaries, validated summary blocks,
and context accounting in each result. Terminal input validation may repeat the current prompt,
but it cannot create another model clarification round or brief revision.

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
Release 0.10.0 renames model source to LLM provider and prints the global provider and model before
Scope. It keeps only the latest validated summary in mutable state, does not retain replaced
histories, and adds no output file or run directory.

Later releases may add artifacts only when their specifications require them. Historical
generated artifacts remain local and ignored by Git.

## Configuration

`.env.example` documents required variables. The runtime loads `.env` through
`python-dotenv`; values already present in the process environment take precedence. The
actual `.env` file remains local and must never be read into documentation, printed during
diagnostics, or committed.

Current variables:

- `LLM_PROVIDER` (required; accepts exactly `openai` or `deepseek`)
- `OPENAI_API_KEY` (required only for `openai`)
- `OPENAI_MODEL_NAME` (required only for `openai`)
- `DEEPSEEK_API_KEY` (required only for `deepseek`)
- `DEEPSEEK_MODEL_NAME` (required only for `deepseek`)
- `TAVILY_API_KEY`

## Data Representation

Use plain dictionaries and lists for Responses API history and tool results while they remain
easy to understand. Release 0.7.0 introduces named standard-library dataclasses only for the
explicit Worker request, mutable run state, and terminal result that later hosted-model comparisons,
context sessions, and rollout consumers must share. Release 0.8.0 extends the mutable state with
one plain source dictionary and read counter and extends the result only with read usage and its
limit; it adds no source class or retrieval abstraction. Release 0.9.0 adds one bounded model
source enum and adds the selected source and model name to the existing request and result. It
does not store a client or endpoint in Agent state or introduce a model configuration class.
Release 0.10.0 renames that enum to `LLMProvider` and its request/result field to `llm_provider`,
matching the now-global routing decision. It extends the existing state and result dataclasses
with context counters and peak token values rather than introducing a context-manager class.

Pydantic 2 validates only model-generated application boundaries:
`ClarificationAssessment`, `ResearchBrief`, `SupervisorDecision`, and nested
`ResearchTask`, plus the v0.10.0 `ResearchStateSummary` and nested `SummarySource`. Application
checks add source-ID uniqueness, registry membership, and non-inflated evidence levels after
Pydantic validation. Scope, Supervisor, and summary instructions repeat their Pydantic list bounds
and use conservative text-generation targets below the hard character limits because real DeepSeek
acceptance showed that schema acceptance and exact hard-limit prompting do not guarantee every
generated value remains within those bounds. They also request raw JSON explicitly and forbid
Markdown code fences, matching DeepSeek's official JSON-output prompting guidance.
Clarification answers, approval state,
Responses API histories, and
Supervisor `ResearchState` remain plain strings, lists, and dictionaries. The current Supervisor
state keys are `approved_brief`, `worker_results`, and `stop_reason`; `worker_results` is an ordered
list of completed `AgentRunResult` values.

## Quality Checks

Release 0.10.0 requires Python compilation, CLI startup, configuration-matrix checks, controlled
client state-machine checks, and real OpenAI and DeepSeek protocol scenarios defined in its
specification. Ruff, mypy, and a general automated test suite remain intentionally omitted while
the learning runtime is kept minimal.

## Dependency Policy

A dependency is acceptable only when:

1. the current specification requires the capability;
2. the standard library would distract from the Agent mechanism being studied;
3. the dependency does not introduce a framework that owns the agent loop; and
4. its purpose is recorded in this document.

The project currently depends on the OpenAI SDK, Pydantic 2, `python-dotenv`,
`tavily-python`, and `requests` at runtime. DeepSeek uses the existing SDK and adds no dependency.

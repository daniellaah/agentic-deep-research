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

- **Provider:** OpenAI only
- **API:** Responses API only
- **SDK:** official OpenAI Python SDK
- **Client:** `OpenAI`
- **Model configuration:** required through `MODEL_NAME`
- **Credentials:** required through `OPENAI_API_KEY`

The model name is configuration rather than a hard-coded project decision. This keeps the
learning code stable while model availability changes.

### Release 0.5.0 state

The runtime begins with a bounded Scope stage. Pydantic 2 defines
`ClarificationAssessment` and `ResearchBrief`, and the official SDK's
`client.responses.parse` validates both Structured Outputs boundaries. Scope always makes
one clarification-assessment request and one initial-brief request. It may ask at most
three questions in one round and may make one replacement-brief request after a user
revision. Standard-library terminal input collects answers and explicit approval; invalid
local input repeats only the current prompt without making another model request.

Planning cannot begin until the user approves a validated brief. The application renders
that exact brief once and passes only the rendered approved brief into the existing static
planner, task research, Write, Critic, and Revise. The original question, clarification
artifacts, superseded brief, revision request, and approval commands remain isolated inside
Scope. The `ResearchPlan` and `ResearchTask` schemas, immutable plan behavior, sequential
task execution, fresh task histories, and independent model-turn and tool-call limits
remain unchanged from release 0.4.0.

Every request remains stateless and uses `store=False`. The thin `llm_call` provider
primitive supports mutually exclusive custom tools or a Pydantic text format, accepts an
optional tool choice for requests with tools, and returns the official SDK `Response`
without wrapping it. Research uses `tool_choice="auto"` while tool budget remains. Its
final request after budget exhaustion uses the same Research instructions and tools with
`tool_choice="none"`, plus one application-owned input directing the model to return notes
from existing evidence. `research_workflow` owns scoping, planning, sequential research,
and the three report stages. `main` owns the CLI harness, client construction, top-level
error handling, and final report output. `agent_instructions.py` owns the plain instruction
and model-input constants for every model-facing stage. `agent_tools.py` owns the tool
schemas, functions, and dispatch.

### Manual history retained from release 0.3.0

The explicit `agent_loop` invokes `llm_call` repeatedly and maintains one ordered
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

The official OpenAI documentation states that manually managed history should preserve
prior user inputs and every response output item. Recheck the
[Responses API reference](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)
and [current model guidance](https://developers.openai.com/api/docs/guides/latest-model)
when specifying or implementing a multi-call release.

### Hosted capabilities

Custom function tools and application-owned history are implemented before built-in
OpenAI tools, MCP, programmatic tool calling, hosted deep-research models, or multi-agent
features are adopted. Hosted capabilities remain later comparison points.

## Research Tools

- **Web search:** synchronous Tavily Python SDK
- **Paper search:** arXiv query API with Atom XML parsing
- **Tool execution:** synchronous custom function calls in response order

`TAVILY_API_KEY` authenticates general web search. `tavily_search_tool` returns only title,
content, and URL fields from Tavily results. `arxiv_search_tool` uses `requests` to call the
arXiv API and the standard library to parse title, author, date, abstract URL, summary, and
PDF URL fields from its Atom feed.

Both JSON tool definitions and both function implementations live in `agent_tools.py` so
`deep_research.py` shows the Agent loop without network-client details.

Release 0.3.0 deliberately omits selected-page reading, arbitrary URL fetching, network
destination controls, citation allowlists, and output correction. `tavily-python` and
`requests` keep the two tool implementations short while the application continues to own
the Agent loop.

## Command-Line Interface

The invocation shape is:

```text
uv run deep_research.py "Question"
```

The CLI has one initial question and emits concise live progress. Release 0.5.0 adds one
bounded mid-run intake flow: optional clarification answers, explicit ResearchBrief
approval or cancellation, and at most one revision request followed by final approval.
Local input validation may repeat the current prompt, but it cannot create another model
clarification round or brief revision.

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
memory. The CLI uses consistent plain-text sections to print progress, clarification
questions, pending briefs, the plan, every task's notes, combined research notes, draft,
critique, and final Markdown report. It does not create a run directory.

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

## Data Representation

Use plain dictionaries and lists for Responses API history, tool results, and task-result
collections while they remain easy to understand.

Pydantic 2 validates only model-generated application boundaries:
`ClarificationAssessment`, `ResearchBrief`, `ResearchPlan`, and `ResearchTask`. Bounded list
sizes, required non-empty strings, and forbidden extra fields are runtime invariants.
Clarification answers, approval state, Responses API histories, and task-result collections
remain plain strings, lists, and dictionaries.

## Quality Checks

Release 0.5.0 uses Python compilation, CLI startup, and the manual acceptance scenarios in
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

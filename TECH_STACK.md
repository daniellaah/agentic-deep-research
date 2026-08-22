# Tech Stack

## Status

This document records current technical decisions. It is intentionally narrower than the
roadmap. A future capability does not belong here until the project has decided to adopt
it.

## Runtime

- **Language:** Python 3.12
- **Environment and dependency management:** uv
- **Workflow runtime:** `deep_research.py` at the repository root
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

### Release 0.3.0 state

The runtime begins with an explicit synchronous `agent_loop` in which the model can select
simple Tavily web search and arXiv paper search functions. The loop owns an ordered
in-memory history and enforces model-turn and tool-call limits. Its final research notes
become the evidence input to the existing explicit Write, Critic, and Revise calls.

Every request remains stateless and uses `store=False`. The provider primitive is the thin
`llm_call(client, model_name, instructions, model_input, tools=None) -> Response` function,
which returns the official SDK `Response` without wrapping it. `research_workflow` owns the
Agent and three report stages, while `main` owns the CLI harness, client construction, and
top-level error handling and prints the final report. `agent_tools.py` owns the tool
schemas, functions, and dispatch.

### Manual history from release 0.3.0

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

The CLI has one initial question and emits concise live progress. It does not accept
mid-run input until the scoping release introduces bounded clarification and an explicit
ResearchBrief approval point.

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
Release 0.3.0 removes file output to keep the first Agent release focused. Research notes,
draft, critique, and final report remain in memory. The CLI prints progress and the final
Markdown report. It does not create a run directory.

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

Starting in release 0.3.0, use plain dictionaries and lists for Responses API history and
tool results while they remain easy to understand.

Introduce a named data structure only when repeated validation or invariants make the
plain representation harder to follow. Pydantic may validate model-generated planning,
brief, decision, or result structures at explicit boundaries, but should not replace plain
internal history data merely for consistency.

## Quality Checks

Release 0.3.0 uses Python compilation, CLI startup, and the manual acceptance scenarios in
its specification. Ruff, mypy, and an automated test suite are intentionally omitted while
the learning runtime is kept minimal.

## Dependency Policy

A dependency is acceptable only when:

1. the current specification requires the capability;
2. the standard library would distract from the Agent mechanism being studied;
3. the dependency does not introduce a framework that owns the agent loop; and
4. its purpose is recorded in this document.

The project currently depends on the OpenAI SDK, `python-dotenv`, `tavily-python`, and
`requests` at runtime.

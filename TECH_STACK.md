# Tech Stack

## Status

This document records current technical decisions. It is intentionally narrower than the
roadmap. A future capability does not belong here until the project has decided to adopt
it.

## Runtime

- **Language:** Python 3.12
- **Environment and dependency management:** uv
- **Runtime file:** `deep_research.py` at the repository root
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

### Release 0.1.0 state

The first release makes exactly one Responses API call. It has no tool, loop, conversation
history, or structured trace.

The provider primitive is the thin
`llm_call(client, model_name, model_input) -> Response` function. It returns the official
SDK `Response` without wrapping it, while `main` owns the CLI harness and invokes
`llm_call` once. Release 0.2.0 will reuse this primitive in three explicit fixed workflow
stages. Release 0.3.0 will place the first `agent_loop` around it.

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

## Run Artifacts

Release 0.1.0 writes one report to a unique directory under `runs/`:

```text
runs/<run-id>/
└── report.md
```

Release 0.2.0 adds `draft.md` and `critique.md` as visible intermediate workflow artifacts.
Structured `trace.jsonl` output begins in release 0.3.0, when the first Agent loop
introduces model, tool, history, and stopping transitions worth inspecting.

`report.md` contains the final user-facing result. Later releases may add evidence or
checkpoint artifacts only when their specifications require them.

Generated run artifacts are local and ignored by Git.

## Configuration

`.env.example` documents required variables. The runtime loads `.env` through
`python-dotenv`; values already present in the process environment take precedence. The
actual `.env` file remains local and must never be read into documentation, printed during
diagnostics, or committed.

Initial variables:

- `OPENAI_API_KEY`
- `MODEL_NAME`

New search-provider credentials may be added only when the relevant tool release selects a
provider.

## Data Representation

Starting in release 0.3.0, use plain dictionaries and lists for trace events and Responses
API history while they remain easy to understand. Serialize trace events with the standard
`json` module.

Introduce a named data structure only when repeated validation or invariants make the
plain representation harder to follow. Pydantic may validate model-generated planning,
brief, decision, or result structures at explicit boundaries, but should not replace plain
internal history and trace data merely for consistency.

## Quality Tooling

- Ruff for static checks and linting
- mypy in strict mode once runtime code exists
- manual acceptance runs defined by each release specification
- no automated test suite during the initial learning releases

The baseline repository check is:

```text
uv run ruff check .
```

Once `deep_research.py` exists, the expected checks become:

```text
uv run ruff check deep_research.py
uv run mypy deep_research.py
```

## Dependency Policy

A dependency is acceptable only when:

1. the current specification requires the capability;
2. the standard library would distract from the Agent mechanism being studied;
3. the dependency does not introduce a framework that owns the agent loop; and
4. its purpose is recorded in this document.

The project currently depends on the OpenAI SDK and `python-dotenv` at runtime.

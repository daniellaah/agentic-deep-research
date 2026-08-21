# Tech Stack

## Status

This document records current technical decisions. It is intentionally narrower than the
roadmap. A future capability does not belong here until the project has decided to adopt
it.

## Runtime

- **Language:** Python 3.12
- **Environment and dependency management:** uv
- **Runtime file:** `deep_research.py` at the repository root
- **Execution model:** asynchronous Python with `asyncio`
- **Default implementation preference:** Python standard library before new dependencies

Asynchronous execution is selected early because later releases will need streaming,
concurrent tool calls, and parallel research branches. It does not imply an internal
framework or multiple runtime modules.

## Model API

- **Provider:** OpenAI only
- **API:** Responses API only
- **SDK:** official OpenAI Python SDK
- **Client:** `AsyncOpenAI`
- **Model configuration:** required through `MODEL_NAME`
- **Credentials:** required through `OPENAI_API_KEY`

The model name is configuration rather than a hard-coded project decision. This keeps the
learning code stable while model availability changes.

The first releases use custom function tools and an application-managed loop. Built-in
OpenAI tools, MCP, programmatic tool calling, hosted deep-research models, and multi-agent
features are later comparison points. They must not hide a mechanism before the project
has implemented and observed that mechanism directly.

The first release continues iterations with `previous_response_id`. Manual provider
history ownership and response compaction are deferred until a long-horizon release so
they can be learned as a separate context-engineering concern.

Current Responses API capabilities and schemas must be checked against the
[official OpenAI documentation](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)
when a release is specified or implemented.

## Command-Line Interface

The planned invocation shape is:

```text
uv run --env-file .env deep_research.py "Research question"
```

The CLI has one initial question and emits live progress. It does not accept mid-run input
until a planning release introduces an explicit approval point.

The first release uses plain terminal text. A terminal rendering dependency may be added
only when a release requires behavior that is difficult to express clearly with the
standard library.

## Run Artifacts

Every run writes to a unique directory under `runs/`:

```text
runs/<run-id>/
├── trace.jsonl
└── report.md
```

`trace.jsonl` is the canonical record of observable run events. Terminal output is a
human-readable projection of those events rather than a separate logging model.

`report.md` contains the final user-facing result. Later releases may add evidence or
checkpoint artifacts only when their specifications require them.

Generated run artifacts are local and ignored by Git.

## Configuration

`.env.example` documents required variables. The actual `.env` file remains local and
must never be read into documentation, printed during diagnostics, or committed.

Initial variables:

- `OPENAI_API_KEY`
- `MODEL_NAME`

New search-provider credentials may be added only when the relevant tool release selects a
provider.

## Data Representation

Use built-in values and frozen, slotted dataclasses when named trusted runtime values make
the single file easier to understand. Serialize trace events with the standard `json`
module.

Do not introduce Pydantic merely for internal values. It may be adopted later for a
genuinely untrusted structured boundary or provider-generated structured output.

## Quality Tooling

- Ruff for formatting-independent static checks and linting
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

The project currently depends only on the OpenAI SDK at runtime.

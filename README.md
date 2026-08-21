# Agentic Deep Research

A single-file deep-research command-line agent built as a sequence of small, complete,
observable learning releases.

## Status

The repository currently contains the SDD foundation and the draft specification for
`v0.1.0`. Runtime implementation has not started, and no release tag exists.

The previous Web application is preserved in the
`archive/observable-web-application` branch.

## Project Shape

All runtime behavior will live in:

```text
deep_research.py
```

Every completed run will produce:

```text
runs/<run-id>/
├── trace.jsonl
└── report.md
```

Terminal progress and the saved trace describe the same small set of run milestones.

## Documentation

- [Mission](MISSION.md)
- [Tech stack](TECH_STACK.md)
- [Roadmap](ROADMAP.md)
- [Research notes](RESEARCH.md)
- [Repository instructions](AGENTS.md)
- [Release specifications](specs/README.md)
- [v0.1.0 specification](specs/v0.1.0-single-response.md)

## Development Sequence

For each release:

1. identify the learning question and current limitation;
2. write or revise the release specification;
3. implement the cumulative change in `deep_research.py`;
4. run the specification's manual acceptance scenario;
5. inspect the trace and report;
6. update documentation when a decision changed; and
7. create the release tag only after verification.

## Environment

The project uses Python 3.12 and uv.

Create a local environment file from `.env.example` and provide:

- `OPENAI_API_KEY`
- `MODEL_NAME`

Never commit the resulting `.env` file.

The planned command shape is:

```text
uv run --env-file .env deep_research.py "Research question"
```

This command will become available when `v0.1.0` is implemented.

## Current Check

```text
uv run ruff check .
```

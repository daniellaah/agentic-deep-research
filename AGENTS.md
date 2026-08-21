# Repository Instructions

## Project Purpose

Build a deep-research command-line agent as a sequence of small, complete, observable
learning releases. Each release should make one or more agent mechanisms understandable
through a runnable implementation, a structured trace, and a final Markdown report.

## Development Model

- Use spec-driven development. Define each release in `specs/` before implementing it.
- Keep all runtime behavior in the repository-root `deep_research.py` file.
- Evolve that file cumulatively; every release must remain complete and runnable.
- Prefer direct, explicit code over frameworks, layers, adapters, and premature abstractions.
- Add a dependency only when the current release specification requires it.
- Do not add a web application, service API, provider-neutral gateway, or evaluation
  framework unless a later specification explicitly changes the project scope.
- Automated tests are not required unless the current specification requests them.

## Product Boundaries

- Provide a command-line interface with live progress output.
- Do not support mid-run user interaction until a planning release explicitly introduces
  a human approval point.
- Use only the OpenAI Responses API for model interaction.
- Begin with custom function-tool loops so that the agent harness remains visible.
- Introduce paper search and web search incrementally through release specifications.
- Produce a structured, serializable trace and a final Markdown report for completed runs.
- Trace model and tool activity without claiming to expose private chain-of-thought.

## Language

All repository-authored content must be in English, including documentation, source code,
comments, prompts, CLI text, configuration, commit messages, and release notes.

## Branches and Releases

- Branch names must not contain `codex`.
- Do not use release versions in branch names.
- Use `archive/<name>` for preserved historical implementations.
- Use `chore/<name>` for tooling and project structure.
- Use `feat/<name>` for product capabilities.
- Use `fix/<name>` for corrections.
- Use `docs/<name>` for documentation-only work.
- Follow Semantic Versioning for project versions.
- Release tags use the form `vMAJOR.MINOR.PATCH`.
- Create a release tag only after its specification has been implemented and manually
  verified.

## Generated Artifacts

- Store generated run artifacts under `runs/`.
- Do not commit generated traces, reports, caches, virtual environments, or secrets.
- Keep `.env` local and document required variables in `.env.example`.

## Required Checks

Run the checks configured for the current release. Until runtime code is introduced, run:

```text
uv run ruff check .
```

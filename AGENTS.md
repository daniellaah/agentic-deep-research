# Repository Instructions

## Project Purpose

Build a deep-research command-line agent as a sequence of small, complete, observable
learning releases. Each release should make one or more model or agent mechanisms
understandable through a runnable implementation and a final Markdown report. Structured
tracing is deferred until a release specification explicitly introduces it.

## Development Model

- Use spec-driven development. Define each release in `specs/` before implementing it.
- Keep the workflow and Agent loop in the repository-root `deep_research.py` file.
- Keep custom tool definitions and implementations in the repository-root
  `agent_tools.py` file.
- Evolve those files cumulatively; every release must remain complete and runnable.
- Prefer direct, explicit code over frameworks, layers, adapters, and premature abstractions.
- Prefer plain functions, dictionaries, and lists until observed complexity justifies a
  named structure.
- Keep execution synchronous until a release requires concurrency or streaming.
- Add a dependency only when the current release specification requires it.
- Do not add a web application, service API, provider-neutral gateway, or evaluation
  framework unless a later specification explicitly changes the project scope.
- Automated tests are not required unless the current specification requests them.

## Specification Workflow

- Before implementing or reviewing a release, read `MISSION.md`, `TECH_STACK.md`,
  `ROADMAP.md`, `specs/README.md`, and the relevant release specification.
- Treat the active release specification as the source of truth for that release's
  observable behavior, runtime flow, output, stopping behavior, implementation constraints,
  and completion criteria.
- Treat `ROADMAP.md` as sequencing direction, not as an implementation specification.
- Do not implement a planned capability unless it is included in the active release
  specification.
- If the required release specification is missing, contradictory, or materially
  incomplete, resolve the specification before changing runtime code.
- When a roadmap or specification decision depends on time-sensitive API behavior or
  current Agent research, verify it against current primary sources and record only the
  decision-relevant rationale in the applicable release specification.

## Product Boundaries

- Provide a command-line interface with live progress output.
- Do not support mid-run user interaction until a scoping release explicitly introduces
  bounded clarification and a ResearchBrief approval point.
- Use only the OpenAI Responses API for model interaction.
- Implement a custom function-tool loop before adopting hosted tool orchestration so that
  the agent harness remains visible.
- Introduce paper search and web search incrementally through release specifications.
- Produce a final Markdown report for completed runs.
- Add structured tracing only when an active release specification requires it, without
  claiming to expose private chain-of-thought.

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

Release 0.3.0 requires only Python compilation, CLI startup, and its manual acceptance
scenario:

```text
uv run python -m py_compile deep_research.py agent_tools.py
uv run deep_research.py --help
```

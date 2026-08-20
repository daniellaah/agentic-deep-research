# Repository Instructions

## Project Purpose

Build a production-style deep-research agent as a sequence of small, complete,
observable vertical slices. The codebase should remain understandable at every step.

## Initial Release Scope

The first release is `0.1.0`. Do not create the `v0.1.0` Git tag until all of the
following are complete and verified:

- a minimal agent loop;
- a web application that can start a run;
- structured trace collection;
- a web view of the run trajectory.

Evaluation is intentionally outside the initial release.

## Collaboration Workflow

The repository owner implements feature code in order to understand the complete
development path. Unless the owner explicitly asks for implementation, an agent must:

1. explain the feature goal and design first;
2. identify the files and responsibilities involved;
3. guide the owner through small implementation steps;
4. review and explain the owner's changes;
5. help verify the result before moving to the next feature.

Agents may create and maintain directory structure, dependency configuration,
development tooling, and other project scaffolding. They must not directly implement
agent behavior, trace behavior, web business logic, or evaluation unless explicitly asked.

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

## Architecture Boundaries

Backend code lives under `src/agentic_deep_research`:

- `domain`: trusted internal concepts and rules;
- `application`: use cases and ports;
- `infrastructure`: implementations of application ports;
- `api`: HTTP boundary and transport schemas.

Dependency direction is `api -> application -> domain`. Infrastructure may depend on
application ports and domain types, but domain and application code must not depend on
FastAPI, databases, or model-provider SDKs.

The frontend lives under `web/src`. Organize product capabilities under `features`,
shared visual elements under `components`, and framework-independent helpers under `lib`.

Do not add an abstraction or dependency until the current feature requires it.

## Data Models

- Use frozen, slotted standard-library dataclasses for trusted internal domain values.
- Use Pydantic models at untrusted boundaries such as HTTP requests, configuration,
  serialized data, and provider structured output.
- Do not use Pydantic as the default model for every internal object.

## Trace Requirements

Every completed agent run must eventually produce structured, serializable trace events.
Trace data may include execution state, tool activity, timing, inputs, and outputs. It must
not claim to expose a model's private chain-of-thought.

## Required Checks

Run these checks after changing the corresponding area:

```text
uv run ruff check src tests
uv run mypy src
uv run pytest
npm --prefix web run check
npm --prefix web run test
npm --prefix web run build
```

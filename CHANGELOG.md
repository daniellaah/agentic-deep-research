# Changelog

All notable changes to this project will be documented in this file.

The project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] - 2026-08-22

### Added

- Added an explicit synchronous Agent loop with simple Tavily web search and direct arXiv
  paper search tools.
- Added application-owned Responses API history with linked function outputs, hard model
  and tool budgets, and visible Agent progress.
- Added `agent_tools.py` for custom tool definitions, implementations, and dispatch.
- Added source-grounded research notes before the write-critic-revise report stages.
- Added the v0.3.0 release specification and current primary-source API rationale.

### Changed

- Grounded report drafting, critique, and revision in the Research Agent's source-linked
  notes.
- Kept the first tool release focused by deferring selected-page reading, network URL
  controls, deterministic citation validation, and output correction.
- Added `tavily-python` and `requests` so both research tools remain short and direct.
- Deferred structured tracing so v0.3.0 stays focused on the custom Agent loop and explicit
  report-refinement stages.
- Simplified v0.3.0 to keep all stage output in memory and print only progress and the final
  report, without creating run artifacts.
- Removed Ruff and mypy from the current release workflow so verification focuses on a
  complete runnable CLI and its manual acceptance scenarios.
- Allowed multiple function calls in one Research response while keeping tool execution
  synchronous and ordered.
- Required `TAVILY_API_KEY` for web research and updated the project version to
  0.3.0.

## [0.2.0] - 2026-08-21

### Added

- Added a fixed three-stage write-critic-revise workflow with explicit progress output.
- Added inspectable `draft.md` and `critique.md` artifacts alongside the final report.
- Added the v0.2.0 release specification and its partial-artifact failure contract.

### Changed

- Updated report generation from one model call to three stateless Responses API calls
  with explicit stage inputs.
- Reworked the near-term roadmap around fixed report refinement, observable research
  tools, static planning, scoping, adaptive supervision, and bounded parallel workers.
- Defined the required release-specification reading workflow and replaced persistent
  research notes with on-demand research captured in the relevant release specification.

## [0.1.0] - 2026-08-21

### Added

- A single-command research CLI.
- One synchronous OpenAI Responses API request for report generation.
- Terminal progress and generated report output.
- Markdown reports stored in unique run directories.
- Environment-based API credential and model configuration.
- Ruff and mypy checks for the single-file runtime.

# Changelog

All notable changes to this project will be documented in this file.

The project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Added run-level `LLM_PROVIDER` selection between OpenAI and the official hosted DeepSeek
  Responses-compatible endpoint.
- Added provider-specific API-key and model-name environment configuration.
- Added the in-progress v0.10.0 long-horizon context-session specification.
- Added bounded `ResearchStateSummary` Structured Outputs with application validation for source
  identity and selected-source evidence levels.
- Added visible context sessions, deterministic projection boundaries, summary replacement, and
  context accounting to every Agent run result.

### Changed

- Removed local vLLM Worker configuration and client routing from the current runtime.
- Changed every LLM stage to use the one provider and model selected for the run.
- Expanded each Worker to 15 model turns, 10 tool attempts, four selected-source reads, three
  context sessions, and two context summaries.
- Requested non-parallel Worker tool calls while accepting provider-returned call batches,
  executing budget-permitted calls synchronously in response order, and projecting every linked
  output.
- Updated the project version, lock metadata, and arXiv `User-Agent` to 0.10.0 without adding a
  dependency.

## [0.9.0] - 2026-08-23

### Added

- Added one run-level `openai` or `local` Worker backend selection while Scope, Supervisor,
  Write, Critic, and Revise remain OpenAI-hosted.
- Added conditional local vLLM endpoint, API-key, and model configuration through a separate
  official OpenAI SDK client.
- Added model source and model name to every `AgentRunRequest`, immutable `AgentRunResult`, and
  terminal result summary.

### Changed

- Reused the existing Responses item loop, tools, retrieval state, limits, and stopping behavior
  for both hosted and local Worker policies without adding a provider adapter or vLLM dependency.
- Made the selected Worker model visible before Scope and kept endpoint values and credentials
  out of terminal output.
- Updated the project version and arXiv `User-Agent` to 0.9.0 without adding a dependency.

## [0.8.0] - 2026-08-23

### Added

- Added `read_source_tool` for one bounded Tavily Extract result selected by a per-Worker source
  ID and focused within-source query.
- Added an application-owned source registry, two-attempt source-read budget, safe destination
  checks, and visible selected-source progress.
- Added source-read usage and its limit to every immutable `AgentRunResult` and terminal summary.

### Changed

- Enriched eligible search results with Worker-local source IDs while keeping discovery distinct
  from selected-source extraction.
- Bounded every read to one registered non-PDF HTTP(S) destination and 6,000 content characters;
  every permitted read consumes both its own budget and the existing total tool-call budget.
- Updated the project version and arXiv `User-Agent` to 0.8.0 without adding a dependency.

## [0.7.0] - 2026-08-23

### Added

- Added frozen `AgentRunRequest` and `AgentRunResult` dataclasses plus one private mutable
  `AgentRunState` for every Research Worker invocation.
- Added bounded Agent run statuses and termination reasons covering completion, tool and turn
  limits, refusal, model and tool errors, reserved context limits, and cancellation.
- Added one terminal Worker summary showing status, termination reason, model-turn usage, and
  tool-call usage for successful and unsuccessful results.

### Changed

- Replaced the Worker's string-or-exception boundary with one immutable result for expected
  operational outcomes while preserving the explicit synchronous Responses function-tool loop.
- Changed `ResearchState` to retain only completed `AgentRunResult` values; failed and cancelled
  results remain visible and stop the run before another Supervisor or report request.
- Updated the project version and arXiv `User-Agent` to 0.7.0 without adding a dependency.

## [0.6.0] - 2026-08-22

### Added

- Added a Pydantic-validated Research Supervisor decision with a zero-or-one next-task
  list.
- Added application-owned `ResearchState` and sequential, isolated Research Workers that
  retain the explicit custom function-tool loop.

### Changed

- Replaced the static Plan stage with visible Supervisor decisions, Worker boundaries,
  central result merging, and an explicit stop reason.
- Removed the unused Supervisor decision summary and nullable nested task, and tightened
  the Supervisor prompt around concise standalone tasks.
- Replaced report-oriented task completion criteria with one through three bounded evidence
  targets; limited each assignment to one primary subject or direct comparison and prohibited
  report-sized Worker assignments.
- Reduced each Worker to six model turns, five tool execution attempts, and three results per
  search; bounded long result text and Supervisor output tokens before adding retries or context
  compaction.
- Removed the fixed Worker output-token limit after real 4,000- and 8,000-token runs demonstrated
  that reasoning tokens could consume the limit before final notes; Worker requests now use the
  configured model's default.
- Made required free-text boundaries distinguish incomplete responses and refusals from completed
  responses with empty text.
- Disabled official SDK retries so failed model requests reach the visible application failure
  boundary without hidden provider-owned attempts.
- Renamed functions and state fields around Scope, Supervisor, Worker, tool execution, response
  validation, report workflow, and top-level execution so the code mirrors the system
  architecture.
- Extracted one Worker tool-call boundary, one Worker lifecycle boundary, and the fixed report
  workflow without adding classes, modules, dependencies, or behavior.
- Clarified that ResearchBrief output requirements and success criteria describe the final
  Markdown report rather than the brief artifact itself.
- Replaced the original question with the approved brief as the sole contract passed into
  supervised research, drafting, critique, and final revision.
- Updated the project version and arXiv `User-Agent` to 0.6.0 without adding a dependency.

## [0.5.0] - 2026-08-22

### Added

- Added one bounded Scope stage that detects material ambiguity and asks at most three
  clarification questions in one round.
- Added a Pydantic-validated `ResearchBrief` with explicit objective, audience, scope,
  exclusions, time horizon, source preferences, output requirements, and success criteria.
- Added explicit brief approval, cancellation, and one optional revision with final
  approval before planning can begin.
- Added local input validation that repeats only the current prompt without consuming a
  model call or expanding the bounded Scope workflow.

### Changed

- Expanded terminal progress from five stages to Scope, Plan, Research, Write, Critic, and
  Revise.
- Replaced the original question with the approved brief as the sole contract passed into
  planning, task research, drafting, critique, and final revision.
- Updated the project version and arXiv `User-Agent` to 0.5.0 without adding a dependency.

## [0.4.0] - 2026-08-22

### Added

- Added a one-shot Pydantic-validated research plan before tool-using research.
- Added visible ordered plan tasks and completion criteria.
- Added sequential task execution with fresh Responses API history and independent Agent
  limits for every task.
- Added combined research notes as the evidence input to Write, Critic, and Revise without
  exposing plan or task metadata to the report stages.
- Added `agent_instructions.py` as the single home for model-stage instructions and
  application-owned model input text.

### Changed

- Expanded terminal progress from four stages to Plan, Research, Write, Critic, and Revise.
- Added Pydantic as a direct runtime dependency and updated the project version to 0.4.0.
- Expanded the Plan, Research, Write, Critic, and Revise instructions with explicit
  evidence, scope, and output-boundary requirements.
- Replaced the separate Research synthesis instructions with a budget-exhausted input and
  a final Research request using `tool_choice="none"`.
- Kept the static plan immutable and deferred approval, replanning, adaptive supervision,
  and concurrency to later releases.

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

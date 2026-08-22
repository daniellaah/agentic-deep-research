# Changelog

All notable changes to this project will be documented in this file.

The project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

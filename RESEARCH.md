# Research Notes

Last reviewed: 2026-08-21

## Purpose

This document connects external research to project decisions. It is not a general reading
list. A source belongs here when it changes the roadmap, clarifies a mechanism, or provides
a useful comparison for a planned release.

API capabilities are time-sensitive and must be rechecked when their release is specified.
Research claims should be treated as evidence, not permanent architecture rules.

## Current Synthesis

### Deep research is an active, multi-stage process

Recent surveys consistently distinguish deep research from single-shot retrieval. The
useful baseline includes planning, query development, iterative information acquisition,
and report synthesis. The roadmap introduces those capabilities separately so their
effects remain visible.

### A visible tool loop should precede hosted research features

The Responses API supports custom function tools, built-in tools, MCP, streaming,
conversation continuation, and tool limits. Starting with a custom function tool keeps the
application-owned loop, arguments, outputs, stopping decision, and trace observable before
hosted features are compared.

### Evidence should become explicit before planning becomes complex

Adaptive planning without durable evidence risks repeatedly searching for the same
information or revising a plan from lossy conversational summaries. Paper and web
retrieval therefore precede an explicit evidence ledger, which then precedes adaptive
planning.

### Draft-first refinement is a late report-quality technique

Test-Time Diffusion Deep Researcher treats a preliminary draft as an evolving research
artifact and retrieves information to improve weak sections. This project adopts the idea
only after it has a gather-then-write baseline, evidence provenance, and citation checks.

### Multi-agent research is conditional, not the default

Production research systems demonstrate an orchestrator-worker pattern for breadth-first
questions. Controlled research also shows that multi-agent systems can hurt sequential
tasks and add coordination cost. The project therefore implements parallel research before
multi-agent delegation and retains a single-agent path.

## Source Map

| Source | Relevant finding | Roadmap effect |
| --- | --- | --- |
| [OpenAI Responses API reference](https://developers.openai.com/api/reference/cli/resources/responses/methods/create) | Responses can use custom functions, built-in tools, MCP, streaming, continuation state, parallel calls, and explicit limits | Implement a custom loop first; reserve hosted capabilities for later comparisons |
| [OpenAI model guidance](https://developers.openai.com/api/docs/guides/latest-model) | Current model guidance includes persisted reasoning, compaction-related practices, programmatic tool calling, and multi-agent capabilities | Recheck these features during long-context and frontier releases instead of binding early code to them |
| [Deep Research: A Survey of Autonomous Research Agents](https://arxiv.org/abs/2508.12752) | Organizes deep research around planning, question development, web exploration, and report generation | Supports the staged progression from retrieval to planning and synthesis |
| [Deep Research: A Systematic Survey](https://arxiv.org/abs/2512.02038) | Highlights query planning, information acquisition, memory management, and answer generation as core components | Places memory and context work after real acquisition and planning state exist |
| [Deep Researcher with Test-Time Diffusion](https://research.google/pubs/deep-researcher-with-test-time-diffusion/) | Uses a preliminary draft as an evolving skeleton and retrieval-guided iterative refinement | Motivates the draft-first release after an evidence-grounded report baseline |
| [How Anthropic built its multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) | Describes a lead researcher, parallel subagents, persistent planning context, adaptive continuation, and citation processing | Provides a production comparison for later orchestrator-worker and citation releases |
| [Towards a science of scaling agent systems](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/) | Reports gains on parallelizable work and penalties on sequential tasks, with coordination and error-amplification tradeoffs | Requires task-shape justification and a retained single-agent path before multi-agent execution |

## Adoption Rules

Before adding a research technique to a release specification, record:

1. the limitation observed in the current trace or report;
2. the technique's proposed mechanism;
3. the baseline it will be compared against;
4. the expected observable change;
5. the added cost, coordination, or failure modes; and
6. the conditions under which the technique should not be used.

## Deferred Topics

The following topics are relevant but deliberately lack scheduled implementation details:

- formal report-quality evaluation and benchmark harnesses;
- reinforcement-learning training of the research policy;
- browser GUI automation;
- unrestricted code execution;
- long-term cross-user memory;
- distributed agent infrastructure.

They may be researched later, but they are not current project commitments.

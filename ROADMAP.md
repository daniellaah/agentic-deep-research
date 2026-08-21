# Roadmap

## Roadmap Policy

The roadmap is a learning sequence, not a promise to implement every named technique
unchanged. Each release builds cumulatively on `deep_research.py`, remains runnable, and
receives a Git tag only after manual verification.

Near-term releases are concrete. Later releases are provisional and must be reviewed
against observed trace failures and current research before their specifications are
written.

## Release Ladder

| Version | Learning focus | Observable outcome | Status |
| --- | --- | --- | --- |
| 0.1.0 | Minimal observable function-tool loop | One local tool can be called; the run produces live progress, a JSONL trace, and a Markdown report | Spec drafted |
| 0.2.0 | Paper research | The agent searches scholarly metadata and produces a source-backed short report | Planned |
| 0.3.0 | Web research | The agent searches the web, reads selected sources, and combines paper and web evidence | Planned |
| 0.4.0 | Evidence and provenance | Findings are normalized, deduplicated, and linked to sources in an explicit evidence ledger | Planned |
| 0.5.0 | Research planning and approval | The agent decomposes a question, shows its plan, and waits for one user approval before execution | Planned |
| 0.6.0 | Adaptive exploration | The agent detects gaps and contradictions, revises queries, and stops using explicit sufficiency rules | Planned |
| 0.7.0 | Long-horizon harness | Budgets, retries, context management, compaction, and recoverable failures are visible in the trace | Planned |
| 0.8.0 | Report and citation verification | Claims are checked against evidence and unsupported or weakly supported claims are surfaced | Planned |
| 0.9.0 | Draft-first refinement | A preliminary report guides retrieval and is iteratively revised with new evidence | Planned |
| 0.10.0 | Parallel research | Independent research branches run concurrently under explicit concurrency and budget limits | Planned |
| 0.11.0 | Orchestrator-worker research | A lead agent delegates suitable subproblems, validates findings, and synthesizes the report | Planned |
| 0.12.0 | Durable human collaboration | Runs can pause, resume, accept bounded steering, and preserve approved research state | Planned |
| 0.13.0 | Frontier comparisons | Selected hosted tools, programmatic tool calling, memory, or multi-agent APIs are compared with the visible baseline | Planned |

## Phase 1: Visible Agent Foundations

### 0.1.0 — Minimal Observable Function-Tool Loop

Establish the smallest complete AgentRun:

- one initial research question;
- explicit Responses API iteration;
- one deterministic local function tool;
- bounded stopping behavior;
- live event-level progress;
- structured JSONL trace;
- final Markdown report;
- trace preservation on failure.

This release makes the harness visible before real retrieval introduces network and source
quality concerns.

## Phase 2: Evidence Acquisition

### 0.2.0 — Paper Research

Add a paper-search function tool and enough source metadata to support a short
evidence-backed report. The release should expose query construction, result selection,
and source attribution.

### 0.3.0 — Web Research

Add web search and source reading. The release should make search results, selected pages,
failed fetches, and extracted information visible.

### 0.4.0 — Evidence and Provenance

Separate collected evidence from conversational context. Introduce stable source identity,
deduplication, claim support, and provenance without splitting the runtime into modules.

## Phase 3: Research Control

### 0.5.0 — Research Planning and Approval

Create an explicit research plan before execution. This is the first release with mid-run
input: the user may approve, reject, or revise the proposed plan once.

### 0.6.0 — Adaptive Exploration

Use the evolving plan and evidence state to decide what to research next. Add explicit gap,
contradiction, marginal-value, and sufficiency signals.

### 0.7.0 — Long-Horizon Harness

Introduce practical controls only after real long runs expose the need for them:

- iteration, tool, token, time, and concurrency budgets;
- transient retry policy;
- partial failure semantics;
- context pruning and compaction;
- resumable checkpoints where justified.

## Phase 4: Report Reliability

### 0.8.0 — Report and Citation Verification

Audit whether important claims are supported by the cited evidence. Record verification
results in the trace and make uncertainty visible in the report.

### 0.9.0 — Draft-First Refinement

Create a preliminary report skeleton, use its weak sections to guide further retrieval, and
revise it iteratively. Compare this process with the earlier gather-then-write baseline.

## Phase 5: Scaling Research

### 0.10.0 — Parallel Research

Run genuinely independent retrieval branches concurrently. Preserve branch identity,
resource use, and evidence provenance in a single trace.

### 0.11.0 — Orchestrator-Worker Research

Introduce specialized worker agents only for decomposable research tasks. The orchestrator
owns delegation, overlap control, validation, budgets, and synthesis.

### 0.12.0 — Durable Human Collaboration

Support pause, resume, bounded steering, and durable approved state without turning the
CLI into an open-ended chat application.

## Phase 6: Frontier Comparisons

### 0.13.0 and Later

Evaluate selected techniques against the accumulated visible baseline, including:

- OpenAI built-in web search versus application-owned search;
- manual history management versus response continuation and compaction;
- direct tool calls versus programmatic tool calling;
- single-agent versus hosted or application-managed multi-agent execution;
- cross-run memory and experience distillation;
- self-critique, candidate generation, and self-improvement loops.

A frontier technique enters an implementation spec only when it has a clear learning
question, a suitable baseline, and an observable comparison.

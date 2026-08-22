# Roadmap

## Roadmap Policy

The roadmap is a cumulative learning sequence, not a commitment to preserve every future
design unchanged. Each release adds one primary mechanism to the same small runnable
program and should make that mechanism observable without hiding it behind an agent
framework.

Near-term releases through 0.7.0 are concrete enough to guide specifications. Work beyond
0.7.0 is intentionally described as one broad direction and will be divided into releases
only after earlier runs and reports reveal the next useful learning problems.

Every release must be specified before implementation, manually verified before tagging,
and kept as simple as its learning objective allows.

## Release Ladder

| Version | Primary mechanism | Observable outcome | Status |
| --- | --- | --- | --- |
| 0.1.0 | One Responses API call | One question produces live progress and a Markdown report | Released |
| 0.2.0 | Fixed report-refinement workflow | Separate write, critic, and revise calls expose the value and limits of deterministic orchestration | Verified |
| 0.3.0 | Tool-using research Agent | A visible custom tool loop searches the web and arXiv, then produces a source-grounded report through write, critic, and revise stages | Verified |
| 0.4.0 | Static structured planning | A one-shot planner creates a validated research plan that the research Agent executes sequentially | Verified |
| 0.5.0 | Scoping and ResearchBrief | Bounded clarification and user approval establish an explicit research contract before planning | Planned |
| 0.6.0 | Adaptive supervisor-worker research | A supervisor repeatedly observes shared research state and delegates one next task to an isolated worker until sufficient or budget-limited | Planned |
| 0.7.0 | Bounded parallel research | The supervisor dispatches independent worker tasks concurrently while preserving budgets, provenance, and failure visibility | Planned |
| Later | Advanced deep-research reliability and scale | Later mechanisms are selected from failures observed in the implemented releases | Direction |

## Phase 1: Model and Workflow Foundations

### 0.1.0 — One Responses API Call

Establish the smallest complete runnable baseline:

- one initial question;
- one synchronous Responses API call;
- one explicit `llm_call` provider boundary;
- concise live progress around that call;
- final output saved as a Markdown report; and
- no structured trace for the single linear call.

This release deliberately has no tool, loop, history, iteration model, or Agent
abstraction.

### 0.2.0 — Fixed Write-Critic-Revise Workflow

Introduce deterministic multi-call orchestration before an autonomous Agent loop:

```text
question -> write -> critic -> revise -> report
```

- `write` creates the initial report draft;
- `critic` reviews structure, reasoning, completeness, and instruction following;
- `revise` receives the draft and critique and produces the final report;
- the run saves `draft.md`, `critique.md`, and `report.md`; and
- live progress makes each fixed stage and its outcome visible.

The three calls do not share an Agent loop. Each stage receives only the explicit upstream
artifacts it needs. Because this release has no research tools or external evidence, the
critic must not claim to verify factual accuracy or citations. A structured Agent trace is
still unnecessary; the intermediate artifacts are the observable workflow record.

## Phase 2: Research and Planning

### 0.3.0 — Tool-Using Research Agent

Introduce the first actual Agent:

```text
question -> research Agent -> write -> critic -> revise -> report
```

- implement an explicit synchronous function-tool loop around `llm_call`;
- provide simple Tavily web search and arXiv paper search as custom function tools;
- let the application own and replay the ordered Responses API input history rather than
  use `previous_response_id` or the Conversations API;
- bound tool calls and loop iterations with explicit stopping behavior;
- preserve source metadata in structured research results;
- keep tool schemas and implementations in `agent_tools.py`; and
- expose execution through live progress and the final report printed in the terminal.

The collected research becomes the evidence input to the existing write-critic-revise
workflow. The report should cite its sources, but formal claim-level citation verification
is deferred. Planning, Pydantic models, parallel execution, and a supervisor are also out
of scope so the tool loop remains easy to inspect. Structured tracing is deferred to a
later release whose learning question requires it.

### 0.4.0 — Static Structured Research Planning

Add a one-shot planner stage before research:

```text
question -> static plan -> sequential research -> write -> critic -> revise -> report
```

- generate one bounded plan with explicit research tasks and completion criteria;
- validate model-generated planning data with Pydantic models such as `ResearchPlan` and
  `ResearchTask`;
- execute the plan sequentially with the existing research Agent;
- associate research results with their plan task; and
- show the plan in live output, then continue without a mid-run approval point.

This is a planner stage, not a stateful planner Agent. The plan is immutable during the
run, which creates a clear baseline for later adaptive control. Pydantic is limited to
validated model or tool boundaries; internal history remains plain data.

### 0.5.0 — Scoping and ResearchBrief

Turn the user's initial question into an approved research contract before planning:

```text
question -> scoping -> ResearchBrief approval -> static plan -> research -> report workflow
```

- detect ambiguities that would materially change the research;
- when needed, ask at most three targeted questions in one clarification round;
- create a structured `ResearchBrief` covering objective, audience, scope, exclusions,
  time horizon, source preferences, output requirements, and success criteria;
- show the brief and allow one bounded approval or revision point; and
- pass only the approved brief into planning and downstream research.

This is the first release with mid-run user interaction. It remains a bounded research
intake flow rather than an open-ended chat experience.

## Phase 3: Adaptive Research

### 0.6.0 — Adaptive Supervisor-Worker Research

Replace the immutable static plan with a living, observation-driven research state:

```text
ResearchBrief
    -> initialize ResearchState
    -> supervisor observes state
    -> dispatch one isolated research worker
    -> merge structured result
    -> repeat or finish
    -> write -> critic -> revise -> report
```

- maintain explicit task, evidence, gap, contradiction, failure, and budget ledgers in
  `ResearchState`;
- require each supervisor turn to return a validated `SupervisorDecision`, such as
  dispatching a next task or finishing;
- give every research worker a fresh, independent Responses API history containing only
  the approved brief, its task, relevant evidence, available tools, and worker budget;
- return a structured `ResearchResult` rather than the worker's full transcript to the
  supervisor;
- let the supervisor observe ledgers and concise decision summaries without recording or
  claiming to expose private chain-of-thought; and
- enforce hard limits for supervisor rounds, worker tasks, tool calls, tokens, elapsed
  time, and estimated cost.

The supervisor may finish early when coverage and evidence are sufficient, contradictions
are addressed, important gaps are exhausted, or another task has low expected value. Hard
budgets remain authoritative. Workers execute sequentially in this release so adaptive
delegation and context isolation can be understood before concurrency is introduced.

### 0.7.0 — Bounded Parallel Research Workers

Add concurrency only after the sequential supervisor-worker control loop is observable:

- let the supervisor emit a bounded batch of independent research tasks;
- execute workers concurrently under explicit global and per-worker limits;
- keep each worker's context, tool budget, task identity, and trace branch isolated;
- merge results centrally into the shared evidence and task ledgers;
- surface partial failures, retries, cancellations, overlap, and deduplication; and
- preserve parent-child trace relationships and deterministic final synthesis inputs.

Parallelism is a scheduling optimization, not a new research policy. Dependent tasks stay
sequential, and the system must retain a single-worker path for questions that do not
benefit from decomposition.

## Later Direction: Advanced Deep-Research Reliability and Scale

After 0.7.0, improve evidence reliability, citation verification, long-horizon execution,
recoverability, human collaboration, context management, and frontier architecture
comparisons based on failures observed in the earlier releases. These capabilities remain
unscheduled until the implemented runs, reports, costs, and failure modes provide a
clear reason to split them into new specifications.

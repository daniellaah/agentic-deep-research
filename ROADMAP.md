# Roadmap

## Roadmap Policy

The roadmap is a cumulative learning sequence, not a commitment to preserve every future
design unchanged. Each release adds one primary mechanism to the same small runnable
program and should make that mechanism observable without hiding it behind an agent
framework.

Near-term releases through 0.8.0 are concrete enough to guide specifications. Work beyond
0.8.0 is intentionally described as one broad direction and will be divided into releases
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
| 0.5.0 | Scoping and ResearchBrief | Bounded clarification and user approval establish an explicit research contract before planning | Verified |
| 0.6.0 | Sequential research supervisor and workers | A supervisor repeatedly observes application-owned research state and delegates one bounded task to an isolated worker until it decides to finish or reaches a hard limit | In progress |
| 0.7.0 | Bounded parallel research workers | The supervisor may dispatch independent worker tasks concurrently while preserving centralized state ownership, budgets, and failure visibility | Planned |
| 0.8.0 | Persistent research state and recovery | An interrupted run can resume from an application-owned snapshot without repeating completed research work | Planned |
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

## Phase 3: Adaptive Multi-Agent Research

### 0.6.0 — Sequential Research Supervisor and Workers

Replace the immutable static plan with an application-owned, observation-driven research
loop while retaining deterministic scoping and report workflows:

```text
question
    -> Scope workflow
    -> approved ResearchBrief
    -> initialize ResearchState
    -> Research Supervisor observes state
       -> delegate one bounded task to an isolated Research Worker
       -> application merges the worker result into ResearchState
       -> observe again or finish
    -> Report workflow
    -> final report
```

- keep clarification, ResearchBrief approval, and write-critic-revise as deterministic
  workflow stages rather than naming every model call as an Agent;
- make the Research Supervisor the single owner of research strategy: it decides the next
  task or decides that research should finish;
- invoke each Research Worker as a bounded capability with one task, a fresh Responses API
  history, the existing research tools, and explicit loop and tool-call limits;
- let the application own `ResearchState`, worker dispatch, result merging, and all hard
  limits instead of allowing workers to mutate shared state;
- keep only the approved brief, delegated tasks, compact worker results, budget usage, and
  stop reason in the initial `ResearchState` representation;
- return bounded task results to the supervisor without transferring control of the user
  conversation or exposing full worker histories;
- execute workers synchronously, one at a time, so the supervisor loop, context isolation,
  and stopping behavior remain directly observable; and
- show supervisor decisions and worker lifecycle events in terminal progress without adding
  structured tracing or an evaluation framework.

This is a hybrid architecture: a deterministic outer workflow contains one adaptive
supervisor-worker research subsystem. In the terminology of the
[OpenAI orchestration guide](https://developers.openai.com/api/docs/guides/agents/orchestration#choose-the-orchestration-pattern),
workers behave as bounded capabilities while the supervisor keeps control. The project
continues to implement the loop directly with the Responses API; it does not introduce an
Agent framework, generic Agent definitions, parallel execution, worker-to-worker delegation,
research-quality gates, structured tracing, or evaluation in this release.

### 0.7.0 — Bounded Parallel Research Workers

Add concurrency only after the sequential supervisor-worker control loop is observable:

- let the supervisor emit a bounded batch of independent research tasks;
- execute workers concurrently under explicit global and per-worker limits;
- keep each worker's context, tool budget, task identity, and execution branch isolated;
- merge results centrally into the shared evidence and task ledgers;
- surface partial failures, retries, cancellations, overlap, and deduplication; and
- preserve parent-child task relationships in shared state and deterministic final synthesis
  inputs.

Parallelism is a scheduling optimization, not a new research policy. Dependent tasks stay
sequential, and the system must retain a single-worker path for questions that do not
benefit from decomposition.

## Phase 4: Long-Running Research

### 0.8.0 — Persistent Research State and Recovery

Make application-owned research state durable before adding more advanced reliability and
quality mechanisms:

- persist the approved brief, supervisor status, task records, compact worker results,
  budgets, and stop reason in a versioned run snapshot under `runs/`;
- resume an interrupted run without repeating completed worker tasks;
- distinguish pending, active, completed, and failed work so restart behavior is explicit;
- preserve sequential and bounded-parallel execution paths behind the same recovery model;
- write snapshots at deterministic application-owned boundaries rather than from workers;
  and
- surface recovery, retry, and unrecoverable-state decisions in terminal output.

This release adds resumability, not autonomous background execution or distributed task
coordination. Context compression, durable cross-run memory, research-quality evaluation,
structured tracing, and production-grade checkpoint infrastructure remain later concerns.

## Later Direction: Advanced Deep-Research Reliability and Scale

After 0.8.0, improve evidence reliability, citation verification, human collaboration,
context management, durable memory, structured tracing, repeatable evaluation, and frontier
architecture comparisons based on failures observed in the earlier releases. These capabilities
remain unscheduled until the implemented runs, reports, costs, and failure modes provide a clear
reason to split them into new specifications.

# Roadmap

## Roadmap Policy

The roadmap is a cumulative learning sequence, not a commitment to preserve every future
design unchanged. Each release adds one primary mechanism to the same small runnable
program and should make that mechanism observable without hiding it behind an agent
framework.

Near-term releases through 0.13.0 define a harness-first sequence: establish a stable Worker
run contract, deepen retrieval, add an optional local open-weight Worker, manage long-horizon
context, improve multi-agent scheduling, and only then add batch rollout generation. Work
beyond 0.13.0 remains a broad direction until real runs and rollouts reveal the next useful
learning problems.

Every release must be specified before implementation, manually verified before tagging,
and kept as simple as its learning objective allows.

## Release Ladder

| Version | Primary mechanism | Observable outcome | Status |
| --- | --- | --- | --- |
| 0.1.0 | One Responses API call | One question produces live progress and a Markdown report | Released |
| 0.2.0 | Fixed report-refinement workflow | Separate write, critic, and revise calls expose the value and limits of deterministic orchestration | Released |
| 0.3.0 | Tool-using research Agent | A visible custom tool loop searches the web and arXiv, then produces a source-grounded report through write, critic, and revise stages | Released |
| 0.4.0 | Static structured planning | A one-shot planner creates a validated research plan that the research Agent executes sequentially | Released |
| 0.5.0 | Scoping and ResearchBrief | Bounded clarification and user approval establish an explicit research contract before planning | Released |
| 0.6.0 | Sequential research supervisor and workers | A supervisor repeatedly observes application-owned research state and delegates one bounded task to an isolated worker until it decides to finish or reaches a hard limit | Released |
| 0.7.0 | Explicit Agent run contract | Every Worker returns a structured application-owned result with visible status, termination reason, and budget usage | Released |
| 0.8.0 | Deep retrieval Worker | A Worker searches, selects, and reads bounded source content before producing research notes | Released |
| 0.9.0 | Local open-weight Worker | The same Worker harness can run against either OpenAI or a local vLLM Responses-compatible endpoint | Verified |
| 0.10.0 | Long-horizon context sessions | A Worker crosses visible in-memory session boundaries through bounded context summaries instead of replaying unbounded history | Planned |
| 0.11.0 | Bounded task graph and parallel workers | The Supervisor may create a small dependency-aware task batch whose ready tasks run concurrently | Planned |
| 0.12.0 | Failure-aware adaptive orchestration | Failed work becomes visible in shared state so the Supervisor can replace, narrow, or abandon it within hard limits | Planned |
| 0.13.0 | Batch rollout runner | JSON or JSONL tasks can produce multiple concurrent rollouts through the same Worker harness and result contract | Planned |
| Later | Training readiness, reliability, and research quality | Persistence, structured trajectories, evaluation, evidence reliability, SFT, and RL are selected from observed rollout failures | Direction |

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

## Phase 4: QUEST-Aligned Worker Harness

### 0.7.0 — Explicit Agent Run Contract

Replace the Worker's implicit string-or-exception boundary with one visible, application-owned
run contract:

- define the request, mutable run state, and completed run result needed by one Worker;
- retain ordered Responses items, turn usage, tool usage, status, and stopping information in
  that state without introducing a workflow framework;
- use a bounded termination vocabulary such as completed, turn limit, tool limit, context
  limit, refusal, model error, tool error, and cancellation;
- return one result containing the final notes, termination reason, and derived budget usage;
- print the result summary at the Worker boundary; and
- preserve the current synchronous OpenAI execution path, tools, Scope workflow, Supervisor
  policy, and report workflow.

This release introduces the smallest named runtime representation justified by future local
inference and rollout consumers. It does not add selected-page reading, another provider,
concurrency, persistence, structured tracing, evaluation, or training.

### 0.8.0 — Deep Retrieval Worker

Turn search results into sources that a Worker can deliberately select and read:

```text
search -> select returned source -> visit/read -> continue or finish
```

- add one bounded selected-source reading tool;
- permit reading only source identifiers or URLs returned by the active Worker's searches;
- enforce destination, content-size, result-count, and per-Worker reading limits in the
  application;
- keep search discovery distinct from page-content extraction;
- make selected-source activity and remaining budget visible in the terminal; and
- continue to produce one complete final Markdown report through the existing workflow.

This release deepens the Worker's environment before increasing orchestration scale. Full-paper
ingestion, arbitrary URL fetching, citation verification, caching, and evidence scoring remain
out of scope.

### 0.9.0 — Local Open-Weight Worker

Run the research policy locally without replacing the complete hosted workflow at once:

- let only the Research Worker choose between the current OpenAI endpoint and a local
  open-weight model served through a vLLM Responses-compatible endpoint;
- keep the official OpenAI Python SDK and the explicit Responses item loop as the shared client
  and protocol when the selected local runtime supports them;
- configure the Worker model and endpoint separately from Scope, Supervisor, and report stages;
- keep OpenAI-hosted models available as teachers, baselines, and acceptance references;
- complete one real run with a small open-weight model; and
- document any selected-model limitations in reasoning-item replay, function tools, or response
  formats instead of hiding them behind a general provider gateway.

This release adds no SFT, RL, batch rollout generation, provider-neutral API, or local model
server managed by the application.

### 0.10.0 — Long-Horizon Context Sessions

Let one Worker continue beyond a single replayable context without making state durable:

- establish an application-owned context budget distinct from turn and tool budgets;
- trigger a visible session boundary before the active history becomes unsafe to replay;
- produce a bounded in-memory ResearchStateSummary containing completed work, visited sources,
  unresolved questions, and the next useful actions;
- start the next session from the task, approved brief, and validated summary rather than the
  complete old history;
- retain run-level budget and termination accounting across sessions; and
- keep the full mechanism observable without writing a trace or recovery snapshot.

This release studies context engineering, not persistent memory. Cross-run memory, resume,
formal evidence state, and training trajectory output remain later concerns.

## Phase 5: Adaptive Multi-Agent Scheduling

### 0.11.0 — Bounded Task Graph and Parallel Workers

Add concurrency only after the Worker harness, retrieval depth, local runtime path, and context
boundaries are explicit:

- let the Supervisor emit a bounded batch of tasks with stable task identifiers and dependency
  declarations;
- derive ready work from the task graph instead of asking models to mutate shared state;
- execute independent ready tasks concurrently under explicit global and per-Worker limits;
- keep dependent work sequential and retain a single-Worker path;
- isolate each Worker's context, tools, budgets, and result; and
- merge completed results centrally in deterministic task order for the next Supervisor decision
  and final synthesis.

Parallelism remains a scheduling mechanism rather than a second research policy. Retries,
replacement tasks, durable queues, distributed execution, and quality scoring remain out of
scope.

### 0.12.0 — Failure-Aware Adaptive Orchestration

Let the research subsystem respond to failed bounded work without terminating immediately:

- represent task lifecycle as pending, running, completed, failed, or cancelled in in-memory
  application state;
- record a failed result separately from successful Worker notes;
- let the Supervisor choose a bounded replacement, narrower follow-up, or explicit abandonment;
- permit a bounded context handoff containing only the prior result or failure information needed
  by a dependent task;
- prevent repeated equivalent work and infinite recovery loops through application-owned limits;
  and
- make partial completion and final stop reasons visible before the report workflow.

This release adds adaptive failure handling, not persistence, background jobs, worker-to-worker
communication, or a general task system.

## Phase 6: Rollout-Ready Execution

### 0.13.0 — Batch Rollout Runner

Add a second entry path for repeatable research-policy sampling while preserving the interactive
CLI:

- accept a documented JSON or JSONL task format;
- generate a configured number of independent rollouts for each task;
- assign stable task and rollout identifiers and record model and sampling configuration;
- schedule independent task-rollout pairs concurrently under one global limit;
- return the same Agent run result contract used by interactive Workers;
- write generated rollout artifacts under `runs/` without treating them as evaluation scores;
  and
- keep the interactive Scope, approval, Supervisor, and report experience complete and runnable.

This release provides a QUEST-style batch execution boundary, not SFT, RL, reward computation,
benchmark grading, trajectory quality filtering, or interrupted-run recovery.

## Later Direction: Training Readiness, Reliability, and Research Quality

After 0.13.0, select releases from failures observed in real local-model and batch rollouts.
Candidate mechanisms include persistent run state and recovery, structured trajectory recording,
repeatable evaluation, evidence and citation reliability, SFT data preparation, a first bounded
SFT experiment, and agentic RL. Their order and release boundaries remain intentionally
unscheduled until the harness produces the costs, failures, and behavior differences needed to
justify them.

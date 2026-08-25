# Roadmap

## Roadmap Policy

The roadmap is a cumulative learning sequence, not a commitment to preserve every future
design unchanged. Each release adds one primary mechanism to the same small runnable
program and should make that mechanism observable without hiding it behind an agent
framework.

Near-term releases through 0.18.0 define a reliability-first sequence: establish a stable Worker
run contract, deepen retrieval, compare explicit hosted providers, manage long-horizon context,
persist evidence, verify claim provenance, recover interrupted runs, make retries idempotent,
evaluate context quality, and only then add parallel scheduling and batch rollout generation.
Training work remains a later direction until measured artifacts justify it.

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
| 0.9.0 | Local open-weight Worker | The same Worker harness can run against either OpenAI or a local vLLM Responses-compatible endpoint | Released |
| 0.10.0 | Long-horizon context sessions and global hosted-provider selection | One selected OpenAI or DeepSeek model runs the complete workflow, and each Worker crosses visible in-memory session boundaries through bounded context summaries instead of replaying unbounded history | Released |
| 0.11.0 | Persistent evidence ledger | Every discovered or read source becomes an immutable run-scoped evidence record that survives context replacement | Draft |
| 0.12.0 | Structured claims and citation verification | Worker claims link to persisted evidence and a separate citation boundary validates report provenance | Draft |
| 0.13.0 | Durable run checkpoints and resume | Stable workflow boundaries persist versioned state that can resume without repeating completed work | Draft |
| 0.14.0 | Idempotent tools and failure-aware orchestration | External calls replay safely and failed work becomes a bounded Supervisor decision instead of a fatal exception | Draft |
| 0.15.0 | Hybrid context memory | Pinned state, validated summary, recent complete boundaries, and retrieved evidence replace summary-only continuation | Draft |
| 0.16.0 | Structured traces and evaluation harness | Versioned application events and deterministic graders make quality, recovery, cost, and context retention comparable | Draft |
| 0.17.0 | Bounded parallel task graph | Ready tasks run concurrently under shared provenance, persistence, rate, and global-budget contracts | Draft |
| 0.18.0 | Batch rollout runner | Versioned JSONL tasks produce auditable independent rollouts through the same durable run harness | Draft |
| Later | Training readiness | SFT, reward modeling, and RL are selected only from evaluated high-quality trajectories | Direction |

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

- retire the release 0.9.0 local vLLM runtime path and require one run-level `LLM_PROVIDER`
  selection, `openai` or `deepseek`, for every Scope, Supervisor, Worker, summary, Write,
  Critic, and Revise model request;
- read the selected provider's API key and model name from environment configuration, with
  DeepSeek V4 Flash as the real-endpoint acceptance baseline rather than a hard-coded model;
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

## Phase 5: Evidence Reliability

### 0.11.0 — Persistent Evidence Ledger

Persist every normalized search and selected-source result before it enters transient model
history. Assign stable run-scoped evidence IDs, hashes, and tool provenance so evidence survives
context replacement without introducing full workflow resume.

### 0.12.0 — Structured Claims and Citation Verification

Replace free-form Worker notes at the control boundary with structured claims linked to evidence
IDs. Add a separate citation decision and deterministic validator so report URLs and excerpts must
resolve to persisted evidence. Provenance becomes checkable without claiming universal truth
verification.

## Phase 6: Durable and Failure-Aware Execution

### 0.13.0 — Durable Run Checkpoints and Resume

Persist versioned application state at stable workflow boundaries. Add run inspection, pause,
cancel, and resume without repeating committed model or tool work. Keep credentials and private
reasoning outside checkpoints.

### 0.14.0 — Idempotent Tools and Failure-Aware Orchestration

Give external calls stable execution identities, bounded typed retry, and replay-safe results.
Merge failed Worker results into ResearchState so the Supervisor can narrow, replace, accept
partial evidence, abandon, or finish within one global run budget.

## Phase 7: Context Quality and Measurement

### 0.15.0 — Hybrid Context Memory

Compare the summary-only v0.10 baseline with a deterministic context assembled from immutable
pinned state, validated summary, persistent evidence retrieval, and recent complete response/tool
boundaries. Measure retained constraints, evidence, failures, and next actions.

### 0.16.0 — Structured Traces and Evaluation Harness

Emit versioned causal application events and execute a fixed evaluation corpus with deterministic
graders and optional model graders. Establish sequential quality, provenance, recovery, cost, and
latency baselines before adding concurrency. Traces never claim to expose private chain-of-thought.

## Phase 8: Measured Scale

### 0.17.0 — Bounded Parallel Task Graph

Let the Supervisor issue a small acyclic task batch. Schedule ready isolated Workers under shared
global budgets, durable reservations, per-domain limits, cancellation, and deterministic result
ordering. Accept a parallel default only when v0.16 evaluation demonstrates a declared benefit.

### 0.18.0 — Batch Rollout Runner

Accept versioned JSONL tasks and generate independent auditable rollouts through the same durable
interactive harness. Preserve failures, resume partial batches, enforce batch-level budgets, and
aggregate exact configuration and evaluation results.

## Later Direction: Training Readiness

After 0.18.0, select SFT data preparation, reward modeling, and agentic RL only from failures and
high-quality trajectories measured by the evaluation harness. Training must not silently replace
the application's run, evidence, tool, budget, or stopping semantics.

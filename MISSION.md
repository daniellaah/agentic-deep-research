# Mission

## Purpose

Build a deep-research command-line agent to learn how modern agents work by implementing
their mechanisms directly, one observable release at a time.

The project favors understanding over product breadth. Its runtime remains in three focused
root-level Python files: `deep_research.py` for the workflow and Agent loop,
`agent_instructions.py` for model instructions, and `agent_tools.py` for custom tool
definitions and implementations.

## Primary Outcome

A contributor should be able to use the repository to understand and experiment with:

- the OpenAI Responses API and response-item model;
- the difference between hosted research policies from explicitly supported model providers
  using the same visible Responses API Agent loop;
- the difference between a single model call and an agent loop;
- a complete function-tool calling loop;
- agent harness concerns such as run contracts, limits, errors, state, and stopping;
- live progress and terminal-visible workflow output;
- paper and web research tools that separate source discovery from selected-source reading;
- planning, adaptive retrieval, evidence tracking, and citation;
- context engineering and long-horizon execution;
- critique, verification, and iterative report refinement;
- parallel research and orchestrator-worker multi-agent systems;
- repeatable batch rollout generation through the same Worker harness used by interactive runs;
- human approval, recovery, memory, and selected frontier techniques.

## Design Principles

### Learn mechanisms by making them visible

Start with one direct provider call, then implement Agent behavior explicitly before an
equivalent hosted or framework feature is adopted. Provider-managed capabilities may
later be introduced as comparisons.

### Grow through complete vertical slices

Every release must accept a question, run to completion, expose what happened, and produce
a final Markdown report. A release may be small, but it must not be a disconnected code
fragment.

### Keep one small cumulative runtime

Workflow behavior lives in the repository-root `deep_research.py`; model instructions live
in `agent_instructions.py`; custom tool definitions and implementations live in
`agent_tools.py`. New releases evolve these focused files instead of creating parallel
implementations or architectural layers.

### Use the simplest adequate representation

Prefer local variables, functions, dictionaries, and lists. Introduce named data
structures, asynchronous execution, or abstractions only when a concrete release becomes
harder to understand without them.

### Treat observability as product behavior

Release 0.1.0 uses live terminal progress for its single linear model call. Release 0.3.0
keeps the tool-using Agent loop observable entirely through terminal progress and the
printed final report. Structured tracing is deferred until a later learning question
requires it.

### Let observed failures justify complexity

Planning, memory, critique, parallelism, and multi-agent coordination should be introduced
only after an earlier release makes the problem they solve observable.

### Keep inference behavior reusable

Interactive research, hosted-model comparisons, and later batch rollouts should reuse one
explicit Worker run contract and stopping model. A future training adapter may consume the
same behavior and artifacts, but training infrastructure must not own or silently replace the
application's Agent semantics.

### Ground reports in evidence

Research output should evolve toward claim-level provenance, explicit uncertainty, source
quality awareness, and reproducible research steps.

### Keep the frontier provisional

The roadmap may include recent techniques, but later releases remain revisable as models,
APIs, and research results change.

## Success Criteria

The project succeeds when:

- each tagged release is complete, understandable, and manually verified;
- the current runtime can be understood by reading three focused Python files;
- a user can watch the complete workflow and read the final report in the terminal;
- the final report is distinguishable from unsupported model recall;
- each major mechanism has a clear learning question and an observable effect;
- the same Worker harness can expose comparable explicitly supported hosted-model runs without
  a provider-neutral platform layer;
- batch rollout generation can eventually reuse the interactive Worker semantics rather than
  maintain a second inference implementation; and
- later complexity can be traced to limitations discovered in earlier releases.

## Non-Goals

The current mission does not include:

- a web application or HTTP service;
- a general-purpose production agent platform;
- a provider-neutral model gateway;
- a plugin architecture or agent framework;
- distributed infrastructure;
- formal automated evaluation during the early learning releases;
- automated tests unless a release specification explicitly introduces them;
- exposing or reconstructing a model's private chain-of-thought.

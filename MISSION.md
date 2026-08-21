# Mission

## Purpose

Build a deep-research command-line agent to learn how modern agents work by implementing
their mechanisms directly, one observable release at a time.

The project favors understanding over product breadth. Its runtime remains in one
cumulative Python file so the complete agent can always be read, run, and changed without
navigating an application architecture.

## Primary Outcome

A contributor should be able to use the repository to understand and experiment with:

- the OpenAI Responses API and response-item model;
- a complete function-tool calling loop;
- agent harness concerns such as limits, errors, state, and stopping;
- live progress and structured trace collection;
- paper and web research tools;
- planning, adaptive retrieval, evidence tracking, and citation;
- context engineering and long-horizon execution;
- critique, verification, and iterative report refinement;
- parallel research and orchestrator-worker multi-agent systems;
- human approval, recovery, memory, and selected frontier techniques.

## Design Principles

### Learn mechanisms by making them visible

Core behavior should be implemented explicitly before an equivalent hosted or framework
feature is adopted. Provider-managed capabilities may later be introduced as comparisons.

### Grow through complete vertical slices

Every release must accept a question, run to completion, expose what happened, and produce
a final Markdown report. A release may be small, but it must not be a disconnected code
fragment.

### Keep one cumulative runtime

All runtime behavior lives in the repository-root `deep_research.py`. New releases modify
that file instead of creating parallel implementations or architectural layers.

### Treat observability as product behavior

The live terminal trajectory and the saved trace are part of the learning experience, not
debug output added after the agent is built.

### Let observed failures justify complexity

Planning, memory, critique, parallelism, and multi-agent coordination should be introduced
only after an earlier release makes the problem they solve observable.

### Ground reports in evidence

Research output should evolve toward claim-level provenance, explicit uncertainty, source
quality awareness, and reproducible research steps.

### Keep the frontier provisional

The roadmap may include recent techniques, but later releases remain revisable as models,
APIs, and research results change.

## Success Criteria

The project succeeds when:

- each tagged release is complete, understandable, and manually verified;
- the current runtime can be understood by reading one Python file;
- a user can watch a run progress and inspect the saved trajectory afterward;
- the final report is distinguishable from unsupported model recall;
- each major mechanism has a clear learning question and an observable effect;
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

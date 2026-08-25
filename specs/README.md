# Release Specifications

Each release specification defines one complete, cumulative learning slice. A spec is
written before implementation and remains the acceptance reference for its release.

## Releases

- [v0.1.0 — One Responses API Call](v0.1.0-single-response.md) — Released
- [v0.2.0 — Fixed Write-Critic-Revise Workflow](v0.2.0-report-refinement.md) — Released
- [v0.3.0 — Tool-Using Research Agent](v0.3.0-tool-research-agent.md) — Released
- [v0.4.0 — Static Structured Research Planning](v0.4.0-static-research-planning.md) — Released
- [v0.5.0 — Scoping and ResearchBrief](v0.5.0-scoping-research-brief.md) — Released
- [v0.6.0 — Sequential Research Supervisor and Workers](v0.6.0-sequential-research-supervisor-workers.md) — Released
- [v0.7.0 — Explicit Agent Run Contract](v0.7.0-explicit-agent-run-contract.md) — Released
- [v0.8.0 — Deep Retrieval Worker](v0.8.0-deep-retrieval-worker.md) — Released
- [v0.9.0 — Local Open-Weight Worker](v0.9.0-local-open-weight-worker.md) — Released
- [v0.10.0 — Long-Horizon Context Sessions](v0.10.0-long-horizon-context-sessions.md) — Verified
- [v0.11.0 — Persistent Evidence Ledger](v0.11.0-persistent-evidence-ledger.md) — Draft
- [v0.12.0 — Structured Claims and Citation Verification](v0.12.0-claim-evidence-citations.md) — Draft
- [v0.13.0 — Durable Run Checkpoints and Resume](v0.13.0-durable-run-checkpoints.md) — Draft
- [v0.14.0 — Idempotent Tools and Failure-Aware Orchestration](v0.14.0-idempotent-failure-aware-runs.md) — Draft
- [v0.15.0 — Hybrid Context Memory](v0.15.0-hybrid-context-memory.md) — Draft
- [v0.16.0 — Structured Traces and Evaluation Harness](v0.16.0-traces-and-evaluations.md) — Draft
- [v0.17.0 — Bounded Parallel Task Graph](v0.17.0-bounded-parallel-task-graph.md) — Draft
- [v0.18.0 — Batch Rollout Runner](v0.18.0-batch-rollout-runner.md) — Draft

## Required Sections

Every specification should include:

1. **Status and version**
2. **Learning question**
3. **Problem and rationale**
4. **User-visible behavior**
5. **Runtime flow**
6. **CLI contract**
7. **Observability contract**
8. **Artifact contract**
9. **Failure and stopping behavior**
10. **Implementation constraints**
11. **Manual acceptance scenario**
12. **Completion criteria**
13. **Out of scope**
14. **State ownership and persistence boundary**
15. **Idempotency and replay contract**
16. **Security and trust boundary**
17. **Evaluation contract**
18. **Known limitations and deferred decisions**

An early release may use a different historical heading or mark a later concern out of scope. New
specifications must nevertheless state each applicable contract explicitly rather than leaving its
behavior implicit.

## Specification Rules

- Describe observable behavior before implementation details.
- Keep the release runnable as a complete system.
- State what the user can observe during and after a run.
- Include at least one repeatable manual acceptance question.
- Identify the limitation that motivates any new abstraction or dependency.
- Research time-sensitive APIs or Agent techniques when the release depends on them, then
  record the decision-relevant rationale and direct primary-source link in that specification.
- Every release specification must cite at least one official API document, standard, original
  research paper, or first-party production engineering report that supports its primary design
  decision. A citation documents provenance; it does not substitute for local acceptance evidence.
- Distinguish model-generated claims from application-observed state and externally verifiable
  evidence.
- Define which state survives context replacement, process failure, and run completion.
- Define whether every external effect is replay-safe and how ambiguous completion is handled.
- Do not claim citation correctness without a deterministic claim-to-evidence identity contract.
- Every compression mechanism must define information-retention and recovery limitations.
- Every concurrency release must define ordering, isolation, cancellation, rate limits, and global
  budgets.
- Releases that introduce persistence, idempotency, citation validation, concurrency, or recovery
  must include deterministic automated tests for those application-owned contracts.
- Do not silently expand the release while implementing it.
- Record unresolved design choices explicitly.
- Update the roadmap if the release scope changes materially.

## Status Values

Use one of:

- `Draft`
- `Accepted`
- `In progress`
- `Verified`
- `Released`

A Git tag may be created only after the specification reaches `Verified`.
After the release tag is created, update the specification status to `Released`.

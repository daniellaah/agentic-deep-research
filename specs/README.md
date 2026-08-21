# Release Specifications

Each release specification defines one complete, cumulative learning slice. A spec is
written before implementation and remains the acceptance reference for its release.

## Required Sections

Every specification should include:

1. **Status and version**
2. **Learning question**
3. **Problem and rationale**
4. **User-visible behavior**
5. **Runtime flow**
6. **CLI contract**
7. **Trace contract**
8. **Artifact contract**
9. **Failure and stopping behavior**
10. **Implementation constraints**
11. **Manual acceptance scenario**
12. **Completion criteria**
13. **Out of scope**

## Specification Rules

- Describe observable behavior before implementation details.
- Keep the release runnable as a complete system.
- State what changes in the trace and final report.
- Include at least one repeatable manual acceptance question.
- Identify the limitation that motivates any new abstraction or dependency.
- Research time-sensitive APIs or Agent techniques when the release depends on them, then
  record the decision-relevant rationale in the specification.
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

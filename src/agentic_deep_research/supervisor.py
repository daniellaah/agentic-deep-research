"""Deterministic control loop for a bounded adaptive research run."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

from .context import ContextBuilder
from .evidence import EvidenceStore
from .models import (
    AgentRun,
    Evidence,
    EvidenceConflict,
    EvidenceLedger,
    ResearchArtifact,
    ResearchBudget,
    ResearchPlan,
    ResearchQuestion,
    ResearchRequest,
    ResearchStep,
    Source,
    TokenUsage,
)
from .planning import ResearchPlanner
from .runner import AgentRunner
from .worker import IndependentResearchWorker, ResearchWorker


@dataclass(frozen=True)
class SupervisorResult:
    """Internal execution state returned to the public workflow."""

    plan: ResearchPlan
    runs: tuple[tuple[ResearchQuestion, AgentRun], ...]
    trace: tuple[ResearchStep, ...]
    status: str
    stop_reason: str
    planning_usage: TokenUsage
    ledger: EvidenceLedger
    artifacts: tuple[ResearchArtifact, ...]

    @property
    def sources(self) -> tuple[Source, ...]:
        """Compatibility view over the ledger's canonical sources."""
        return self.ledger.sources

    @property
    def evidence(self) -> tuple[Evidence, ...]:
        """Compatibility view over the ledger's evidence entries."""
        return self.ledger.evidence

    @property
    def conflicts(self) -> tuple[EvidenceConflict, ...]:
        """Compatibility view over the ledger's recorded conflicts."""
        return self.ledger.conflicts


class ResearchSupervisor:
    """Own planning, global budgets, stopping, and adaptive replanning."""

    def __init__(self, *, planner: ResearchPlanner, runner: AgentRunner) -> None:
        self._planner = planner
        self._worker: ResearchWorker = IndependentResearchWorker(runner)

    def run(self, request: ResearchRequest) -> SupervisorResult:
        """Execute planned questions without exceeding global harness limits."""
        planning_runs = [
            self._planner.plan(
                request=request,
                context="",
                completed_questions=(),
                max_questions=max(1, request.budget.max_research_steps - 1),
                revision=0,
            )
        ]
        initial_plan = planning_runs[0].plan
        all_questions = list(initial_plan.questions)
        pending = _new_questions((), initial_plan.questions)
        runs: list[tuple[ResearchQuestion, AgentRun]] = []
        evidence_store = EvidenceStore()
        artifacts: list[ResearchArtifact] = []
        context_builder = ContextBuilder(request.budget.max_context_chars)
        trace = [
            ResearchStep(
                action="plan_created",
                detail=(
                    f"revision={initial_plan.revision}, questions={len(initial_plan.questions)}"
                ),
                kind="control",
            )
        ]
        remaining_tool_calls = request.budget.max_tool_calls
        revision = initial_plan.revision
        current_round_start = 0
        stop_reason = "completed"

        while pending:
            if len(runs) >= request.budget.max_research_steps:
                stop_reason = "max_research_steps"
                break
            if remaining_tool_calls < 1:
                stop_reason = "max_tool_calls"
                break

            remaining_steps = request.budget.max_research_steps - len(runs)
            outstanding_questions = len(pending)
            batch_size = min(
                request.budget.max_parallel_workers,
                len(pending),
                remaining_steps,
                remaining_tool_calls,
            )
            batch = pending[:batch_size]
            del pending[:batch_size]
            reserve_for_replan = int(remaining_steps > outstanding_questions)
            allocation_slots = outstanding_questions + reserve_for_replan
            worker_tool_limit = max(1, remaining_tool_calls // allocation_slots)
            worker_budget = replace(
                request.budget,
                max_tool_calls=worker_tool_limit,
            )
            contexts = [
                context_builder.build(
                    objective=planning_runs[-1].plan.objective,
                    active_question=question.question,
                    evidence=evidence_store.evidence,
                    conflicts=evidence_store.conflicts,
                )
                for question in batch
            ]
            trace.append(
                ResearchStep(
                    action="worker_batch_started",
                    detail=f"workers={len(batch)}",
                    kind="control",
                )
            )
            for question, context in zip(batch, contexts, strict=True):
                trace.append(
                    ResearchStep(
                        action="context_built",
                        detail=f"{question.id}: chars={len(context)}",
                        kind="control",
                    )
                )
                trace.append(
                    ResearchStep(
                        action="worker_started",
                        detail=(f"{question.id}: tool_budget={worker_budget.max_tool_calls}"),
                        kind="control",
                    )
                )
            batch_runs = self._run_batch(
                request=request,
                questions=batch,
                contexts=contexts,
                budget=worker_budget,
            )
            runs.extend(zip(batch, batch_runs, strict=True))
            used_tool_calls = 0
            for question, run in zip(batch, batch_runs, strict=True):
                artifact = ResearchArtifact(
                    id=f"finding:{question.id}",
                    kind="worker_finding",
                    content=run.report,
                    question_id=question.id,
                )
                evidence_store.ingest_run(
                    question.id,
                    run,
                    artifact_id=artifact.id,
                )
                artifacts.append(artifact)
                trace.extend(run.trace)
                used_tool_calls += sum(step.kind == "tool" for step in run.trace)
                trace.append(
                    ResearchStep(
                        action="worker_completed",
                        detail=f"{question.id}: {run.status}",
                        kind="control",
                    )
                )
            remaining_tool_calls = max(0, remaining_tool_calls - used_tool_calls)

            if pending:
                continue

            gaps = [
                planned_question
                for planned_question, planned_run in runs[current_round_start:]
                if planned_run.status != "completed"
                or not evidence_store.covers(planned_question.id)
            ]
            if not gaps:
                break
            if len(runs) >= request.budget.max_research_steps:
                stop_reason = "max_research_steps"
                break
            if remaining_tool_calls < 1:
                stop_reason = "max_tool_calls"
                break

            revision += 1
            remaining_steps = request.budget.max_research_steps - len(runs)
            planning_run = self._planner.plan(
                request=request,
                context=context_builder.build(
                    objective=planning_runs[-1].plan.objective,
                    active_question="Identify the highest-value unresolved evidence gaps.",
                    evidence=evidence_store.evidence,
                    conflicts=evidence_store.conflicts,
                ),
                completed_questions=tuple(
                    planned_question.question
                    for planned_question, _ in runs
                    if evidence_store.covers(planned_question.id)
                ),
                max_questions=remaining_steps,
                revision=revision,
            )
            planning_runs.append(planning_run)
            revised_questions = _new_questions(all_questions, planning_run.plan.questions)
            if not revised_questions:
                stop_reason = "no_new_questions"
                break
            all_questions.extend(revised_questions)
            pending.extend(revised_questions)
            current_round_start = len(runs)
            trace.append(
                ResearchStep(
                    action="plan_revised",
                    detail=f"revision={revision}, questions={len(revised_questions)}",
                    kind="control",
                )
            )

        if stop_reason != "completed":
            trace.append(
                ResearchStep(
                    action=(
                        "budget_exhausted"
                        if stop_reason in {"max_tool_calls", "max_research_steps"}
                        else "supervisor_stopped"
                    ),
                    detail=stop_reason,
                    kind="control",
                )
            )

        final_plan = ResearchPlan(
            objective=planning_runs[-1].plan.objective,
            questions=tuple(all_questions),
            revision=revision,
        )
        return SupervisorResult(
            plan=final_plan,
            runs=tuple(runs),
            trace=tuple(trace),
            status=(
                "incomplete"
                if stop_reason in {"max_tool_calls", "max_research_steps"}
                else "completed"
            ),
            stop_reason=stop_reason,
            planning_usage=_sum_usage(*(item.usage for item in planning_runs)),
            ledger=evidence_store.snapshot(),
            artifacts=tuple(artifacts),
        )

    def _run_batch(
        self,
        *,
        request: ResearchRequest,
        questions: list[ResearchQuestion],
        contexts: list[str],
        budget: ResearchBudget,
    ) -> list[AgentRun]:
        if len(questions) == 1:
            return [
                self._worker.run(
                    request=request,
                    question=questions[0],
                    context=contexts[0],
                    budget=budget,
                )
            ]
        with ThreadPoolExecutor(max_workers=len(questions)) as executor:
            futures = [
                executor.submit(
                    self._worker.run,
                    request=request,
                    question=question,
                    context=context,
                    budget=budget,
                )
                for question, context in zip(questions, contexts, strict=True)
            ]
            return [future.result() for future in futures]


def _new_questions(
    existing: tuple[ResearchQuestion, ...] | list[ResearchQuestion],
    candidates: tuple[ResearchQuestion, ...],
) -> list[ResearchQuestion]:
    seen = {item.question.strip().casefold() for item in existing}
    result: list[ResearchQuestion] = []
    for candidate in sorted(candidates, key=lambda item: item.priority):
        key = candidate.question.strip().casefold()
        if key and key not in seen:
            result.append(candidate)
            seen.add(key)
    return result


def _sum_usage(*items: TokenUsage) -> TokenUsage:
    return TokenUsage(
        input_tokens=sum(item.input_tokens for item in items),
        output_tokens=sum(item.output_tokens for item in items),
        total_tokens=sum(item.total_tokens for item in items),
    )

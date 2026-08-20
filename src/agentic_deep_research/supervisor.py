"""Deterministic control loop for a bounded adaptive research run."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

from .context import ContextBuilder
from .evidence import EvidenceStore
from .models import (
    AgentRun,
    BudgetSnapshot,
    ContextPack,
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
from .policy import ResearchDecision, SufficiencyPolicy
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
        self._policy = SufficiencyPolicy()

    def run(
        self,
        request: ResearchRequest,
        *,
        initial_ledger: EvidenceLedger | None = None,
        initial_completed_question_ids: frozenset[str] | None = None,
        initial_max_questions: int | None = None,
    ) -> SupervisorResult:
        """Execute planned questions without exceeding global harness limits."""
        if initial_max_questions is not None and initial_max_questions < 1:
            raise ValueError("initial_max_questions must be at least 1")
        evidence_store = EvidenceStore(initial_ledger)
        context_builder = ContextBuilder(request.budget.max_context_chars)
        initial_budget = _budget_snapshot(
            request.budget,
            runs=0,
            tool_calls=request.budget.max_tool_calls,
        )
        initial_context = context_builder.build(
            purpose="planner",
            ledger=evidence_store.snapshot(),
            budget=initial_budget,
        )
        planning_runs = [
            self._planner.plan(
                request=request,
                context=initial_context.text,
                completed_questions=(),
                max_questions=min(
                    request.budget.max_research_steps,
                    initial_max_questions
                    if initial_max_questions is not None
                    else max(1, request.budget.max_research_steps - 1),
                ),
                revision=0,
            )
        ]
        initial_plan = planning_runs[0].plan
        pending = _new_questions((), initial_plan.questions)
        all_questions = list(pending)
        round_questions = tuple(pending)
        runs: list[tuple[ResearchQuestion, AgentRun]] = []
        artifacts: list[ResearchArtifact] = []
        trace = [
            _context_step("planner:initial", initial_context, initial_budget),
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
        round_start_ledger = evidence_store.snapshot()
        seeded_completed_ids = (
            initial_completed_question_ids
            if initial_completed_question_ids is not None
            else frozenset()
        )
        stop_reason = "completed"

        if not pending:
            decision = self._decide(
                questions=(),
                runs=runs,
                seeded_completed_ids=seeded_completed_ids,
                ledger=evidence_store.snapshot(),
                round_start_ledger=round_start_ledger,
                min_sources=request.min_sources,
                budget=initial_budget,
                revision=revision,
                proposed_questions=(),
            )
            trace.append(
                _decision_step(
                    decision,
                    phase="replan",
                    revision=revision,
                    budget=initial_budget,
                )
            )
            if decision.reason != "sufficient":
                stop_reason = decision.reason

        while pending:
            if (
                len(runs) >= request.budget.max_research_steps
                or remaining_tool_calls < 1
            ):
                exhausted_budget = _budget_snapshot(
                    request.budget,
                    runs=len(runs),
                    tool_calls=remaining_tool_calls,
                )
                decision = self._decide(
                    questions=round_questions,
                    runs=runs,
                    seeded_completed_ids=seeded_completed_ids,
                    ledger=evidence_store.snapshot(),
                    round_start_ledger=round_start_ledger,
                    min_sources=request.min_sources,
                    budget=exhausted_budget,
                    revision=revision,
                )
                trace.append(
                    _decision_step(
                        decision,
                        phase="budget",
                        revision=revision,
                        budget=exhausted_budget,
                    )
                )
                stop_reason = decision.reason
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
            batch_ledger = evidence_store.snapshot()
            batch_budget = _budget_snapshot(
                request.budget,
                runs=len(runs),
                tool_calls=remaining_tool_calls,
            )
            context_packs = [
                context_builder.build(
                    purpose="worker",
                    ledger=batch_ledger,
                    budget=batch_budget,
                    active_question_id=question.id,
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
            for question, context_pack in zip(batch, context_packs, strict=True):
                trace.append(_context_step(question.id, context_pack, batch_budget))
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
                contexts=[pack.text for pack in context_packs],
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

            decision_budget = _budget_snapshot(
                request.budget,
                runs=len(runs),
                tool_calls=remaining_tool_calls,
            )
            decision = self._decide(
                questions=round_questions,
                runs=runs,
                seeded_completed_ids=seeded_completed_ids,
                ledger=evidence_store.snapshot(),
                round_start_ledger=round_start_ledger,
                min_sources=request.min_sources,
                budget=decision_budget,
                revision=revision,
            )
            trace.append(
                _decision_step(
                    decision,
                    phase="round",
                    revision=revision,
                    budget=decision_budget,
                )
            )
            if decision.reason == "sufficient":
                break
            if decision.reason != "continue":
                stop_reason = decision.reason
                break

            next_revision = revision + 1
            remaining_steps = request.budget.max_research_steps - len(runs)
            planning_context = context_builder.build(
                purpose="planner",
                ledger=evidence_store.snapshot(),
                budget=decision_budget,
            )
            trace.append(
                _context_step(
                    f"planner:revision:{next_revision}",
                    planning_context,
                    decision_budget,
                )
            )
            planning_run = self._planner.plan(
                request=request,
                context=planning_context.text,
                completed_questions=tuple(
                    planned_question.question
                    for planned_question, _ in runs
                    if planned_question.id in _completed_question_ids(
                        runs,
                        seeded_completed_ids,
                    )
                    and evidence_store.covers(planned_question.id)
                ),
                max_questions=remaining_steps,
                revision=next_revision,
            )
            planning_runs.append(planning_run)
            revised_questions = _new_questions(all_questions, planning_run.plan.questions)
            proposed_decision = self._decide(
                questions=round_questions,
                runs=runs,
                seeded_completed_ids=seeded_completed_ids,
                ledger=evidence_store.snapshot(),
                round_start_ledger=round_start_ledger,
                min_sources=request.min_sources,
                budget=decision_budget,
                revision=revision,
                proposed_questions=tuple(revised_questions),
            )
            trace.append(
                _decision_step(
                    proposed_decision,
                    phase="replan",
                    revision=revision,
                    budget=decision_budget,
                )
            )
            if proposed_decision.reason != "continue":
                stop_reason = proposed_decision.reason
                break
            revision = next_revision
            all_questions.extend(revised_questions)
            pending.extend(revised_questions)
            round_questions = tuple(revised_questions)
            round_start_ledger = evidence_store.snapshot()
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
                if stop_reason
                in {
                    "max_tool_calls",
                    "max_research_steps",
                    "no_new_evidence",
                    "no_new_questions",
                }
                else "completed"
            ),
            stop_reason=stop_reason,
            planning_usage=_sum_usage(*(item.usage for item in planning_runs)),
            ledger=evidence_store.snapshot(),
            artifacts=tuple(artifacts),
        )

    def _decide(
        self,
        *,
        questions: tuple[ResearchQuestion, ...],
        runs: list[tuple[ResearchQuestion, AgentRun]],
        seeded_completed_ids: frozenset[str],
        ledger: EvidenceLedger,
        round_start_ledger: EvidenceLedger,
        min_sources: int,
        budget: BudgetSnapshot,
        revision: int,
        proposed_questions: tuple[ResearchQuestion, ...] | None = None,
    ) -> ResearchDecision:
        return self._policy.decide(
            questions=questions,
            completed_question_ids=_completed_question_ids(
                runs,
                seeded_completed_ids,
            ),
            ledger=ledger,
            round_start_ledger=round_start_ledger,
            min_sources=min_sources,
            budget=budget,
            revision=revision,
            proposed_questions=proposed_questions,
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


def _budget_snapshot(
    budget: ResearchBudget,
    *,
    runs: int,
    tool_calls: int,
) -> BudgetSnapshot:
    return BudgetSnapshot(
        tool_calls_remaining=max(0, tool_calls),
        research_steps_remaining=max(0, budget.max_research_steps - runs),
    )


def _completed_question_ids(
    runs: list[tuple[ResearchQuestion, AgentRun]],
    seeded_completed_ids: frozenset[str],
) -> frozenset[str]:
    return seeded_completed_ids.union(
        question.id
        for question, run in runs
        if run.status == "completed"
    )


def _context_step(
    target: str,
    pack: ContextPack,
    budget: BudgetSnapshot,
) -> ResearchStep:
    selected = "|".join(pack.selected_evidence_ids) or "-"
    return ResearchStep(
        action="context_built",
        detail=(
            f"target={target}, purpose={pack.purpose}, used_chars={pack.used_chars}, "
            f"selected_evidence_ids={selected}, "
            f"omitted_evidence_count={pack.omitted_evidence_count}, "
            f"included_conflict_count={pack.included_conflict_count}, "
            f"omitted_conflict_count={pack.omitted_conflict_count}, "
            f"tool_calls_remaining={budget.tool_calls_remaining}, "
            f"research_steps_remaining={budget.research_steps_remaining}"
        ),
        kind="control",
    )


def _decision_step(
    decision: ResearchDecision,
    *,
    phase: str,
    revision: int,
    budget: BudgetSnapshot,
) -> ResearchStep:
    uncovered = "|".join(decision.uncovered_question_ids) or "-"
    return ResearchStep(
        action="supervisor_decision",
        detail=(
            f"phase={phase}, revision={revision}, reason={decision.reason}, "
            f"uncovered_question_ids={uncovered}, "
            f"additional_sources_needed={decision.additional_sources_needed}, "
            f"new_evidence_count={decision.new_evidence_count}, "
            f"tool_calls_remaining={budget.tool_calls_remaining}, "
            f"research_steps_remaining={budget.research_steps_remaining}"
        ),
        kind="control",
    )


def _sum_usage(*items: TokenUsage) -> TokenUsage:
    return TokenUsage(
        input_tokens=sum(item.input_tokens for item in items),
        output_tokens=sum(item.output_tokens for item in items),
        total_tokens=sum(item.total_tokens for item in items),
    )

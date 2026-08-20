import threading
import time

from agentic_deep_research import (
    AgentRun,
    Citation,
    EvidenceConflict,
    PlanningRun,
    ResearchBudget,
    ResearchPlan,
    ResearchQuestion,
    ResearchRequest,
    ResearchStep,
    Source,
    TokenUsage,
    run_research,
)
from agentic_deep_research.supervisor import ResearchSupervisor


class FakePlanner:
    def __init__(self, runs: list[PlanningRun]) -> None:
        self._runs = iter(runs)
        self.calls: list[dict[str, object]] = []

    def plan(self, **kwargs: object) -> PlanningRun:
        self.calls.append(kwargs)
        return next(self._runs)


class FakeRunner:
    def __init__(self, runs: dict[str, AgentRun]) -> None:
        self._runs = runs
        self.tasks: list[str] = []
        self._tasks_lock = threading.Lock()

    def run(
        self,
        *,
        instructions: str,
        task: str,
        budget: ResearchBudget,
    ) -> AgentRun:
        del instructions, budget
        with self._tasks_lock:
            self.tasks.append(task)
        for question, run in self._runs.items():
            if f"Research question:\n{question}" in task:
                return run
        raise AssertionError(f"unexpected worker task: {task}")


class ConcurrencyRunner(FakeRunner):
    def __init__(self, runs: dict[str, AgentRun]) -> None:
        super().__init__(runs)
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def run(
        self,
        *,
        instructions: str,
        task: str,
        budget: ResearchBudget,
    ) -> AgentRun:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.02)
            return super().run(
                instructions=instructions,
                task=task,
                budget=budget,
            )
        finally:
            with self._lock:
                self.active -= 1


def _cited_run(question: str, claim: str, suffix: str) -> AgentRun:
    source = Source(
        title=f"Source {suffix}",
        url=f"https://example.com/{suffix}",
    )
    raw_report = f"{claim} [1]"
    start = raw_report.index("[1]")
    return AgentRun(
        report=raw_report,
        sources=(source,),
        citations=(
            Citation(
                source=source,
                start_index=start,
                end_index=start + 3,
            ),
        ),
        usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
    )


def test_run_research_executes_an_explicit_adaptive_plan() -> None:
    questions = (
        ResearchQuestion(
            id="q1",
            question="Which evaluation methods improve reliability?",
            rationale="Establish measurable quality controls.",
            priority=1,
        ),
        ResearchQuestion(
            id="q2",
            question="Which observability practices detect failures?",
            rationale="Cover operational reliability.",
            priority=2,
        ),
    )
    plan = ResearchPlan(
        objective="Explain how to build reliable research agents.",
        questions=questions,
    )
    planner = FakePlanner([PlanningRun(plan=plan, usage=TokenUsage(total_tokens=20))])
    runner = FakeRunner(
        {
            questions[0].question: _cited_run(
                questions[0].question,
                "Evaluation measures answer quality.",
                "evaluation",
            ),
            questions[1].question: _cited_run(
                questions[1].question,
                "Tracing exposes failed tool calls.",
                "tracing",
            ),
        }
    )
    request = ResearchRequest(
        topic="Reliable research agents",
        min_sources=2,
        budget=ResearchBudget(
            max_tool_calls=4,
            max_output_tokens=2_000,
            max_research_steps=2,
            max_parallel_workers=1,
            max_context_chars=500,
        ),
    )

    result = run_research(request, runner=runner, planner=planner)

    assert result.plan == plan
    assert [finding.question_id for finding in result.findings] == ["q1", "q2"]
    assert [evidence.question_id for evidence in result.evidence] == ["q1", "q2"]
    assert len(result.sources) == 2
    assert result.ledger is not None
    assert result.sources == result.ledger.sources
    assert result.evidence == result.ledger.evidence
    assert result.conflicts == result.ledger.conflicts
    assert result.citation_checks == result.ledger.checks
    assert result.status == "completed"
    assert result.stop_reason == "completed"
    assert result.usage.total_tokens == 50
    assert any(step.action == "plan_created" for step in result.trace)


def test_supervisor_returns_one_ledger_snapshot_with_compatibility_views() -> None:
    question = ResearchQuestion(id="q1", question="Which evidence is reliable?")
    planner = FakePlanner(
        [PlanningRun(plan=ResearchPlan("Collect reliable evidence", (question,)))]
    )
    runner = FakeRunner(
        {
            question.question: _cited_run(
                question.question,
                "An evaluation supports the result.",
                "ledger",
            )
        }
    )

    supervision = ResearchSupervisor(planner=planner, runner=runner).run(
        ResearchRequest(topic="Reliable evidence", min_sources=1)
    )

    assert supervision.sources == supervision.ledger.sources
    assert supervision.evidence == supervision.ledger.evidence
    assert supervision.conflicts == supervision.ledger.conflicts
    assert supervision.artifacts[0].id == "finding:q1"


def test_supervisor_marks_an_empty_initial_plan_incomplete() -> None:
    planner = FakePlanner([PlanningRun(plan=ResearchPlan("No plan", ()))])
    runner = FakeRunner({})

    result = run_research(
        ResearchRequest(topic="Unplanned topic", min_sources=1),
        runner=runner,
        planner=planner,
    )

    assert runner.tasks == []
    assert result.status == "incomplete"
    assert result.stop_reason == "no_new_questions"
    assert any(
        step.action == "supervisor_decision"
        and "phase=replan" in step.detail
        and "reason=no_new_questions" in step.detail
        for step in result.trace
    )


def test_supervisor_replans_when_a_question_produces_no_evidence() -> None:
    initial_question = ResearchQuestion(
        id="r1q1",
        question="Which claims need verification?",
    )
    revised_question = ResearchQuestion(
        id="r2q1",
        question="Which primary source verifies the reliability claim?",
    )
    planner = FakePlanner(
        [
            PlanningRun(plan=ResearchPlan("Verify reliability", (initial_question,), revision=0)),
            PlanningRun(plan=ResearchPlan("Verify reliability", (revised_question,), revision=1)),
        ]
    )
    runner = FakeRunner(
        {
            initial_question.question: AgentRun(report="No verifiable source found."),
            revised_question.question: _cited_run(
                revised_question.question,
                "A primary evaluation documents the reliability result.",
                "primary-evaluation",
            ),
        }
    )
    request = ResearchRequest(
        topic="Reliable agents",
        min_sources=1,
        budget=ResearchBudget(
            max_tool_calls=2,
            max_output_tokens=2_000,
            max_research_steps=2,
            max_parallel_workers=1,
        ),
    )

    result = run_research(request, runner=runner, planner=planner)

    assert result.plan is not None
    assert [question.id for question in result.plan.questions] == ["r1q1", "r2q1"]
    assert result.plan.revision == 1
    assert len(result.findings) == 2
    assert result.evidence[0].question_id == "r2q1"
    assert result.status == "completed"
    assert any(step.action == "plan_revised" for step in result.trace)
    assert len(planner.calls) == 2


def test_supervisor_replans_until_global_source_minimum_is_met() -> None:
    initial_question = ResearchQuestion(id="r1q1", question="Establish the claim")
    revised_question = ResearchQuestion(id="r2q1", question="Corroborate the claim")
    planner = FakePlanner(
        [
            PlanningRun(plan=ResearchPlan("Corroborate", (initial_question,))),
            PlanningRun(
                plan=ResearchPlan("Corroborate", (revised_question,), revision=1)
            ),
        ]
    )
    runner = FakeRunner(
        {
            initial_question.question: _cited_run(
                initial_question.question,
                "One evaluation supports the claim.",
                "first-source",
            ),
            revised_question.question: _cited_run(
                revised_question.question,
                "An independent evaluation corroborates the claim.",
                "second-source",
            ),
        }
    )

    result = run_research(
        ResearchRequest(
            topic="Corroborated claim",
            min_sources=2,
            budget=ResearchBudget(
                max_tool_calls=2,
                max_research_steps=2,
                max_parallel_workers=1,
            ),
        ),
        runner=runner,
        planner=planner,
    )

    decisions = [
        step.detail for step in result.trace if step.action == "supervisor_decision"
    ]
    assert len(result.sources) == 2
    assert result.status == "completed"
    assert any("reason=continue" in detail for detail in decisions)
    assert "reason=sufficient" in decisions[-1]


def test_supervisor_marks_zero_yield_revised_round_incomplete() -> None:
    initial_question = ResearchQuestion(id="r1q1", question="Search the first route")
    revised_question = ResearchQuestion(id="r2q1", question="Search another route")
    planner = FakePlanner(
        [
            PlanningRun(plan=ResearchPlan("Find evidence", (initial_question,))),
            PlanningRun(
                plan=ResearchPlan("Find evidence", (revised_question,), revision=1)
            ),
        ]
    )
    runner = FakeRunner(
        {
            initial_question.question: AgentRun(report="No evidence found."),
            revised_question.question: AgentRun(report="Still no evidence found."),
        }
    )

    result = run_research(
        ResearchRequest(
            topic="Hard-to-find evidence",
            min_sources=1,
            budget=ResearchBudget(
                max_tool_calls=3,
                max_research_steps=3,
                max_parallel_workers=1,
            ),
        ),
        runner=runner,
        planner=planner,
    )

    assert result.status == "incomplete"
    assert result.stop_reason == "no_new_evidence"
    assert any(
        step.action == "supervisor_decision" and "reason=no_new_evidence" in step.detail
        for step in result.trace
    )


def test_supervisor_marks_duplicate_replan_incomplete() -> None:
    question = ResearchQuestion(id="r1q1", question="Find one source")
    duplicate = ResearchQuestion(id="r2q1", question="  find ONE source  ")
    planner = FakePlanner(
        [
            PlanningRun(plan=ResearchPlan("Find sources", (question,))),
            PlanningRun(plan=ResearchPlan("Find sources", (duplicate,), revision=1)),
        ]
    )
    runner = FakeRunner(
        {
            question.question: _cited_run(
                question.question,
                "One source supports the result.",
                "only-source",
            )
        }
    )

    result = run_research(
        ResearchRequest(
            topic="Find multiple sources",
            min_sources=2,
            budget=ResearchBudget(
                max_tool_calls=2,
                max_research_steps=2,
                max_parallel_workers=1,
            ),
        ),
        runner=runner,
        planner=planner,
    )

    assert result.status == "incomplete"
    assert result.stop_reason == "no_new_questions"
    assert any(
        step.action == "supervisor_decision" and "reason=no_new_questions" in step.detail
        for step in result.trace
    )


def test_supervisor_stops_before_exceeding_the_global_tool_budget() -> None:
    questions = (
        ResearchQuestion(id="q1", question="First question", priority=1),
        ResearchQuestion(id="q2", question="Second question", priority=2),
    )
    planner = FakePlanner([PlanningRun(plan=ResearchPlan("Bounded research", questions))])
    first_run = _cited_run("First question", "One supported finding.", "one")
    first_run = AgentRun(
        report=first_run.report,
        sources=first_run.sources,
        citations=first_run.citations,
        trace=(ResearchStep(action="search", detail="first query"),),
    )
    runner = FakeRunner(
        {
            "First question": first_run,
            "Second question": _cited_run("Second question", "A result that must not run.", "two"),
        }
    )
    request = ResearchRequest(
        topic="A bounded topic",
        min_sources=1,
        budget=ResearchBudget(
            max_tool_calls=1,
            max_output_tokens=2_000,
            max_research_steps=2,
            max_parallel_workers=1,
        ),
    )

    result = run_research(request, runner=runner, planner=planner)

    assert [finding.question_id for finding in result.findings] == ["q1"]
    assert result.status == "incomplete"
    assert result.stop_reason == "max_tool_calls"
    assert any(step.action == "budget_exhausted" for step in result.trace)


def test_supervisor_runs_independent_workers_with_bounded_parallelism() -> None:
    questions = tuple(
        ResearchQuestion(id=f"q{index}", question=f"Question {index}", priority=index)
        for index in range(1, 4)
    )
    planner = FakePlanner([PlanningRun(plan=ResearchPlan("Parallel research", questions))])
    runner = ConcurrencyRunner(
        {
            question.question: _cited_run(
                question.question,
                f"Supported finding {question.id}.",
                question.id,
            )
            for question in questions
        }
    )
    request = ResearchRequest(
        topic="Parallel research",
        min_sources=3,
        budget=ResearchBudget(
            max_tool_calls=3,
            max_output_tokens=2_000,
            max_research_steps=3,
            max_parallel_workers=2,
        ),
    )

    result = run_research(request, runner=runner, planner=planner)

    assert runner.max_active == 2
    assert [finding.question_id for finding in result.findings] == ["q1", "q2", "q3"]
    batches = [step.detail for step in result.trace if step.action == "worker_batch_started"]
    assert batches == ["workers=2", "workers=1"]
    first_batch_contexts = [
        step.detail
        for step in result.trace
        if step.action == "context_built"
        and ("target=q1," in step.detail or "target=q2," in step.detail)
    ]
    assert len(first_batch_contexts) == 2
    assert all("tool_calls_remaining=3" in detail for detail in first_batch_contexts)
    assert all("research_steps_remaining=3" in detail for detail in first_batch_contexts)
    assert result.status == "completed"


def test_evidence_store_deduplicates_and_tracks_quality_confidence_and_conflicts() -> None:
    questions = (
        ResearchQuestion(id="q1", question="What does the evaluation show?"),
        ResearchQuestion(id="q2", question="Can another source corroborate it?"),
    )
    planner = FakePlanner([PlanningRun(plan=ResearchPlan("Corroborate the result", questions))])
    agency = Source(title="Agency study", url="https://example.gov/study")
    university = Source(title="University study", url="https://example.edu/study")
    claim = "Evaluation improves reliability."
    first_report = f"{claim} [1]"
    first_start = first_report.index("[1]")
    repeated_citation = Citation(
        source=agency,
        start_index=first_start,
        end_index=first_start + 3,
    )
    conflict = EvidenceConflict(
        question_id="q1",
        description="Two sources report different effects for small models.",
        sources=(agency, university),
    )
    first_run = AgentRun(
        report=first_report,
        sources=(agency,),
        citations=(repeated_citation, repeated_citation),
        conflicts=(conflict, conflict),
    )
    second_run = _cited_run(questions[1].question, claim, "unused")
    second_citation = Citation(
        source=university,
        start_index=second_run.citations[0].start_index,
        end_index=second_run.citations[0].end_index,
    )
    second_run = AgentRun(
        report=second_run.report,
        sources=(university,),
        citations=(second_citation,),
    )
    runner = FakeRunner(
        {
            questions[0].question: first_run,
            questions[1].question: second_run,
        }
    )

    result = run_research(
        ResearchRequest(
            topic="Corroborated reliability",
            min_sources=2,
            budget=ResearchBudget(
                max_tool_calls=2,
                max_output_tokens=2_000,
                max_research_steps=2,
                max_parallel_workers=1,
            ),
        ),
        runner=runner,
        planner=planner,
    )

    assert len(result.evidence) == 2
    assert {item.confidence for item in result.evidence} == {"high"}
    assert {item.source.quality for item in result.evidence} == {"unknown"}
    assert [source.quality for source in result.sources] == ["unknown", "unknown"]
    assert all(citation.source in result.sources for citation in result.citations)
    assert result.report.count("https://example.gov/study") == 1
    assert len(result.conflicts) == 1
    assert result.conflicts[0].description.startswith("Two sources")


def test_context_engineering_passes_bounded_evidence_and_preserves_full_artifacts() -> None:
    questions = (
        ResearchQuestion(id="q1", question="Collect the first finding", priority=1),
        ResearchQuestion(id="q2", question="Use the first finding", priority=2),
    )
    planner = FakePlanner(
        [PlanningRun(plan=ResearchPlan("Build evidence progressively", questions))]
    )
    long_claim = "A" * 240 + " verified finding."
    runner = FakeRunner(
        {
            questions[0].question: _cited_run(
                questions[0].question,
                long_claim,
                "first",
            ),
            questions[1].question: _cited_run(
                questions[1].question,
                "The second finding uses prior evidence.",
                "second",
            ),
        }
    )
    request = ResearchRequest(
        topic="Progressive context",
        min_sources=2,
        budget=ResearchBudget(
            max_tool_calls=2,
            max_output_tokens=2_000,
            max_research_steps=2,
            max_parallel_workers=1,
            max_context_chars=180,
        ),
    )

    result = run_research(request, runner=runner, planner=planner)

    second_task = next(task for task in runner.tasks if questions[1].question in task)
    context = second_task.split("Prior evidence:\n", 1)[1].split("\n\nWrite", 1)[0]
    assert len(context) <= 180
    assert '"kind":"context"' in context
    assert '"purpose":"worker"' in context
    assert long_claim not in context
    assert [artifact.question_id for artifact in result.artifacts] == ["q1", "q2"]
    assert result.artifacts[0].content == f"{long_claim} [1]"
    assert any(
        step.action == "context_built"
        and "target=q2," in step.detail
        and "omitted_evidence_count=1" in step.detail
        for step in result.trace
    )

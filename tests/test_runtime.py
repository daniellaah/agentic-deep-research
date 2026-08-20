import json
from threading import Barrier, Event, Thread

import pytest

from agentic_deep_research import (
    AgentRun,
    Citation,
    CitationVerification,
    ClarificationDecision,
    ConversationMessage,
    PlanningRun,
    ReportCritique,
    ReportDraft,
    ResearchBrief,
    ResearchBudget,
    ResearchPlan,
    ResearchQuestion,
    ResearchRequest,
    ScopingRun,
    Source,
    TokenUsage,
)
from agentic_deep_research.runtime import (
    JsonCheckpointStore,
    ResearchRuntime,
    RetryPolicy,
    RunState,
    SQLiteCheckpointStore,
)


class TwoQuestionPlanner:
    def __init__(self) -> None:
        self.call_count = 0

    def plan(self, **kwargs: object) -> PlanningRun:
        self.call_count += 1
        max_questions = int(kwargs["max_questions"])
        questions = (
            ResearchQuestion(id="q1", question="First question", priority=1),
            ResearchQuestion(id="q2", question="Second question", priority=2),
        )
        return PlanningRun(
            plan=ResearchPlan(
                objective="Durable research",
                questions=questions[:max_questions],
            )
        )


class CrashOnceRunner:
    def __init__(self) -> None:
        self.calls = {"First question": 0, "Second question": 0}
        self._crashed = False

    def run(
        self,
        *,
        instructions: str,
        task: str,
        budget: ResearchBudget,
    ) -> AgentRun:
        del instructions, budget
        question = next(item for item in self.calls if f"Research question:\n{item}" in task)
        self.calls[question] += 1
        if question == "Second question" and not self._crashed:
            self._crashed = True
            raise RuntimeError("adapter failed after the first completed effect")
        source = Source(question, f"https://example.com/{question.split()[0].lower()}")
        report = f"{question} answer. [1]"
        marker_start = report.index("[1]")
        return AgentRun(
            report=report,
            sources=(source,),
            citations=(Citation(source, marker_start, marker_start + 3),),
        )


def test_checkpoint_store_round_trips_a_created_run(tmp_path) -> None:
    store = JsonCheckpointStore(tmp_path)
    state = RunState.create(
        ResearchRequest(
            topic="Reliable long-running research",
            language="English",
            budget=ResearchBudget(
                max_tool_calls=6,
                max_output_tokens=12_000,
                max_research_steps=3,
                max_parallel_workers=1,
                max_context_chars=5_000,
                max_verification_tool_calls=1,
                max_revision_rounds=1,
            ),
            min_sources=3,
        ),
        run_id="run-001",
    )

    store.save(state)
    restored = store.load("run-001")

    assert restored == state
    assert restored.request.budget.max_output_tokens == 12_000
    checkpoint_path = tmp_path / "run-001.json"
    assert json.loads(checkpoint_path.read_text())["schema_version"] == 1
    assert sorted(path.name for path in tmp_path.iterdir()) == ["run-001.json"]

    legacy_payload = json.loads(checkpoint_path.read_text())
    legacy_payload.pop("conversation")
    legacy_payload.pop("clarification")
    legacy_payload.pop("plan_controls")
    legacy_payload["request"].pop("brief")
    checkpoint_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

    legacy = store.load("run-001")
    assert legacy.conversation == ()
    assert legacy.clarification is None
    assert legacy.plan_controls == ()
    assert legacy.brief is None


def test_resume_reuses_completed_effects_after_a_process_failure(tmp_path) -> None:
    store = JsonCheckpointStore(tmp_path)
    planner = TwoQuestionPlanner()
    runner = CrashOnceRunner()
    request = ResearchRequest(
        topic="Durable research",
        min_sources=2,
        budget=ResearchBudget(
            max_tool_calls=4,
            max_research_steps=3,
            max_parallel_workers=1,
        ),
    )

    first_runtime = ResearchRuntime(
        store=store,
        planner=planner,
        runner=runner,
    )
    failed = first_runtime.start(request, run_id="run-resume")

    assert failed.state.status == "failed"
    assert runner.calls == {"First question": 1, "Second question": 1}

    restarted_runtime = ResearchRuntime(
        store=store,
        planner=planner,
        runner=runner,
    )
    resumed = restarted_runtime.resume("run-resume")

    assert resumed.state.status == "completed"
    assert resumed.result is not None
    assert resumed.result.status == "completed"
    assert resumed.result.ledger is not None
    assert all(item.id for item in resumed.result.ledger.evidence)
    assert planner.call_count == 1
    assert runner.calls == {"First question": 1, "Second question": 2}


class TransientRunner:
    def __init__(self, error: BaseException) -> None:
        self._error = error
        self.call_count = 0

    def run(
        self,
        *,
        instructions: str,
        task: str,
        budget: ResearchBudget,
    ) -> AgentRun:
        del instructions, task, budget
        self.call_count += 1
        if self.call_count == 1:
            raise self._error
        source = Source("Recovered source", "https://example.com/recovered")
        return AgentRun(
            report="Recovered answer. [1]",
            sources=(source,),
            citations=(Citation(source, 18, 21),),
        )


def test_runtime_retries_a_transient_failure_with_backoff(tmp_path) -> None:
    runner = TransientRunner(ConnectionError("temporary disconnect"))
    delays: list[float] = []
    runtime = ResearchRuntime(
        store=JsonCheckpointStore(tmp_path),
        runner=runner,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_delay_seconds=0.25,
            jitter_ratio=0,
        ),
        sleeper=delays.append,
    )

    outcome = runtime.start(
        ResearchRequest(
            topic="Retry reliable research",
            min_sources=1,
            budget=ResearchBudget(max_tool_calls=1, max_research_steps=1),
        ),
        run_id="run-retry",
    )

    assert outcome.state.status == "completed"
    assert runner.call_count == 2
    assert delays == [0.25]
    assert next(
        effect for effect in outcome.state.effects if effect.kind == "runner.run"
    ).attempts == 2

    restarted = ResearchRuntime(store=JsonCheckpointStore(tmp_path), runner=runner)
    restored = restarted.resume("run-retry")

    assert restored.result == outcome.result
    assert runner.call_count == 2


def test_runtime_does_not_retry_a_programming_error(tmp_path) -> None:
    runner = TransientRunner(ValueError("invalid normalized response"))
    delays: list[float] = []
    runtime = ResearchRuntime(
        store=JsonCheckpointStore(tmp_path),
        runner=runner,
        retry_policy=RetryPolicy(max_attempts=3),
        sleeper=delays.append,
    )

    outcome = runtime.start(
        ResearchRequest(topic="Do not retry invalid data"),
        run_id="run-no-retry",
    )

    assert outcome.state.status == "failed"
    assert runner.call_count == 1
    assert delays == []
    assert next(
        effect for effect in outcome.state.effects if effect.kind == "runner.run"
    ).attempts == 1


class AlwaysFailingRunner:
    def __init__(self) -> None:
        self.call_count = 0

    def run(
        self,
        *,
        instructions: str,
        task: str,
        budget: ResearchBudget,
    ) -> AgentRun:
        del instructions, task, budget
        self.call_count += 1
        raise ConnectionError("provider remains unavailable")


def test_resume_cannot_bypass_the_total_retry_limit(tmp_path) -> None:
    runner = AlwaysFailingRunner()
    store = JsonCheckpointStore(tmp_path)
    runtime = ResearchRuntime(
        store=store,
        runner=runner,
        retry_policy=RetryPolicy(max_attempts=2, initial_delay_seconds=0),
        sleeper=lambda _: None,
    )

    failed = runtime.start(
        ResearchRequest(topic="Bound total retries"),
        run_id="run-retry-limit",
    )
    resumed = runtime.resume("run-retry-limit")

    assert failed.state.status == "failed"
    assert resumed.state.status == "failed"
    assert runner.call_count == 2
    assert next(
        effect for effect in resumed.state.effects if effect.kind == "runner.run"
    ).attempts == 2


def test_resume_rejects_a_different_model_configuration(tmp_path) -> None:
    runner = TransientRunner(ValueError("stop before completion"))
    runner._model = "model-a"
    store = JsonCheckpointStore(tmp_path)
    runtime = ResearchRuntime(store=store, runner=runner)
    runtime.start(
        ResearchRequest(topic="Keep one runtime configuration"),
        run_id="run-config",
    )
    runner._model = "model-b"

    with pytest.raises(ValueError, match="configuration does not match"):
        runtime.resume("run-config")

    assert runner.call_count == 1


class SimulatedProcessCrash(BaseException):
    pass


def test_ambiguous_in_flight_effect_requires_explicit_retry(tmp_path) -> None:
    runner = TransientRunner(SimulatedProcessCrash("process stopped after request"))
    store = JsonCheckpointStore(tmp_path)
    runtime = ResearchRuntime(store=store, runner=runner)
    request = ResearchRequest(
        topic="Do not silently duplicate an ambiguous request",
        min_sources=1,
        budget=ResearchBudget(max_tool_calls=1, max_research_steps=1),
    )

    with pytest.raises(SimulatedProcessCrash):
        runtime.start(request, run_id="run-ambiguous")

    restarted = ResearchRuntime(store=store, runner=runner)
    waiting = restarted.resume("run-ambiguous")

    assert waiting.state.status == "waiting_for_human"
    assert waiting.state.current_step == "ambiguous_effect"
    assert runner.call_count == 1

    completed = restarted.resume("run-ambiguous", retry_ambiguous=True)

    assert completed.state.status == "completed"
    assert runner.call_count == 2


class RecordingRunner:
    def __init__(self, *, entered: Event | None = None, release: Event | None = None) -> None:
        self.calls: list[str] = []
        self._entered = entered
        self._release = release

    def run(
        self,
        *,
        instructions: str,
        task: str,
        budget: ResearchBudget,
    ) -> AgentRun:
        del instructions, budget
        question = next(
            item
            for item in ("First question", "Second question")
            if f"Research question:\n{item}" in task
        )
        self.calls.append(question)
        if len(self.calls) == 1 and self._entered is not None and self._release is not None:
            self._entered.set()
            if not self._release.wait(timeout=2):
                raise TimeoutError("test did not release the runner")
        source = Source(question, f"https://example.com/{question.split()[0].lower()}")
        report = f"{question} answer. [1]"
        marker_start = report.index("[1]")
        return AgentRun(
            report=report,
            sources=(source,),
            citations=(Citation(source, marker_start, marker_start + 3),),
        )


def test_run_waits_for_approval_after_planning(tmp_path) -> None:
    planner = TwoQuestionPlanner()
    runner = RecordingRunner()
    runtime = ResearchRuntime(
        store=JsonCheckpointStore(tmp_path),
        planner=planner,
        runner=runner,
    )
    request = ResearchRequest(
        topic="Review the plan first",
        min_sources=2,
        budget=ResearchBudget(max_research_steps=3, max_parallel_workers=1),
    )

    waiting = runtime.start(request, run_id="run-approval", require_approval=True)

    assert waiting.state.status == "waiting_for_human"
    assert waiting.state.approval_status == "pending"
    assert planner.call_count == 1
    assert runner.calls == []
    assert [effect.kind for effect in waiting.state.effects] == ["planner.plan"]

    approved = runtime.approve("run-approval")

    assert approved.state.status == "completed"
    assert planner.call_count == 1
    assert runner.calls == ["First question", "Second question"]


def test_rejected_plan_never_starts_research(tmp_path) -> None:
    planner = TwoQuestionPlanner()
    runner = RecordingRunner()
    runtime = ResearchRuntime(
        store=JsonCheckpointStore(tmp_path),
        planner=planner,
        runner=runner,
    )
    runtime.start(
        ResearchRequest(topic="Reject this plan"),
        run_id="run-reject",
        require_approval=True,
    )

    rejected = runtime.reject("run-reject", reason="Plan is too broad")
    resumed = runtime.resume("run-reject")

    assert rejected.state.status == "cancelled"
    assert rejected.state.approval_status == "rejected"
    assert rejected.state.termination_reason == "approval_rejected"
    assert resumed.state == rejected.state
    assert runner.calls == []


def test_cancel_during_an_external_call_stops_before_the_next_effect(tmp_path) -> None:
    entered = Event()
    release = Event()
    planner = TwoQuestionPlanner()
    runner = RecordingRunner(entered=entered, release=release)
    database = tmp_path / "runs.sqlite3"
    runtime = ResearchRuntime(
        store=SQLiteCheckpointStore(database),
        planner=planner,
        runner=runner,
        retry_policy=RetryPolicy(max_attempts=1),
    )
    outcomes = []

    thread = Thread(
        target=lambda: outcomes.append(
            runtime.start(
                ResearchRequest(
                    topic="Cooperative cancellation",
                    min_sources=2,
                    budget=ResearchBudget(
                        max_research_steps=3,
                        max_parallel_workers=1,
                    ),
                ),
                run_id="run-cancel",
            )
        )
    )
    thread.start()
    assert entered.wait(timeout=2)

    control_runtime = ResearchRuntime(store=SQLiteCheckpointStore(database))
    requested = control_runtime.request_cancel(
        "run-cancel",
        reason="User stopped the run",
    )
    release.set()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert requested.cancel_requested is True
    assert outcomes[0].state.status == "cancelled"
    assert outcomes[0].state.termination_reason == "User stopped the run"
    assert runner.calls == ["First question"]
    runner_effects = [
        effect for effect in outcomes[0].state.effects if effect.kind == "runner.run"
    ]
    assert len(runner_effects) == 1
    assert runner_effects[0].status == "completed"


def test_execution_lease_prevents_two_runtimes_from_resuming_the_same_run(
    tmp_path,
) -> None:
    entered = Event()
    release = Event()
    database = tmp_path / "leased-runs.sqlite3"
    runner = RecordingRunner(entered=entered, release=release)
    runtime = ResearchRuntime(
        store=SQLiteCheckpointStore(database),
        runner=runner,
    )
    outcomes = []
    thread = Thread(
        target=lambda: outcomes.append(
            runtime.start(
                ResearchRequest(
                    topic="First question",
                    min_sources=1,
                    budget=ResearchBudget(max_tool_calls=1, max_research_steps=1),
                ),
                run_id="run-leased",
            )
        )
    )
    thread.start()
    assert entered.wait(timeout=2)

    competing_runtime = ResearchRuntime(
        store=SQLiteCheckpointStore(database),
        runner=runner,
    )
    with pytest.raises(RuntimeError, match="already executing"):
        competing_runtime.resume("run-leased")

    release.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert outcomes[0].state.status == "completed"
    assert runner.calls == ["First question"]


def test_lease_heartbeat_protects_a_long_external_call(tmp_path) -> None:
    entered = Event()
    release = Event()
    database = tmp_path / "heartbeat-runs.sqlite3"
    runner = RecordingRunner(entered=entered, release=release)
    runtime = ResearchRuntime(
        store=SQLiteCheckpointStore(database),
        runner=runner,
        lease_ttl_seconds=0.12,
    )
    outcomes = []
    thread = Thread(
        target=lambda: outcomes.append(
            runtime.start(
                ResearchRequest(
                    topic="First question",
                    min_sources=1,
                    budget=ResearchBudget(max_tool_calls=1, max_research_steps=1),
                ),
                run_id="run-heartbeat",
            )
        )
    )
    thread.start()
    assert entered.wait(timeout=2)
    assert Event().wait(timeout=0.25) is False

    competing_runtime = ResearchRuntime(
        store=SQLiteCheckpointStore(database),
        runner=runner,
    )
    with pytest.raises(RuntimeError, match="already executing"):
        competing_runtime.resume("run-heartbeat")

    release.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert outcomes[0].state.status == "completed"


class ParallelRunner(RecordingRunner):
    def __init__(self) -> None:
        super().__init__()
        self._barrier = Barrier(2)

    def run(
        self,
        *,
        instructions: str,
        task: str,
        budget: ResearchBudget,
    ) -> AgentRun:
        del instructions, budget
        question = next(
            item
            for item in ("First question", "Second question")
            if f"Research question:\n{item}" in task
        )
        self.calls.append(question)
        self._barrier.wait(timeout=2)
        source = Source(question, f"https://example.com/{question.split()[0].lower()}")
        report = f"{question} answer. [1]"
        marker_start = report.index("[1]")
        return AgentRun(
            report=report,
            sources=(source,),
            citations=(Citation(source, marker_start, marker_start + 3),),
        )


def test_parallel_workers_preserve_each_completed_effect(tmp_path) -> None:
    database = tmp_path / "parallel-runs.sqlite3"
    runner = ParallelRunner()
    runtime = ResearchRuntime(
        store=SQLiteCheckpointStore(database),
        planner=TwoQuestionPlanner(),
        runner=runner,
    )

    outcome = runtime.start(
        ResearchRequest(
            topic="Parallel durable research",
            min_sources=2,
            budget=ResearchBudget(
                max_tool_calls=4,
                max_research_steps=3,
                max_parallel_workers=2,
            ),
        ),
        run_id="run-parallel",
    )

    runner_effects = [
        effect for effect in outcome.state.effects if effect.kind == "runner.run"
    ]
    assert outcome.state.status == "completed"
    assert sorted(runner.calls) == ["First question", "Second question"]
    assert len(runner_effects) == 2
    assert all(effect.status == "completed" for effect in runner_effects)

    restored = runtime.resume("run-parallel")
    assert restored.result == outcome.result
    assert len(runner.calls) == 2


class CancelBeforeStartingStore(SQLiteCheckpointStore):
    def __init__(self, path) -> None:
        super().__init__(path)
        self.owned_updates = 0

    def update_owned(self, run_id, owner_id, ttl_seconds, mutation):
        self.owned_updates += 1
        if self.owned_updates == 2:
            ResearchRuntime(store=self).request_cancel(
                run_id,
                reason="Cancelled before workflow start",
            )
        return super().update_owned(run_id, owner_id, ttl_seconds, mutation)


def test_cancelled_state_never_transitions_back_to_running(tmp_path) -> None:
    runner = RecordingRunner()
    runtime = ResearchRuntime(
        store=CancelBeforeStartingStore(tmp_path / "terminal-state.sqlite3"),
        runner=runner,
    )

    outcome = runtime.start(
        ResearchRequest(topic="First question"),
        run_id="run-terminal",
    )

    assert outcome.state.status == "cancelled"
    assert outcome.state.termination_reason == "Cancelled before workflow start"
    assert runner.calls == []


def _clear_brief() -> ResearchBrief:
    return ResearchBrief(
        research_question="How do reliable research agents work?",
        objective="Help engineers compare reliable research-agent designs.",
        scope_inclusions=("Planning", "Evidence verification"),
        constraints=("Prefer primary sources",),
        deliverable="A cited Markdown report.",
        success_criteria=("Compare the major design patterns",),
    )


class SequencedScoper:
    def __init__(self, outcomes: list[ScopingRun]) -> None:
        self._outcomes = iter(outcomes)
        self._model = "scope-model"
        self.calls: list[tuple[ConversationMessage, ...]] = []

    def scope(self, *, messages: tuple[ConversationMessage, ...]) -> ScopingRun:
        self.calls.append(messages)
        return next(self._outcomes)


class UsagePlanner:
    def __init__(self) -> None:
        self.call_count = 0

    def plan(self, **kwargs: object) -> PlanningRun:
        self.call_count += 1
        return PlanningRun(
            plan=ResearchPlan(
                objective="Generated objective",
                questions=(
                    ResearchQuestion(
                        id="generated-q1",
                        question="Generated question",
                        priority=1,
                    ),
                ),
            ),
            usage=TokenUsage(input_tokens=3, output_tokens=4, total_tokens=7),
        )


class AnyQuestionRunner:
    def __init__(self) -> None:
        self.questions: list[str] = []
        self.tasks: list[str] = []

    def run(
        self,
        *,
        instructions: str,
        task: str,
        budget: ResearchBudget,
    ) -> AgentRun:
        del instructions, budget
        question = task.split("Research question:\n", 1)[1].split("\n", 1)[0]
        self.questions.append(question)
        self.tasks.append(task)
        source = Source(question, "https://example.com/edited-question")
        report = f"Answer for {question}. [1]"
        marker_start = report.index("[1]")
        return AgentRun(
            report=report,
            sources=(source,),
            citations=(Citation(source, marker_start, marker_start + 3),),
        )


class EmptyPlanner:
    def plan(self, **kwargs: object) -> PlanningRun:
        del kwargs
        return PlanningRun(
            plan=ResearchPlan(objective="Empty objective", questions=())
        )


class ObjectiveRecordingPlanner:
    def __init__(self) -> None:
        self.objectives: list[str | None] = []

    def plan(self, **kwargs: object) -> PlanningRun:
        request = kwargs["request"]
        revision = int(kwargs["revision"])
        assert isinstance(request, ResearchRequest)
        self.objectives.append(
            None if request.brief is None else request.brief.objective
        )
        return PlanningRun(
            plan=ResearchPlan(
                objective=f"Generated objective {revision}",
                questions=(
                    ResearchQuestion(
                        id=f"q{revision + 1}",
                        question=f"Research question {revision + 1}",
                    ),
                ),
                revision=revision,
            )
        )


class ObjectiveRecordingRunner:
    def __init__(self) -> None:
        self.tasks: list[str] = []

    def run(
        self,
        *,
        instructions: str,
        task: str,
        budget: ResearchBudget,
    ) -> AgentRun:
        del instructions, budget
        self.tasks.append(task)
        question = task.split("Research question:\n", 1)[1].split("\n", 1)[0]
        source = Source(question, f"https://example.com/source-{len(self.tasks)}")
        report = f"Evidence for {question}. [1]"
        marker_start = report.index("[1]")
        return AgentRun(
            report=report,
            sources=(source,),
            citations=(Citation(source, marker_start, marker_start + 3),),
        )


class ObjectiveRecordingReportAgent:
    def __init__(self) -> None:
        self.objectives: list[str | None] = []
        self.critique_calls = 0

    def _record(self, kwargs: dict[str, object]) -> None:
        request = kwargs["request"]
        assert isinstance(request, ResearchRequest)
        self.objectives.append(
            None if request.brief is None else request.brief.objective
        )

    def write(self, **kwargs: object) -> ReportDraft:
        self._record(kwargs)
        return ReportDraft(report="Initial synthesis.")

    def critique(self, **kwargs: object) -> ReportCritique:
        self._record(kwargs)
        self.critique_calls += 1
        if self.critique_calls == 1:
            return ReportCritique(
                clarity_issues=("Clarify the conclusion.",),
                revision_instructions=("Revise the conclusion.",),
            )
        return ReportCritique()

    def verify(self, **kwargs: object) -> CitationVerification:
        self._record(kwargs)
        return CitationVerification()

    def revise(self, **kwargs: object) -> ReportDraft:
        self._record(kwargs)
        return ReportDraft(report="Revised synthesis.")


def test_create_persists_a_run_without_executing_components(tmp_path) -> None:
    planner = UsagePlanner()
    runner = AnyQuestionRunner()
    scoper = SequencedScoper([])
    runtime = ResearchRuntime(
        store=JsonCheckpointStore(tmp_path),
        runner=runner,
        planner=planner,
        scoper=scoper,
    )

    state = runtime.create(
        ResearchRequest(topic="Create this job"),
        run_id="run-created-only",
        require_approval=True,
    )

    assert state.status == "created"
    assert state.effects == ()
    assert state.conversation == (
        ConversationMessage(role="user", content="Create this job"),
    )
    assert scoper.calls == []
    assert planner.call_count == 0
    assert runner.questions == []


def test_an_explicit_brief_skips_the_optional_scoper(tmp_path) -> None:
    scoper = SequencedScoper([])
    planner = UsagePlanner()
    runtime = ResearchRuntime(
        store=JsonCheckpointStore(tmp_path),
        scoper=scoper,
        planner=planner,
        runner=AnyQuestionRunner(),
    )

    waiting = runtime.start(
        ResearchRequest(topic="Raw request", brief=_clear_brief()),
        run_id="run-direct-brief",
        require_approval=True,
    )

    assert waiting.state.status == "waiting_for_human"
    assert waiting.state.brief == _clear_brief()
    assert [effect.kind for effect in waiting.state.effects] == ["planner.plan"]
    assert scoper.calls == []
    assert planner.call_count == 1


def test_default_topic_planner_is_durable_and_can_be_approved(tmp_path) -> None:
    runner = AnyQuestionRunner()
    runtime = ResearchRuntime(
        store=JsonCheckpointStore(tmp_path),
        runner=runner,
    )
    waiting = runtime.start(
        ResearchRequest(
            topic="Fallback planner topic",
            min_sources=1,
            budget=ResearchBudget(max_tool_calls=1, max_research_steps=1),
        ),
        run_id="run-default-planner-approval",
        require_approval=True,
    ).state

    assert waiting.status == "waiting_for_human"
    assert waiting.plan is not None
    assert [effect.kind for effect in waiting.effects] == ["planner.plan"]
    completed = runtime.approve("run-default-planner-approval")

    assert completed.state.status == "completed"
    assert runner.questions == ["Fallback planner topic"]


def test_empty_generated_plan_cannot_bypass_approval(tmp_path) -> None:
    runner = AnyQuestionRunner()
    runtime = ResearchRuntime(
        store=JsonCheckpointStore(tmp_path),
        planner=EmptyPlanner(),
        runner=runner,
    )
    waiting = runtime.start(
        ResearchRequest(topic="Empty plan", budget=ResearchBudget(max_research_steps=1)),
        run_id="run-empty-plan",
        require_approval=True,
    ).state

    assert waiting.status == "waiting_for_human"
    assert waiting.plan is not None
    assert waiting.plan.questions == ()
    assert runner.questions == []
    with pytest.raises(ValueError, match="at least one question"):
        runtime.approve_plan(
            "run-empty-plan",
            expected_plan_hash=waiting.plan_hash or "",
        )


def test_clarification_pauses_before_planning_and_answer_only_mutates_state(
    tmp_path,
) -> None:
    clarification = ClarificationDecision(
        needs_clarification=True,
        reason="The comparison target is missing.",
        question="Which products should be compared?",
    )
    scoper = SequencedScoper(
        [
            ScopingRun(clarification=clarification),
            ScopingRun(
                clarification=ClarificationDecision(
                    needs_clarification=False,
                    reason="The answer resolves the scope.",
                ),
                brief=_clear_brief(),
            ),
        ]
    )
    planner = UsagePlanner()
    runner = AnyQuestionRunner()
    runtime = ResearchRuntime(
        store=JsonCheckpointStore(tmp_path),
        scoper=scoper,
        planner=planner,
        runner=runner,
    )

    waiting = runtime.start(
        ResearchRequest(topic="Compare them"),
        run_id="run-clarification",
        require_approval=True,
    )

    assert waiting.state.status == "waiting_for_human"
    assert waiting.state.current_step == "clarification"
    assert waiting.state.brief is None
    assert [effect.kind for effect in waiting.state.effects] == ["scope.resolve"]
    assert planner.call_count == 0
    assert runner.questions == []
    still_waiting = runtime.resume("run-clarification")
    assert still_waiting.state == waiting.state
    assert len(scoper.calls) == 1

    with pytest.raises(ValueError, match="run state changed"):
        runtime.answer_clarification(
            "run-clarification",
            answer="This stale answer must not be accepted.",
            expected_state_version=waiting.state.state_version - 1,
        )

    answered = runtime.answer_clarification(
        "run-clarification",
        answer="Compare product A with product B.",
        expected_state_version=waiting.state.state_version,
    )
    assert answered.status == "created"
    assert answered.current_step == "clarification_answered"
    assert answered.clarification is None
    assert [message.role for message in answered.conversation] == [
        "user",
        "assistant",
        "user",
    ]
    assert len(scoper.calls) == 1
    assert planner.call_count == 0

    plan_wait = runtime.resume("run-clarification")

    assert plan_wait.state.status == "waiting_for_human"
    assert plan_wait.state.current_step == "approval"
    assert plan_wait.state.brief == _clear_brief()
    scope_effects = [
        effect for effect in plan_wait.state.effects if effect.kind == "scope.resolve"
    ]
    assert len(scope_effects) == 2
    assert scope_effects[0].effect_id != scope_effects[1].effect_id
    assert len(scoper.calls) == 2
    assert planner.call_count == 1
    assert runner.questions == []


def test_scoper_can_request_only_one_clarification_round(tmp_path) -> None:
    clarification = ClarificationDecision(
        needs_clarification=True,
        reason="The target is still ambiguous.",
        question="Which target should be researched?",
    )
    clarification_run = ScopingRun(clarification=clarification)
    scoper = SequencedScoper([clarification_run, clarification_run])
    runtime = ResearchRuntime(
        store=JsonCheckpointStore(tmp_path),
        scoper=scoper,
        planner=UsagePlanner(),
        runner=AnyQuestionRunner(),
    )
    waiting = runtime.start(
        ResearchRequest(topic="Ambiguous target"),
        run_id="run-one-clarification",
    ).state
    answered = runtime.answer_clarification(
        "run-one-clarification",
        answer="Research target A.",
        expected_state_version=waiting.state_version,
    )
    failed = runtime.resume("run-one-clarification").state

    assert answered.status == "created"
    assert failed.status == "failed"
    assert failed.error_type == "RuntimeError"
    assert "another clarification" in (failed.error_message or "")
    assert len(scoper.calls) == 2


def test_edit_and_approve_are_append_only_and_execute_the_approved_plan(
    tmp_path,
) -> None:
    scoper = SequencedScoper(
        [
            ScopingRun(
                clarification=ClarificationDecision(
                    needs_clarification=False,
                    reason="The request is clear.",
                ),
                brief=_clear_brief(),
                usage=TokenUsage(input_tokens=5, output_tokens=6, total_tokens=11),
            )
        ]
    )
    planner = UsagePlanner()
    runner = AnyQuestionRunner()
    runtime = ResearchRuntime(
        store=JsonCheckpointStore(tmp_path),
        scoper=scoper,
        planner=planner,
        runner=runner,
    )
    request = ResearchRequest(
        topic="Research reliable agents",
        min_sources=1,
        budget=ResearchBudget(max_tool_calls=1, max_research_steps=1),
    )
    waiting = runtime.start(
        request,
        run_id="run-plan-edit",
        require_approval=True,
    ).state
    original_hash = waiting.plan_hash
    assert original_hash is not None
    original_planner_result = next(
        effect.result for effect in waiting.effects if effect.kind == "planner.plan"
    )
    assert original_planner_result["plan"]["questions"][0]["question"] == (
        "Generated question"
    )

    edited_plan = ResearchPlan(
        objective="Human-scoped objective",
        questions=(
            ResearchQuestion(
                id="human-q1",
                question="Human-edited question",
                rationale="This is the highest-value comparison.",
                priority=1,
            ),
        ),
    )
    edited = runtime.edit_plan(
        "run-plan-edit",
        edited_plan,
        expected_plan_hash=original_hash,
    )

    assert edited.plan == edited_plan
    assert edited.approved_plan is None
    assert len(edited.plan_controls) == 1
    assert edited.plan_controls[0].kind == "edit"
    assert [effect.kind for effect in edited.effects] == [
        "scope.resolve",
        "planner.plan",
    ]
    assert next(
        effect.result for effect in edited.effects if effect.kind == "planner.plan"
    ) == original_planner_result
    assert runtime.edit_plan(
        "run-plan-edit",
        edited_plan,
        expected_plan_hash=original_hash,
    ) == edited

    with pytest.raises(ValueError, match="plan changed"):
        runtime.approve_plan(
            "run-plan-edit",
            expected_plan_hash=original_hash,
        )

    edited_hash = edited.plan_hash
    assert edited_hash is not None
    approved = runtime.approve_plan(
        "run-plan-edit",
        expected_plan_hash=edited_hash,
    )

    assert approved.status == "created"
    assert approved.current_step == "approved"
    assert approved.approved_plan == edited_plan
    assert [control.kind for control in approved.plan_controls] == ["edit", "approve"]
    assert all(
        effect.kind not in {"plan.edit", "plan.approve"}
        for effect in approved.effects
    )

    restarted = ResearchRuntime(
        store=JsonCheckpointStore(tmp_path),
        scoper=scoper,
        planner=planner,
        runner=runner,
    )
    completed = restarted.resume("run-plan-edit")

    assert completed.state.status == "completed"
    assert runner.questions == ["Human-edited question"]
    assert planner.call_count == 1
    assert scoper.calls and len(scoper.calls) == 1
    assert completed.result is not None
    assert completed.result.usage == TokenUsage(
        input_tokens=8,
        output_tokens=10,
        total_tokens=18,
    )
    assert '"objective": "Human-scoped objective"' in runner.tasks[0]
    assert completed.state.brief == _clear_brief()


def test_an_objective_only_edit_reaches_every_downstream_agent(tmp_path) -> None:
    planner = ObjectiveRecordingPlanner()
    runner = ObjectiveRecordingRunner()
    report_agent = ObjectiveRecordingReportAgent()
    runtime = ResearchRuntime(
        store=JsonCheckpointStore(tmp_path),
        planner=planner,
        runner=runner,
        report_agent=report_agent,
    )
    original_request = ResearchRequest(
        topic="Trace one approved objective",
        brief=_clear_brief(),
        min_sources=2,
        require_citations=False,
        budget=ResearchBudget(
            max_tool_calls=2,
            max_research_steps=2,
            max_parallel_workers=1,
            max_revision_rounds=1,
        ),
    )
    waiting = runtime.start(
        original_request,
        run_id="run-objective-propagation",
        require_approval=True,
    ).state
    assert waiting.plan is not None
    original_hash = waiting.plan_hash
    assert original_hash is not None
    approved_objective = "Human-approved objective"
    edited = runtime.edit_plan(
        "run-objective-propagation",
        ResearchPlan(
            objective=approved_objective,
            questions=waiting.plan.questions,
        ),
        expected_plan_hash=original_hash,
    )
    completed = runtime.approve("run-objective-propagation")

    assert completed.state.status == "completed"
    assert completed.result is not None
    assert completed.result.plan.objective == approved_objective
    assert planner.objectives == [_clear_brief().objective, approved_objective]
    assert len(runner.tasks) == 2
    assert all(
        f'"objective": "{approved_objective}"' in task for task in runner.tasks
    )
    assert report_agent.objectives
    assert set(report_agent.objectives) == {approved_objective}
    assert completed.state.request == original_request

    retried = runtime.approve_plan(
        "run-objective-propagation",
        expected_plan_hash=edited.plan_hash or "",
    )
    assert retried == completed.state


class CrashAfterCompletedScopeStore(JsonCheckpointStore):
    def __init__(self, root) -> None:
        super().__init__(root)
        self._crashed = False

    def update_owned(self, run_id, owner_id, ttl_seconds, mutation):
        current = self.load(run_id)
        if (
            not self._crashed
            and current.request.brief is None
            and any(
                effect.kind == "scope.resolve" and effect.status == "completed"
                for effect in current.effects
            )
        ):
            self._crashed = True
            raise SimulatedProcessCrash("crash after the scope effect was committed")
        return super().update_owned(run_id, owner_id, ttl_seconds, mutation)


def test_resume_replays_a_completed_scope_effect_without_recalling_the_model(
    tmp_path,
) -> None:
    scoper = SequencedScoper(
        [
            ScopingRun(
                clarification=ClarificationDecision(
                    needs_clarification=False,
                    reason="The request is clear.",
                ),
                brief=_clear_brief(),
            )
        ]
    )
    store = CrashAfterCompletedScopeStore(tmp_path)
    runtime = ResearchRuntime(
        store=store,
        scoper=scoper,
        planner=UsagePlanner(),
        runner=AnyQuestionRunner(),
    )

    with pytest.raises(SimulatedProcessCrash):
        runtime.start(
            ResearchRequest(topic="Crash-safe scoping"),
            run_id="run-scope-replay",
            require_approval=True,
        )

    assert len(scoper.calls) == 1
    waiting = runtime.resume("run-scope-replay")

    assert waiting.state.status == "waiting_for_human"
    assert waiting.state.current_step == "approval"
    assert waiting.state.brief == _clear_brief()
    assert len(scoper.calls) == 1


def test_sqlite_plan_edit_compare_and_swap_allows_only_one_concurrent_edit(
    tmp_path,
) -> None:
    database = tmp_path / "plan-controls.sqlite3"
    runtime = ResearchRuntime(
        store=SQLiteCheckpointStore(database),
        planner=UsagePlanner(),
        runner=AnyQuestionRunner(),
    )
    waiting = runtime.start(
        ResearchRequest(
            topic="Concurrent plan editing",
            min_sources=1,
            budget=ResearchBudget(max_tool_calls=1, max_research_steps=1),
        ),
        run_id="run-concurrent-edit",
        require_approval=True,
    ).state
    base_hash = waiting.plan_hash
    assert base_hash is not None
    barrier = Barrier(2)
    successes: list[RunState] = []
    errors: list[Exception] = []

    def edit(question: str) -> None:
        control = ResearchRuntime(
            store=SQLiteCheckpointStore(database),
            planner=UsagePlanner(),
            runner=AnyQuestionRunner(),
        )
        barrier.wait(timeout=2)
        try:
            successes.append(
                control.edit_plan(
                    "run-concurrent-edit",
                    ResearchPlan(
                        objective="Generated objective",
                        questions=(
                            ResearchQuestion(id="q1", question=question, priority=1),
                        ),
                    ),
                    expected_plan_hash=base_hash,
                )
            )
        except Exception as error:  # noqa: BLE001 - collect the losing CAS result.
            errors.append(error)

    threads = [
        Thread(target=edit, args=("First competing edit",)),
        Thread(target=edit, args=("Second competing edit",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert len(successes) == 1
    assert len(errors) == 1
    assert "plan changed" in str(errors[0])
    restored = SQLiteCheckpointStore(database).load("run-concurrent-edit")
    assert len(restored.plan_controls) == 1


def test_plan_edit_rejects_invalid_or_late_changes(tmp_path) -> None:
    runtime = ResearchRuntime(
        store=JsonCheckpointStore(tmp_path),
        planner=UsagePlanner(),
        runner=AnyQuestionRunner(),
    )
    waiting = runtime.start(
        ResearchRequest(
            topic="Validate plan edits",
            budget=ResearchBudget(max_research_steps=2),
        ),
        run_id="run-invalid-edit",
        require_approval=True,
    ).state
    plan_hash = waiting.plan_hash
    assert plan_hash is not None
    assert waiting.plan is not None

    with pytest.raises(ValueError, match="must differ"):
        runtime.edit_plan(
            "run-invalid-edit",
            waiting.plan,
            expected_plan_hash=plan_hash,
        )

    with pytest.raises(ValueError, match="at least one question"):
        runtime.edit_plan(
            "run-invalid-edit",
            ResearchPlan(objective="Generated objective", questions=()),
            expected_plan_hash=plan_hash,
        )
    with pytest.raises(ValueError, match="must be unique"):
        runtime.edit_plan(
            "run-invalid-edit",
            ResearchPlan(
                objective="Generated objective",
                questions=(
                    ResearchQuestion(id="q1", question="Same question"),
                    ResearchQuestion(id="q2", question=" same   QUESTION "),
                ),
            ),
            expected_plan_hash=plan_hash,
        )

    approved = runtime.approve_plan(
        "run-invalid-edit",
        expected_plan_hash=plan_hash,
    )
    with pytest.raises(ValueError, match="not waiting"):
        runtime.edit_plan(
            "run-invalid-edit",
            ResearchPlan(
                objective="Generated objective",
                questions=(ResearchQuestion(id="q1", question="Too late"),),
            ),
            expected_plan_hash=approved.plan_hash or "",
        )


def test_reject_optionally_compares_the_plan_seen_by_the_caller(tmp_path) -> None:
    runner = AnyQuestionRunner()
    runtime = ResearchRuntime(
        store=JsonCheckpointStore(tmp_path),
        planner=UsagePlanner(),
        runner=runner,
    )
    waiting = runtime.start(
        ResearchRequest(
            topic="Reject with optimistic concurrency",
            budget=ResearchBudget(max_research_steps=1),
        ),
        run_id="run-reject-cas",
        require_approval=True,
    ).state
    original_hash = waiting.plan_hash
    assert original_hash is not None
    edited = runtime.edit_plan(
        "run-reject-cas",
        ResearchPlan(
            objective="Generated objective",
            questions=(ResearchQuestion(id="q1", question="Edited question"),),
        ),
        expected_plan_hash=original_hash,
    )

    with pytest.raises(ValueError, match="plan changed"):
        runtime.reject(
            "run-reject-cas",
            expected_plan_hash=original_hash,
        )

    current = runtime.get("run-reject-cas")
    assert current.status == "waiting_for_human"
    assert current.plan == edited.plan
    rejected = runtime.reject(
        "run-reject-cas",
        reason="The edited plan is still too broad.",
        expected_plan_hash=edited.plan_hash,
    )
    assert rejected.state.status == "cancelled"
    assert runner.questions == []


def test_plan_controls_reject_a_different_runtime_before_mutating_state(
    tmp_path,
) -> None:
    runner = AnyQuestionRunner()
    runner._model = "model-a"
    runtime = ResearchRuntime(
        store=JsonCheckpointStore(tmp_path),
        planner=UsagePlanner(),
        runner=runner,
    )
    waiting = runtime.start(
        ResearchRequest(topic="Protect plan controls"),
        run_id="run-control-config",
        require_approval=True,
    ).state
    plan_hash = waiting.plan_hash
    assert plan_hash is not None

    with pytest.raises(RuntimeError, match="runner configuration is required"):
        ResearchRuntime(store=JsonCheckpointStore(tmp_path)).approve_plan(
            "run-control-config",
            expected_plan_hash=plan_hash,
        )
    assert runtime.get("run-control-config") == waiting

    runner._model = "model-b"
    with pytest.raises(ValueError, match="configuration does not match"):
        runtime.approve_plan(
            "run-control-config",
            expected_plan_hash=plan_hash,
        )

    assert runtime.get("run-control-config") == waiting


def test_plan_hash_tracks_edit_lineage_to_prevent_aba_updates(tmp_path) -> None:
    runtime = ResearchRuntime(
        store=JsonCheckpointStore(tmp_path),
        planner=UsagePlanner(),
        runner=AnyQuestionRunner(),
    )
    waiting = runtime.start(
        ResearchRequest(
            topic="Prevent stale plan writes",
            budget=ResearchBudget(max_research_steps=1),
        ),
        run_id="run-plan-aba",
        require_approval=True,
    ).state
    original_plan = waiting.plan
    original_hash = waiting.plan_hash
    assert original_plan is not None
    assert original_hash is not None

    edited = runtime.edit_plan(
        "run-plan-aba",
        ResearchPlan(
            objective=original_plan.objective,
            questions=(ResearchQuestion(id="q1", question="Temporary question"),),
        ),
        expected_plan_hash=original_hash,
    )
    restored_content = runtime.edit_plan(
        "run-plan-aba",
        original_plan,
        expected_plan_hash=edited.plan_hash or "",
    )

    assert restored_content.plan == original_plan
    assert restored_content.plan_hash != original_hash
    with pytest.raises(ValueError, match="plan changed"):
        runtime.edit_plan(
            "run-plan-aba",
            ResearchPlan(
                objective=original_plan.objective,
                questions=(ResearchQuestion(id="q1", question="Stale edit"),),
            ),
            expected_plan_hash=original_hash,
        )
    assert len(runtime.get("run-plan-aba").plan_controls) == 2


def test_checkpoint_rejects_a_corrupted_plan_control_hash(tmp_path) -> None:
    store = JsonCheckpointStore(tmp_path)
    runtime = ResearchRuntime(
        store=store,
        planner=UsagePlanner(),
        runner=AnyQuestionRunner(),
    )
    waiting = runtime.start(
        ResearchRequest(
            topic="Detect checkpoint corruption",
            budget=ResearchBudget(max_research_steps=1),
        ),
        run_id="run-corrupt-plan-control",
        require_approval=True,
    ).state
    assert waiting.plan is not None
    runtime.edit_plan(
        "run-corrupt-plan-control",
        ResearchPlan(
            objective=waiting.plan.objective,
            questions=(ResearchQuestion(id="q1", question="Edited question"),),
        ),
        expected_plan_hash=waiting.plan_hash or "",
    )
    checkpoint_path = tmp_path / "run-corrupt-plan-control.json"
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    payload["plan_controls"][0]["plan_hash"] = "f" * 64
    checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid plan hash"):
        store.load("run-corrupt-plan-control")

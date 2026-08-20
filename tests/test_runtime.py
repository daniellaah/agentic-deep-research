import json
from threading import Event, Thread

from agentic_deep_research import (
    AgentRun,
    Citation,
    PlanningRun,
    ResearchBudget,
    ResearchPlan,
    ResearchQuestion,
    ResearchRequest,
    Source,
)
from agentic_deep_research.runtime import (
    JsonCheckpointStore,
    ResearchRuntime,
    RetryPolicy,
    RunState,
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
            raise ConnectionError("temporary provider disconnect")
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

    no_automatic_retry = RetryPolicy(max_attempts=1)
    first_runtime = ResearchRuntime(
        store=store,
        planner=planner,
        runner=runner,
        retry_policy=no_automatic_retry,
    )
    failed = first_runtime.start(request, run_id="run-resume")

    assert failed.state.status == "failed"
    assert runner.calls == {"First question": 1, "Second question": 1}

    restarted_runtime = ResearchRuntime(
        store=store,
        planner=planner,
        runner=runner,
        retry_policy=no_automatic_retry,
    )
    resumed = restarted_runtime.resume("run-resume")

    assert resumed.state.status == "completed"
    assert resumed.result is not None
    assert resumed.result.status == "completed"
    assert planner.call_count == 1
    assert runner.calls == {"First question": 1, "Second question": 2}


class TransientRunner:
    def __init__(self, error: Exception) -> None:
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
    assert outcome.state.effects[0].attempts == 2

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
    assert outcome.state.effects[0].attempts == 1


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
    runtime = ResearchRuntime(
        store=JsonCheckpointStore(tmp_path),
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

    requested = runtime.request_cancel("run-cancel", reason="User stopped the run")
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

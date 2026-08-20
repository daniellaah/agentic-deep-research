import time
from threading import Event, Lock

from agentic_deep_research.checkpoint import SQLiteCheckpointStore
from agentic_deep_research.models import (
    AgentRun,
    Citation,
    PlanningRun,
    ResearchBudget,
    ResearchPlan,
    ResearchQuestion,
    ResearchRequest,
    Source,
)
from agentic_deep_research.runtime import ResearchRuntime
from agentic_deep_research.service import ResearchService


class BlockingRunner:
    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()

    def run(
        self,
        *,
        instructions: str,
        task: str,
        budget: ResearchBudget,
    ) -> AgentRun:
        del instructions, task, budget
        self.started.set()
        assert self.release.wait(timeout=5)
        return _cited_run("Blocking research finished.")


class OneQuestionPlanner:
    def __init__(self) -> None:
        self.call_count = 0

    def plan(self, **_: object) -> PlanningRun:
        self.call_count += 1
        return PlanningRun(
            plan=ResearchPlan(
                objective="Original objective",
                questions=(
                    ResearchQuestion(
                        id="q1",
                        question="Original research question",
                        priority=1,
                    ),
                ),
            )
        )


class CapturingRunner:
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
        return _cited_run("Edited research finished.")


class BlockingCapturingRunner(CapturingRunner):
    def __init__(self) -> None:
        super().__init__()
        self.started = Event()
        self.release = Event()

    def run(
        self,
        *,
        instructions: str,
        task: str,
        budget: ResearchBudget,
    ) -> AgentRun:
        self.started.set()
        assert self.release.wait(timeout=5)
        return super().run(instructions=instructions, task=task, budget=budget)


class ResumeCounter:
    def __init__(self) -> None:
        self._lock = Lock()
        self.value = 0

    def increment(self) -> None:
        with self._lock:
            self.value += 1


class CountingRuntime(ResearchRuntime):
    def __init__(self, *args, counter: ResumeCounter, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._counter = counter

    def resume(self, run_id: str, *, retry_ambiguous: bool = False):
        self._counter.increment()
        return super().resume(run_id, retry_ambiguous=retry_ambiguous)


def test_service_persists_a_run_before_background_provider_work(tmp_path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.sqlite3")
    runner = BlockingRunner()
    service = ResearchService(
        store=store,
        runtime_factory=lambda _: ResearchRuntime(store=store, runner=runner),
        max_concurrent_runs=1,
    )
    try:
        created = service.start(
            _request("Background research"),
            run_id="background-run",
            require_approval=False,
        )

        assert created.status == "created"
        assert store.load(created.run_id).run_id == created.run_id
        assert runner.started.wait(timeout=5)
        assert store.load(created.run_id).status == "running"

        runner.release.set()
        completed = _wait_for_status(store, created.run_id, "completed")
        assert completed.result is not None
    finally:
        runner.release.set()
        service.close()


def test_service_executes_the_approved_edit_without_mutating_the_original_plan(
    tmp_path,
) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.sqlite3")
    planner = OneQuestionPlanner()
    runner = CapturingRunner()
    service = ResearchService(
        store=store,
        runtime_factory=lambda _: ResearchRuntime(
            store=store,
            runner=runner,
            planner=planner,
        ),
        max_concurrent_runs=1,
    )
    try:
        service.start(
            _request("Review a generated plan"),
            run_id="edited-plan-run",
            require_approval=True,
        )
        waiting = _wait_for_status(store, "edited-plan-run", "waiting_for_human")
        assert waiting.plan_hash is not None
        original_effect = next(
            effect for effect in waiting.effects if effect.kind == "planner.plan"
        )

        edited = service.edit_plan(
            waiting.run_id,
            objective="Edited objective",
            questions=(("Edited research question", "Narrow the scope"),),
            expected_plan_hash=waiting.plan_hash,
        )
        assert edited.plan_hash is not None
        service.approve_plan(
            waiting.run_id,
            expected_plan_hash=edited.plan_hash,
        )

        completed = _wait_for_status(store, waiting.run_id, "completed")
        preserved_effect = next(
            effect for effect in completed.effects if effect.kind == "planner.plan"
        )
        assert preserved_effect == original_effect
        assert planner.call_count == 1
        assert len(runner.tasks) == 1
        assert "Edited research question" in runner.tasks[0]
        assert "Original research question" not in runner.tasks[0]
        assert [control.kind for control in completed.plan_controls] == [
            "edit",
            "approve",
        ]
    finally:
        service.close()


def test_duplicate_approval_schedules_only_one_continuation(tmp_path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.sqlite3")
    planner = OneQuestionPlanner()
    runner = BlockingCapturingRunner()
    counter = ResumeCounter()

    def runtime_factory(_: ResearchRequest) -> ResearchRuntime:
        return CountingRuntime(
            store=store,
            runner=runner,
            planner=planner,
            counter=counter,
        )

    service = ResearchService(
        store=store,
        runtime_factory=runtime_factory,
        max_concurrent_runs=1,
    )
    service.start(
        _request("Idempotent approval"),
        run_id="duplicate-approval",
        require_approval=True,
    )
    waiting = _wait_for_status(store, "duplicate-approval", "waiting_for_human")
    assert waiting.plan_hash is not None

    service.approve_plan(
        waiting.run_id,
        expected_plan_hash=waiting.plan_hash,
    )
    assert runner.started.wait(timeout=5)
    for _ in range(25):
        service.approve_plan(
            waiting.run_id,
            expected_plan_hash=waiting.plan_hash,
        )

    runner.release.set()
    _wait_for_status(store, waiting.run_id, "completed")
    service.approve_plan(
        waiting.run_id,
        expected_plan_hash=waiting.plan_hash,
    )
    service.close()

    assert counter.value == 2
    assert len(runner.tasks) == 1


def _request(topic: str) -> ResearchRequest:
    return ResearchRequest(
        topic=topic,
        min_sources=1,
        budget=ResearchBudget(
            max_tool_calls=1,
            max_research_steps=1,
            max_parallel_workers=1,
        ),
    )


def _cited_run(report_text: str) -> AgentRun:
    source = Source("Product source", "https://example.com/product")
    report = f"{report_text} [1]"
    start = report.index("[1]")
    return AgentRun(
        report=report,
        sources=(source,),
        citations=(Citation(source, start, start + 3),),
    )


def _wait_for_status(
    store: SQLiteCheckpointStore,
    run_id: str,
    expected: str,
):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = store.load(run_id)
        if state.status == expected:
            return state
        time.sleep(0.01)
    raise AssertionError(
        f"run {run_id} did not reach {expected}; current={store.load(run_id).status}"
    )

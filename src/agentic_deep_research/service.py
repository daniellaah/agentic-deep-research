"""In-process product service for durable background research jobs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from threading import RLock
from typing import Any

from .checkpoint import RunState, SQLiteCheckpointStore, _transition
from .models import ResearchPlan, ResearchQuestion, ResearchRequest
from .runtime import ResearchRuntime


class ServiceBusyError(RuntimeError):
    """The bounded local job queue has no capacity for another run."""


class RunAlreadyScheduledError(RuntimeError):
    """A live local worker already owns the requested run."""


@dataclass(frozen=True)
class ServiceLimits:
    """Server-side ceilings that bound user-controlled provider cost."""

    max_pending_runs: int = 32
    max_tool_calls: int = 32
    max_output_tokens: int = 50_000
    max_research_steps: int = 16
    max_parallel_workers: int = 4
    max_context_chars: int = 50_000
    max_verification_tool_calls: int = 8
    max_revision_rounds: int = 4
    max_min_sources: int = 10
    max_brief_chars: int = 40_000

    def __post_init__(self) -> None:
        if min(
            self.max_pending_runs,
            self.max_tool_calls,
            self.max_output_tokens,
            self.max_research_steps,
            self.max_parallel_workers,
            self.max_context_chars,
            self.max_min_sources,
            self.max_brief_chars,
        ) < 1:
            raise ValueError("positive service limits must be at least 1")
        if self.max_verification_tool_calls < 0:
            raise ValueError("max_verification_tool_calls must not be negative")
        if self.max_revision_rounds < 0:
            raise ValueError("max_revision_rounds must not be negative")

    def validate(self, request: ResearchRequest) -> None:
        """Reject a syntactically valid request that exceeds service policy."""
        budget = request.budget
        checks = {
            "max_tool_calls": (budget.max_tool_calls, self.max_tool_calls),
            "max_output_tokens": (
                budget.max_output_tokens,
                self.max_output_tokens,
            ),
            "max_research_steps": (
                budget.max_research_steps,
                self.max_research_steps,
            ),
            "max_parallel_workers": (
                budget.max_parallel_workers,
                self.max_parallel_workers,
            ),
            "max_context_chars": (
                budget.max_context_chars,
                self.max_context_chars,
            ),
            "max_verification_tool_calls": (
                budget.max_verification_tool_calls,
                self.max_verification_tool_calls,
            ),
            "max_revision_rounds": (
                budget.max_revision_rounds,
                self.max_revision_rounds,
            ),
            "min_sources": (request.min_sources, self.max_min_sources),
        }
        for name, (actual, maximum) in checks.items():
            if actual > maximum:
                raise ValueError(f"{name} exceeds the service limit of {maximum}")
        if request.brief is not None:
            brief_chars = sum(
                len(value)
                for value in (
                    request.brief.research_question,
                    request.brief.objective,
                    request.brief.deliverable,
                    *request.brief.scope_inclusions,
                    *request.brief.scope_exclusions,
                    *request.brief.constraints,
                    *request.brief.success_criteria,
                )
            )
            if brief_chars > self.max_brief_chars:
                raise ValueError(
                    "research brief exceeds the service limit of "
                    f"{self.max_brief_chars} characters"
                )


RuntimeFactory = Callable[[ResearchRequest], ResearchRuntime]


class ResearchService:
    """Persist controls synchronously and execute provider work in bounded threads."""

    def __init__(
        self,
        *,
        store: SQLiteCheckpointStore,
        runtime_factory: RuntimeFactory,
        max_concurrent_runs: int = 2,
        limits: ServiceLimits | None = None,
    ) -> None:
        if max_concurrent_runs < 1:
            raise ValueError("max_concurrent_runs must be at least 1")
        self._store = store
        self._runtime_factory = runtime_factory
        self._limits = limits or ServiceLimits()
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent_runs,
            thread_name_prefix="research-job",
        )
        self._jobs: dict[str, Future[Any]] = {}
        self._scheduled_plan_controls: set[tuple[str, str]] = set()
        self._lock = RLock()
        self._closed = False

    @property
    def store(self) -> SQLiteCheckpointStore:
        """Return the durable store used by status, events, and artifacts."""
        return self._store

    def start(
        self,
        request: ResearchRequest,
        *,
        run_id: str | None = None,
        require_approval: bool = True,
    ) -> RunState:
        """Create a checkpoint before enqueueing any provider-facing work."""
        self._limits.validate(request)
        runtime = self._runtime_factory(request)
        with self._lock:
            self._ensure_open()
            if len(self._jobs) >= self._limits.max_pending_runs:
                raise ServiceBusyError("research job queue is full")
            state = runtime.create(
                request,
                run_id=run_id,
                require_approval=require_approval,
            )
            self._submit_locked(
                state.run_id,
                lambda: runtime.resume(state.run_id),
                predecessor=None,
            )
        return state

    def get(self, run_id: str) -> RunState:
        """Return a durable snapshot without starting work."""
        return self._store.load(run_id)

    def answer_clarification(
        self,
        run_id: str,
        *,
        answer: str,
        expected_state_version: int | None = None,
    ) -> RunState:
        """Persist one answer, then continue scoping after the prior job releases."""
        runtime = self._runtime_for(run_id)
        state = runtime.answer_clarification(
            run_id,
            answer=answer,
            expected_state_version=expected_state_version,
        )
        self._enqueue_chained(run_id, lambda: runtime.resume(run_id))
        return state

    def edit_plan(
        self,
        run_id: str,
        *,
        objective: str | None,
        questions: Sequence[tuple[str, str]],
        expected_plan_hash: str,
    ) -> RunState:
        """Append a human plan edit while preserving the model's original plan."""
        runtime = self._runtime_for(run_id)
        current = runtime.get(run_id)
        if current.plan is None:
            raise ValueError("run has no plan to edit")
        revision = current.plan.revision
        plan = ResearchPlan(
            objective=(objective or current.plan.objective).strip(),
            questions=tuple(
                ResearchQuestion(
                    id=f"h{revision + 1}q{index}",
                    question=question.strip(),
                    rationale=rationale.strip(),
                    priority=index,
                )
                for index, (question, rationale) in enumerate(questions, start=1)
            ),
            revision=revision,
        )
        return runtime.edit_plan(
            run_id,
            plan,
            expected_plan_hash=expected_plan_hash,
        )

    def approve_plan(self, run_id: str, *, expected_plan_hash: str) -> RunState:
        """Persist the exact approved snapshot, then enqueue research execution."""
        runtime = self._runtime_for(run_id)
        state = runtime.approve_plan(
            run_id,
            expected_plan_hash=expected_plan_hash,
        )
        if state.status not in {"completed", "cancelled"}:
            approval = next(
                (
                    control
                    for control in reversed(state.plan_controls)
                    if control.kind == "approve"
                ),
                None,
            )
            if approval is None:
                raise RuntimeError("approved run has no durable approval record")
            self._enqueue_plan_control_once(
                run_id,
                approval.control_id,
                lambda: runtime.resume(run_id),
            )
        return state

    def reject_plan(
        self,
        run_id: str,
        *,
        reason: str = "",
        expected_plan_hash: str | None = None,
    ) -> RunState:
        """Reject a pending plan without enqueueing research."""
        runtime = self._runtime_for(run_id)
        return runtime.reject(
            run_id,
            reason=reason,
            expected_plan_hash=expected_plan_hash,
        ).state

    def cancel(self, run_id: str, *, reason: str = "") -> RunState:
        """Request cooperative cancellation of a queued or active run."""
        runtime = self._runtime_for(run_id)
        return runtime.request_cancel(run_id, reason=reason)

    def resume(self, run_id: str, *, retry_ambiguous: bool = False) -> RunState:
        """Enqueue an explicit recovery when no local job is already scheduled."""
        runtime = self._runtime_for(run_id)
        state = runtime.get(run_id)
        self._enqueue_new(
            run_id,
            lambda: runtime.resume(run_id, retry_ambiguous=retry_ambiguous),
        )
        return state

    def close(self, *, wait: bool = True) -> None:
        """Stop accepting work and shut down the local executor."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=False)

    def _runtime_for(self, run_id: str) -> ResearchRuntime:
        state = self._store.load(run_id)
        return self._runtime_factory(state.request)

    def _enqueue_new(self, run_id: str, operation: Callable[[], object]) -> None:
        with self._lock:
            self._ensure_open()
            predecessor = self._jobs.get(run_id)
            if predecessor is not None and not predecessor.done():
                raise RunAlreadyScheduledError(f"run is already scheduled: {run_id}")
            if predecessor is None and len(self._jobs) >= self._limits.max_pending_runs:
                raise ServiceBusyError("research job queue is full")
            self._submit_locked(run_id, operation, predecessor=None)

    def _enqueue_chained(self, run_id: str, operation: Callable[[], object]) -> None:
        with self._lock:
            self._ensure_open()
            predecessor = self._jobs.get(run_id)
            if predecessor is None and len(self._jobs) >= self._limits.max_pending_runs:
                raise ServiceBusyError("research job queue is full")
            self._submit_locked(run_id, operation, predecessor=predecessor)

    def _enqueue_plan_control_once(
        self,
        run_id: str,
        control_id: str,
        operation: Callable[[], object],
    ) -> None:
        """Schedule at most one continuation for one durable approval record."""
        key = (run_id, control_id)
        with self._lock:
            self._ensure_open()
            if key in self._scheduled_plan_controls:
                return
            predecessor = self._jobs.get(run_id)
            if predecessor is None and len(self._jobs) >= self._limits.max_pending_runs:
                raise ServiceBusyError("research job queue is full")
            self._scheduled_plan_controls.add(key)
            try:
                self._submit_locked(run_id, operation, predecessor=predecessor)
            except BaseException:
                self._scheduled_plan_controls.discard(key)
                raise

    def _submit_locked(
        self,
        run_id: str,
        operation: Callable[[], object],
        *,
        predecessor: Future[Any] | None,
    ) -> None:
        def execute() -> object | None:
            if predecessor is not None:
                with suppress(Exception):
                    predecessor.result()
            try:
                return operation()
            except Exception as error:  # noqa: BLE001 - make background failure visible.
                self._record_background_failure(run_id, error)
                return None

        future = self._executor.submit(execute)
        self._jobs[run_id] = future

        def discard(completed: Future[Any]) -> None:
            try:
                terminal = self._store.load(run_id).status in {
                    "completed",
                    "cancelled",
                }
            except (FileNotFoundError, TypeError, ValueError):
                terminal = False
            with self._lock:
                if self._jobs.get(run_id) is completed:
                    del self._jobs[run_id]
                if terminal:
                    self._scheduled_plan_controls = {
                        key
                        for key in self._scheduled_plan_controls
                        if key[0] != run_id
                    }

        future.add_done_callback(discard)

    def _record_background_failure(self, run_id: str, error: Exception) -> None:
        if isinstance(error, RuntimeError) and "already executing" in str(error):
            return

        def mutation(state: RunState) -> RunState:
            if state.status in {"completed", "cancelled", "waiting_for_human"}:
                return state
            return _transition(
                state,
                status="failed",
                error_type="BackgroundExecutionError",
                error_message=str(error),
                current_step="background_execution",
            )

        try:
            self._store.update(run_id, mutation)
        except (FileNotFoundError, RuntimeError, ValueError):
            # A concurrent terminal transition or deleted run wins.
            return

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("research service is closed")

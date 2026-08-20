"""Public durable runtime for long-running research workflows."""

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import Event, Thread
from uuid import uuid4

from ._journal import (
    RetryPolicy,
    _component_model,
    _EffectJournal,
    _engine_fingerprint,
    _JournaledPlanner,
    _JournaledReportAgent,
    _JournaledRunner,
    _RunCancelled,
    _WaitingForApproval,
)
from .checkpoint import (
    CheckpointStore,
    EffectRecord,
    JsonCheckpointStore,
    RunState,
    SQLiteCheckpointStore,
    _cancelled_state,
    _transition,
)
from .models import ResearchRequest, ResearchResult
from .planning import ResearchPlanner
from .reporting import ReportAgent
from .runner import AgentRunner
from .workflow import run_research

__all__ = [
    "CheckpointStore",
    "EffectRecord",
    "JsonCheckpointStore",
    "ResearchRuntime",
    "RetryPolicy",
    "RunState",
    "RuntimeOutcome",
    "SQLiteCheckpointStore",
]


@dataclass(frozen=True)
class RuntimeOutcome:
    """Latest durable state plus a result when the run completed."""

    state: RunState
    result: ResearchResult | None = None


class ResearchRuntime:
    """Run the existing workflow with durable provider-call checkpoints."""

    def __init__(
        self,
        *,
        store: CheckpointStore,
        runner: AgentRunner | None = None,
        planner: ResearchPlanner | None = None,
        report_agent: ReportAgent | None = None,
        retry_policy: RetryPolicy | None = None,
        lease_ttl_seconds: float = 30,
        sleeper: Callable[[float], None] = time.sleep,
        random_source: Callable[[], float] = random.random,
    ) -> None:
        self._store = store
        self._runner = runner
        self._planner = planner
        self._report_agent = report_agent
        self._retry_policy = retry_policy or RetryPolicy()
        if lease_ttl_seconds <= 0:
            raise ValueError("lease_ttl_seconds must be positive")
        self._lease_ttl_seconds = lease_ttl_seconds
        self._sleeper = sleeper
        self._random_source = random_source

    @property
    def model_name(self) -> str:
        """Return the runner model included in durable configuration identity."""
        return _component_model(self._runner)

    @property
    def planner_model_name(self) -> str:
        """Return the planner model, when the planner is model-backed."""
        return _component_model(self._planner)

    @property
    def report_model_name(self) -> str:
        """Return the report-agent model, when reporting is model-backed."""
        return _component_model(self._report_agent)

    @property
    def corpus_sha256(self) -> str | None:
        """Return the runner's fixed-corpus identity, when one is configured."""
        value = getattr(self._runner, "corpus_sha256", None)
        return value if isinstance(value, str) else None

    @property
    def retry_policy(self) -> RetryPolicy:
        """Return the retry policy enforced and journaled by this runtime."""
        return self._retry_policy

    def start(
        self,
        request: ResearchRequest,
        *,
        run_id: str | None = None,
        require_approval: bool = False,
    ) -> RuntimeOutcome:
        """Create and execute a new durable research run."""
        engine_fingerprint = self._current_engine_fingerprint()
        state = replace(
            RunState.create(request, run_id=run_id),
            requires_approval=require_approval,
            engine_fingerprint=engine_fingerprint,
            model_name=_component_model(self._runner),
        )
        self._store.create(state)
        return self._execute(state)

    def resume(
        self,
        run_id: str,
        *,
        retry_ambiguous: bool = False,
    ) -> RuntimeOutcome:
        """Resume a run, replaying completed effects from its checkpoint."""
        state = self._store.load(run_id)
        if state.status == "completed":
            return RuntimeOutcome(state=state, result=state.result)
        if state.status == "cancelled":
            return RuntimeOutcome(state=state)
        return self._execute(
            state,
            resume_mode=True,
            retry_ambiguous=retry_ambiguous,
        )

    def get(self, run_id: str) -> RunState:
        """Return the latest persisted state without executing the workflow."""
        return self._store.load(run_id)

    def approve(self, run_id: str) -> RuntimeOutcome:
        """Approve a pending plan and continue the durable run."""

        def mutation(state: RunState) -> RunState:
            if state.status != "waiting_for_human" or state.approval_status != "pending":
                raise ValueError("run is not waiting for approval")
            return _transition(
                state,
                status="running",
                approval_status="approved",
                approval_reason=None,
            )

        return self._execute(self._store.update(run_id, mutation), resume_mode=True)

    def reject(self, run_id: str, *, reason: str = "") -> RuntimeOutcome:
        """Reject a pending plan without starting its research effects."""

        def mutation(state: RunState) -> RunState:
            if state.status != "waiting_for_human" or state.approval_status != "pending":
                raise ValueError("run is not waiting for approval")
            return _transition(
                state,
                status="cancelled",
                approval_status="rejected",
                approval_reason=reason.strip() or None,
                termination_reason="approval_rejected",
            )

        return RuntimeOutcome(state=self._store.update(run_id, mutation))

    def request_cancel(self, run_id: str, *, reason: str = "") -> RunState:
        """Request cooperative cancellation, or cancel an idle run immediately."""

        def mutation(state: RunState) -> RunState:
            if state.status in {"completed", "cancelled"}:
                return state
            termination_reason = reason.strip() or "cancel_requested"
            if state.status == "running":
                return _transition(
                    state,
                    cancel_requested=True,
                    termination_reason=termination_reason,
                )
            return _transition(
                state,
                status="cancelled",
                cancel_requested=True,
                termination_reason=termination_reason,
            )

        return self._store.update(run_id, mutation)

    def _execute(
        self,
        state: RunState,
        *,
        resume_mode: bool = False,
        retry_ambiguous: bool = False,
    ) -> RuntimeOutcome:
        owner_id = f"executor-{uuid4().hex}"
        self._store.acquire_lease(state.run_id, owner_id, self._lease_ttl_seconds)
        heartbeat = _LeaseHeartbeat(
            self._store,
            state.run_id,
            owner_id,
            self._lease_ttl_seconds,
        )
        heartbeat.start()
        try:
            outcome = self._execute_with_lease(
                state,
                owner_id,
                resume_mode=resume_mode,
                retry_ambiguous=retry_ambiguous,
            )
            heartbeat.raise_if_failed()
            return outcome
        finally:
            heartbeat.stop()
            self._store.release_lease(state.run_id, owner_id)

    def _execute_with_lease(
        self,
        state: RunState,
        owner_id: str,
        *,
        resume_mode: bool,
        retry_ambiguous: bool,
    ) -> RuntimeOutcome:
        state = self._store.update_owned(
            state.run_id,
            owner_id,
            self._lease_ttl_seconds,
            lambda latest: latest,
        )
        if state.status == "completed":
            return RuntimeOutcome(state=state, result=state.result)
        if state.status == "cancelled":
            return RuntimeOutcome(state=state)
        if resume_mode:
            has_ambiguous_effect = any(
                effect.status == "started" for effect in state.effects
            )
            if has_ambiguous_effect and not retry_ambiguous:
                waiting = self._store.update_owned(
                    state.run_id,
                    owner_id,
                    self._lease_ttl_seconds,
                    _ambiguous_state,
                )
                return RuntimeOutcome(state=waiting)
            if (
                state.status == "waiting_for_human"
                and state.current_step != "ambiguous_effect"
            ):
                return RuntimeOutcome(state=state)
        engine_fingerprint = self._current_engine_fingerprint()
        if state.engine_fingerprint and state.engine_fingerprint != engine_fingerprint:
            raise ValueError(
                "runtime configuration does not match the checkpoint; "
                "use the original model and agent configuration"
            )
        state = self._store.update_owned(
            state.run_id,
            owner_id,
            self._lease_ttl_seconds,
            _starting_state,
        )
        if state.status in {"completed", "cancelled"}:
            return RuntimeOutcome(state=state, result=state.result)
        journal = _EffectJournal(
            self._store,
            state,
            retry_policy=self._retry_policy,
            lease_owner=owner_id,
            lease_ttl_seconds=self._lease_ttl_seconds,
            sleeper=self._sleeper,
            random_source=self._random_source,
        )
        planner = (
            _JournaledPlanner(self._planner, journal) if self._planner is not None else None
        )
        report_agent = (
            _JournaledReportAgent(self._report_agent, journal)
            if self._report_agent is not None
            else None
        )
        try:
            result = run_research(
                state.request,
                runner=_JournaledRunner(self._runner, journal),
                planner=planner,
                report_agent=report_agent,
            )
        except _WaitingForApproval:
            return RuntimeOutcome(state=journal.state)
        except _RunCancelled:
            return RuntimeOutcome(state=journal.state)
        except Exception as error:  # noqa: BLE001 - persist arbitrary adapter failures.
            caught_error = error
            failed = self._store.update_owned(
                state.run_id,
                owner_id,
                self._lease_ttl_seconds,
                lambda latest: _failure_state(latest, caught_error),
            )
            return RuntimeOutcome(state=failed)

        completed = self._store.update_owned(
            state.run_id,
            owner_id,
            self._lease_ttl_seconds,
            lambda latest: _completion_state(latest, result),
        )
        return RuntimeOutcome(state=completed, result=completed.result)

    def _current_engine_fingerprint(self) -> str:
        if self._runner is None:
            raise RuntimeError("runner is required to start or resume research")
        return _engine_fingerprint(
            runner=self._runner,
            planner=self._planner,
            report_agent=self._report_agent,
            retry_policy=self._retry_policy,
        )


class _LeaseHeartbeat:
    def __init__(
        self,
        store: CheckpointStore,
        run_id: str,
        owner_id: str,
        ttl_seconds: float,
    ) -> None:
        self._store = store
        self._run_id = run_id
        self._owner_id = owner_id
        self._ttl_seconds = ttl_seconds
        self._stopped = Event()
        self._error: Exception | None = None
        self._thread = Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        self._thread.join()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"execution lease heartbeat failed: {self._run_id}") from self._error

    def _run(self) -> None:
        interval = max(0.01, self._ttl_seconds / 3)
        while not self._stopped.wait(interval):
            try:
                self._store.renew_lease(
                    self._run_id,
                    self._owner_id,
                    self._ttl_seconds,
                )
            except Exception as error:  # noqa: BLE001 - report background lease loss.
                self._error = error
                return


def _ambiguous_state(state: RunState) -> RunState:
    if state.status in {"completed", "cancelled"}:
        return state
    return _transition(
        state,
        status="waiting_for_human",
        current_step="ambiguous_effect",
        error_type="AmbiguousEffect",
        error_message=(
            "A provider call may have completed before its result was saved. "
            "Retry only after confirming the original process stopped."
        ),
    )


def _starting_state(state: RunState) -> RunState:
    if state.status in {"completed", "cancelled"}:
        return state
    if state.cancel_requested:
        return _cancelled_state(state)
    return _transition(
        state,
        status="running",
        current_step="starting",
        error_type=None,
        error_message=None,
    )


def _failure_state(state: RunState, error: Exception) -> RunState:
    if state.cancel_requested or state.status == "cancelled":
        return _cancelled_state(state)
    return _transition(
        state,
        status="failed",
        error_type=type(error).__name__,
        error_message=str(error),
    )


def _completion_state(state: RunState, result: ResearchResult) -> RunState:
    if state.cancel_requested or state.status == "cancelled":
        return _cancelled_state(state)
    return _transition(
        state,
        status="completed",
        current_step="completed",
        result=result,
        error_type=None,
        error_message=None,
    )

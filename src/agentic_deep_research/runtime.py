"""Durable execution for long-running research workflows."""

import hashlib
import json
import os
import random
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock, RLock
from types import UnionType
from typing import Any, Protocol, TypeVar, Union, cast, get_args, get_origin, get_type_hints
from uuid import uuid4

from openai import APIConnectionError, APIStatusError

from .models import (
    AgentRun,
    CitationClaim,
    CitationVerification,
    Evidence,
    EvidenceConflict,
    PlanningRun,
    ReportCritique,
    ReportDraft,
    ResearchBudget,
    ResearchFinding,
    ResearchRequest,
    ResearchResult,
    Source,
)
from .planning import ResearchPlanner
from .reporting import ReportAgent
from .runner import AgentRunner
from .workflow import run_research

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_RUN_STATUSES = {
    "created",
    "running",
    "waiting_for_human",
    "completed",
    "failed",
    "cancelled",
}
_APPROVAL_STATUSES = {"not_requested", "pending", "approved", "rejected"}
_RESULT_TYPES = {
    "AgentRun": AgentRun,
    "PlanningRun": PlanningRun,
    "ReportDraft": ReportDraft,
    "ReportCritique": ReportCritique,
    "CitationVerification": CitationVerification,
}
_T = TypeVar("_T")


class _WaitingForApproval(Exception):
    pass


class _RunCancelled(Exception):
    pass


@dataclass(frozen=True)
class EffectRecord:
    """One durable record of a provider-facing operation."""

    effect_id: str
    kind: str
    input_hash: str
    status: str
    attempts: int
    result_type: str
    result: object | None = None
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"started", "completed", "failed"}:
            raise ValueError(f"unsupported effect status: {self.status}")
        if self.result_type not in _RESULT_TYPES:
            raise ValueError(f"unsupported effect result type: {self.result_type}")
        if self.attempts < 1:
            raise ValueError("effect attempts must be at least 1")


@dataclass(frozen=True)
class RunState:
    """Serializable control state for one durable research run."""

    run_id: str
    request: ResearchRequest
    status: str = "created"
    current_step: str = "created"
    completed_steps: tuple[str, ...] = ()
    effects: tuple[EffectRecord, ...] = ()
    result: ResearchResult | None = None
    requires_approval: bool = False
    approval_status: str = "not_requested"
    approval_reason: str | None = None
    cancel_requested: bool = False
    termination_reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    state_version: int = 0
    schema_version: int = 1
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        _validate_run_id(self.run_id)
        if self.status not in _RUN_STATUSES:
            raise ValueError(f"unsupported run status: {self.status}")
        if self.schema_version != 1:
            raise ValueError(f"unsupported checkpoint schema: {self.schema_version}")
        if self.approval_status not in _APPROVAL_STATUSES:
            raise ValueError(f"unsupported approval status: {self.approval_status}")
        if self.state_version < 0:
            raise ValueError("state_version must not be negative")

    @classmethod
    def create(
        cls,
        request: ResearchRequest,
        *,
        run_id: str | None = None,
    ) -> "RunState":
        """Create a new durable run identity without starting external work."""
        timestamp = _timestamp()
        return cls(
            run_id=run_id or f"run-{uuid4().hex}",
            request=request,
            created_at=timestamp,
            updated_at=timestamp,
        )


@dataclass(frozen=True)
class RuntimeOutcome:
    """Latest durable state plus a result when the run completed."""

    state: RunState
    result: ResearchResult | None = None


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential-backoff policy for transient provider failures."""

    max_attempts: int = 3
    initial_delay_seconds: float = 0.5
    max_delay_seconds: float = 8.0
    max_retry_after_seconds: float = 60.0
    jitter_ratio: float = 0.25

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_delay_seconds < 0:
            raise ValueError("initial_delay_seconds must not be negative")
        if self.max_delay_seconds < 0:
            raise ValueError("max_delay_seconds must not be negative")
        if self.max_retry_after_seconds < 0:
            raise ValueError("max_retry_after_seconds must not be negative")
        if not 0 <= self.jitter_ratio <= 1:
            raise ValueError("jitter_ratio must be between 0 and 1")


class CheckpointStore(Protocol):
    """Persist and restore versioned research run state."""

    def save(self, state: RunState) -> None:
        """Atomically persist the latest state for a run."""
        ...

    def load(self, run_id: str) -> RunState:
        """Load the latest state for a run."""
        ...

    def update(
        self,
        run_id: str,
        mutation: Callable[[RunState], RunState],
    ) -> RunState:
        """Atomically mutate the latest state for a run."""
        ...


class JsonCheckpointStore:
    """Store one atomic, human-inspectable JSON checkpoint per run."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._lock = RLock()

    def save(self, state: RunState) -> None:
        """Write a checkpoint through a same-directory temporary file."""
        with self._lock:
            self._root.mkdir(parents=True, exist_ok=True)
            destination = self._path(state.run_id)
            temporary = self._root / f".{state.run_id}.{uuid4().hex}.tmp"
            try:
                with temporary.open("x", encoding="utf-8") as output:
                    json.dump(_state_to_dict(state), output, ensure_ascii=False, indent=2)
                    output.write("\n")
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)

    def load(self, run_id: str) -> RunState:
        """Restore a checkpoint and reject unknown schema versions."""
        with self._lock:
            path = self._path(run_id)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError as error:
                raise FileNotFoundError(f"checkpoint not found: {run_id}") from error
            if not isinstance(payload, dict):
                raise TypeError("checkpoint must contain a JSON object")
            return _state_from_dict(payload)

    def update(
        self,
        run_id: str,
        mutation: Callable[[RunState], RunState],
    ) -> RunState:
        """Apply one process-safe read-modify-write operation."""
        with self._lock:
            updated = mutation(self.load(run_id))
            self.save(updated)
            return updated

    def _path(self, run_id: str) -> Path:
        _validate_run_id(run_id)
        return self._root / f"{run_id}.json"


class ResearchRuntime:
    """Run the existing workflow with durable provider-call checkpoints."""

    def __init__(
        self,
        *,
        store: CheckpointStore,
        runner: AgentRunner,
        planner: ResearchPlanner | None = None,
        report_agent: ReportAgent | None = None,
        retry_policy: RetryPolicy | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        random_source: Callable[[], float] = random.random,
    ) -> None:
        self._store = store
        self._runner = runner
        self._planner = planner
        self._report_agent = report_agent
        self._retry_policy = retry_policy or RetryPolicy()
        self._sleeper = sleeper
        self._random_source = random_source

    def start(
        self,
        request: ResearchRequest,
        *,
        run_id: str | None = None,
        require_approval: bool = False,
    ) -> RuntimeOutcome:
        """Create and execute a new durable research run."""
        state = replace(
            RunState.create(request, run_id=run_id),
            requires_approval=require_approval,
        )
        try:
            self._store.load(state.run_id)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"checkpoint already exists: {state.run_id}")
        self._store.save(state)
        return self._execute(state)

    def resume(self, run_id: str) -> RuntimeOutcome:
        """Resume a run, replaying completed effects from its checkpoint."""
        state = self._store.load(run_id)
        if state.status == "completed":
            return RuntimeOutcome(state=state, result=state.result)
        if state.status in {"cancelled", "waiting_for_human"}:
            return RuntimeOutcome(state=state)
        return self._execute(state)

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

        return self._execute(self._store.update(run_id, mutation))

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

    def _execute(self, state: RunState) -> RuntimeOutcome:
        state = self._store.update(
            state.run_id,
            lambda latest: _transition(
                latest,
                status="running",
                current_step="starting",
                error_type=None,
                error_message=None,
            ),
        )
        journal = _EffectJournal(
            self._store,
            state,
            retry_policy=self._retry_policy,
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
            failed = self._store.update(
                state.run_id,
                lambda latest: _failure_state(latest, caught_error),
            )
            return RuntimeOutcome(state=failed)

        completed = self._store.update(
            state.run_id,
            lambda latest: _completion_state(latest, result),
        )
        return RuntimeOutcome(state=completed, result=completed.result)


class _EffectJournal:
    def __init__(
        self,
        store: CheckpointStore,
        state: RunState,
        *,
        retry_policy: RetryPolicy,
        sleeper: Callable[[float], None],
        random_source: Callable[[], float],
    ) -> None:
        self._store = store
        self._state = state
        self._retry_policy = retry_policy
        self._sleeper = sleeper
        self._random_source = random_source
        self._lock = Lock()

    @property
    def state(self) -> RunState:
        with self._lock:
            self._state = self._store.load(self._state.run_id)
            return self._state

    def invoke(
        self,
        *,
        kind: str,
        component: object,
        inputs: object,
        result_type: str,
        call: Callable[[], _T],
    ) -> _T:
        self._check_control(kind)
        effect_id, input_hash = _effect_identity(
            run_id=self._state.run_id,
            kind=kind,
            component=component,
            inputs=inputs,
        )
        with self._lock:
            existing = _find_effect(self._state.effects, effect_id)
            if existing is not None and existing.status == "completed":
                return cast(_T, _decode_result(existing.result_type, existing.result))
            completed_attempts = 0 if existing is None else existing.attempts

        for attempt_offset in range(self._retry_policy.max_attempts):
            started = EffectRecord(
                effect_id=effect_id,
                kind=kind,
                input_hash=input_hash,
                status="started",
                attempts=completed_attempts + attempt_offset + 1,
                result_type=result_type,
            )
            with self._lock:
                self._save_effect(started)

            try:
                result = call()
            except Exception as error:
                failed = replace(
                    started,
                    status="failed",
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
                with self._lock:
                    self._save_effect(failed)
                delay = _retry_delay(
                    error,
                    failed_attempt_number=attempt_offset + 1,
                    policy=self._retry_policy,
                    random_source=self._random_source,
                )
                if delay is None or attempt_offset + 1 >= self._retry_policy.max_attempts:
                    raise
                self._check_control(kind, enforce_approval=False)
                self._sleeper(delay)
                self._check_control(kind, enforce_approval=False)
                continue

            completed = replace(
                started,
                status="completed",
                result=_to_jsonable(result),
            )
            with self._lock:
                self._save_effect(completed)
            self._check_control(kind, enforce_approval=False)
            return result

        raise AssertionError("retry loop exhausted without returning or raising")

    def _save_effect(self, effect: EffectRecord) -> None:
        def mutation(state: RunState) -> RunState:
            effects = _replace_effect(state.effects, effect)
            completed_steps = state.completed_steps
            if effect.status == "completed" and effect.effect_id not in completed_steps:
                completed_steps = (*completed_steps, effect.effect_id)
            return _transition(
                state,
                current_step=effect.kind,
                effects=effects,
                completed_steps=completed_steps,
            )

        self._state = self._store.update(
            self._state.run_id,
            mutation,
        )

    def _check_control(self, kind: str, *, enforce_approval: bool = True) -> None:
        with self._lock:
            state = self._store.load(self._state.run_id)
            if state.cancel_requested or state.status == "cancelled":
                self._state = self._store.update(state.run_id, _cancelled_state)
                raise _RunCancelled
            if not enforce_approval or kind != "runner.run" or not state.requires_approval:
                self._state = state
                return
            if state.approval_status == "approved":
                self._state = state
                return
            if state.approval_status == "rejected":
                self._state = self._store.update(state.run_id, _cancelled_state)
                raise _RunCancelled
            if state.approval_status == "not_requested":
                state = self._store.update(
                    state.run_id,
                    lambda latest: _transition(
                        latest,
                        status="waiting_for_human",
                        approval_status="pending",
                        approval_reason="Review the generated plan before research begins.",
                        current_step="approval",
                    ),
                )
            self._state = state
            raise _WaitingForApproval


class _JournaledPlanner:
    def __init__(self, inner: ResearchPlanner, journal: _EffectJournal) -> None:
        self._inner = inner
        self._journal = journal

    def plan(
        self,
        *,
        request: ResearchRequest,
        context: str,
        completed_questions: tuple[str, ...],
        max_questions: int,
        revision: int,
    ) -> PlanningRun:
        inputs = {
            "request": request,
            "context": context,
            "completed_questions": completed_questions,
            "max_questions": max_questions,
            "revision": revision,
        }
        return self._journal.invoke(
            kind="planner.plan",
            component=self._inner,
            inputs=inputs,
            result_type="PlanningRun",
            call=lambda: self._inner.plan(
                request=request,
                context=context,
                completed_questions=completed_questions,
                max_questions=max_questions,
                revision=revision,
            ),
        )


class _JournaledRunner:
    def __init__(self, inner: AgentRunner, journal: _EffectJournal) -> None:
        self._inner = inner
        self._journal = journal

    def run(
        self,
        *,
        instructions: str,
        task: str,
        budget: ResearchBudget,
    ) -> AgentRun:
        inputs = {"instructions": instructions, "task": task, "budget": budget}
        return self._journal.invoke(
            kind="runner.run",
            component=self._inner,
            inputs=inputs,
            result_type="AgentRun",
            call=lambda: self._inner.run(
                instructions=instructions,
                task=task,
                budget=budget,
            ),
        )


class _JournaledReportAgent:
    def __init__(self, inner: ReportAgent, journal: _EffectJournal) -> None:
        self._inner = inner
        self._journal = journal

    def write(
        self,
        *,
        request: ResearchRequest,
        findings: tuple[ResearchFinding, ...],
        evidence: tuple[Evidence, ...],
        conflicts: tuple[EvidenceConflict, ...],
        sources: tuple[Source, ...],
    ) -> ReportDraft:
        return self._journal.invoke(
            kind="report.write",
            component=self._inner,
            inputs={
                "request": request,
                "findings": findings,
                "evidence": evidence,
                "conflicts": conflicts,
                "sources": sources,
            },
            result_type="ReportDraft",
            call=lambda: self._inner.write(
                request=request,
                findings=findings,
                evidence=evidence,
                conflicts=conflicts,
                sources=sources,
            ),
        )

    def critique(
        self,
        *,
        request: ResearchRequest,
        report: str,
        findings: tuple[ResearchFinding, ...],
        evidence: tuple[Evidence, ...],
        conflicts: tuple[EvidenceConflict, ...],
        sources: tuple[Source, ...],
    ) -> ReportCritique:
        return self._journal.invoke(
            kind="report.critique",
            component=self._inner,
            inputs={
                "request": request,
                "report": report,
                "findings": findings,
                "evidence": evidence,
                "conflicts": conflicts,
                "sources": sources,
            },
            result_type="ReportCritique",
            call=lambda: self._inner.critique(
                request=request,
                report=report,
                findings=findings,
                evidence=evidence,
                conflicts=conflicts,
                sources=sources,
            ),
        )

    def verify(
        self,
        *,
        request: ResearchRequest,
        claims: tuple[CitationClaim, ...],
        max_tool_calls: int,
    ) -> CitationVerification:
        return self._journal.invoke(
            kind="report.verify",
            component=self._inner,
            inputs={
                "request": request,
                "claims": claims,
                "max_tool_calls": max_tool_calls,
            },
            result_type="CitationVerification",
            call=lambda: self._inner.verify(
                request=request,
                claims=claims,
                max_tool_calls=max_tool_calls,
            ),
        )

    def revise(
        self,
        *,
        request: ResearchRequest,
        report: str,
        critique: ReportCritique,
        verification: CitationVerification,
        findings: tuple[ResearchFinding, ...],
        evidence: tuple[Evidence, ...],
        conflicts: tuple[EvidenceConflict, ...],
        sources: tuple[Source, ...],
    ) -> ReportDraft:
        return self._journal.invoke(
            kind="report.revise",
            component=self._inner,
            inputs={
                "request": request,
                "report": report,
                "critique": critique,
                "verification": verification,
                "findings": findings,
                "evidence": evidence,
                "conflicts": conflicts,
                "sources": sources,
            },
            result_type="ReportDraft",
            call=lambda: self._inner.revise(
                request=request,
                report=report,
                critique=critique,
                verification=verification,
                findings=findings,
                evidence=evidence,
                conflicts=conflicts,
                sources=sources,
            ),
        )


def _effect_identity(
    *,
    run_id: str,
    kind: str,
    component: object,
    inputs: object,
) -> tuple[str, str]:
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "kind": kind,
        "component": _component_fingerprint(component),
        "inputs": _to_jsonable(inputs),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    input_hash = hashlib.sha256(encoded).hexdigest()
    return f"{kind}:{input_hash}", input_hash


def _component_fingerprint(component: object) -> dict[str, object]:
    fingerprint: dict[str, object] = {
        "type": f"{type(component).__module__}.{type(component).__qualname__}",
    }
    for attribute in ("_model", "_max_output_tokens"):
        value = getattr(component, attribute, None)
        if isinstance(value, (str, int, float, bool)):
            fingerprint[attribute.removeprefix("_")] = value
    return fingerprint


def _retry_delay(
    error: Exception,
    *,
    failed_attempt_number: int,
    policy: RetryPolicy,
    random_source: Callable[[], float],
) -> float | None:
    if not _is_retryable(error):
        return None
    retry_after = _retry_after_seconds(error)
    if retry_after is not None:
        if 0 < retry_after <= policy.max_retry_after_seconds:
            return retry_after
        return None
    exponential = min(
        policy.initial_delay_seconds * (2 ** (failed_attempt_number - 1)),
        policy.max_delay_seconds,
    )
    random_value = min(1.0, max(0.0, random_source()))
    jitter = 1 + ((2 * random_value - 1) * policy.jitter_ratio)
    return max(0.0, exponential * jitter)


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, (APIConnectionError, ConnectionError, TimeoutError)):
        return True
    if not isinstance(error, APIStatusError):
        return False
    retry_header = _response_header(error, "x-should-retry")
    if retry_header == "true":
        return True
    if retry_header == "false":
        return False
    return error.status_code in {408, 409, 429} or error.status_code >= 500


def _retry_after_seconds(error: Exception) -> float | None:
    milliseconds = _response_header(error, "retry-after-ms")
    if milliseconds is not None:
        try:
            return float(milliseconds) / 1_000
        except ValueError:
            pass
    seconds = _response_header(error, "retry-after")
    if seconds is not None:
        try:
            return float(seconds)
        except ValueError:
            pass
    return None


def _response_header(error: Exception, name: str) -> str | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get(name)
    return str(value).strip().lower() if value is not None else None


def _find_effect(
    effects: tuple[EffectRecord, ...],
    effect_id: str,
) -> EffectRecord | None:
    return next((effect for effect in effects if effect.effect_id == effect_id), None)


def _replace_effect(
    effects: tuple[EffectRecord, ...],
    replacement: EffectRecord,
) -> tuple[EffectRecord, ...]:
    if not any(effect.effect_id == replacement.effect_id for effect in effects):
        return (*effects, replacement)
    return tuple(
        replacement if effect.effect_id == replacement.effect_id else effect
        for effect in effects
    )


def _transition(state: RunState, **changes: object) -> RunState:
    return replace(
        state,
        **changes,
        state_version=state.state_version + 1,
        updated_at=_timestamp(),
    )


def _cancelled_state(state: RunState) -> RunState:
    if state.status == "cancelled":
        return state
    return _transition(
        state,
        status="cancelled",
        cancel_requested=True,
        termination_reason=state.termination_reason or "cancel_requested",
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


def _state_to_dict(state: RunState) -> dict[str, object]:
    payload = asdict(state)
    payload["completed_steps"] = list(state.completed_steps)
    payload["effects"] = [asdict(effect) for effect in state.effects]
    return payload


def _state_from_dict(payload: dict[str, object]) -> RunState:
    schema_version = payload.get("schema_version")
    if schema_version != 1:
        raise ValueError(f"unsupported checkpoint schema: {schema_version}")
    request = _decode_dataclass(ResearchRequest, payload.get("request"))
    raw_completed_steps = payload.get("completed_steps", [])
    if not isinstance(raw_completed_steps, list):
        raise TypeError("checkpoint completed_steps must be a JSON array")
    raw_effects = payload.get("effects", [])
    if not isinstance(raw_effects, list):
        raise TypeError("checkpoint effects must be a JSON array")
    effects = tuple(_decode_dataclass(EffectRecord, item) for item in raw_effects)
    return RunState(
        run_id=str(payload.get("run_id", "")),
        request=request,
        status=str(payload.get("status", "")),
        current_step=str(payload.get("current_step", "")),
        completed_steps=tuple(str(item) for item in raw_completed_steps),
        effects=effects,
        result=(
            None
            if payload.get("result") is None
            else _decode_dataclass(ResearchResult, payload.get("result"))
        ),
        requires_approval=bool(payload.get("requires_approval", False)),
        approval_status=str(payload.get("approval_status", "not_requested")),
        approval_reason=_optional_string(payload.get("approval_reason")),
        cancel_requested=bool(payload.get("cancel_requested", False)),
        termination_reason=_optional_string(payload.get("termination_reason")),
        error_type=_optional_string(payload.get("error_type")),
        error_message=_optional_string(payload.get("error_message")),
        state_version=int(payload.get("state_version", 0)),
        schema_version=int(schema_version),
        created_at=str(payload.get("created_at", "")),
        updated_at=str(payload.get("updated_at", "")),
    )


def _decode_result(result_type: str, payload: object) -> object:
    try:
        model = _RESULT_TYPES[result_type]
    except KeyError as error:
        raise ValueError(f"unsupported effect result type: {result_type}") from error
    return _decode_dataclass(model, payload)


def _decode_dataclass[T](model: type[T], payload: object) -> T:
    if not isinstance(payload, dict):
        raise TypeError(f"{model.__name__} checkpoint value must be a JSON object")
    hints = get_type_hints(model)
    values = {
        item.name: _decode_value(payload[item.name], hints[item.name])
        for item in fields(model)
        if item.name in payload
    }
    return model(**values)


def _decode_value(value: object, annotation: object) -> object:
    if value is None:
        return None
    origin = get_origin(annotation)
    if origin is tuple:
        if not isinstance(value, list):
            raise TypeError("tuple checkpoint value must be a JSON array")
        arguments = get_args(annotation)
        item_type = arguments[0] if arguments else Any
        return tuple(_decode_value(item, item_type) for item in value)
    if origin in {Union, UnionType}:
        option = next(item for item in get_args(annotation) if item is not type(None))
        return _decode_value(value, option)
    if isinstance(annotation, type) and is_dataclass(annotation):
        return _decode_dataclass(annotation, value)
    return value


def _to_jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _to_jsonable(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported checkpoint value: {type(value).__name__}")


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _validate_run_id(run_id: str) -> None:
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must contain only letters, digits, hyphens, or underscores")

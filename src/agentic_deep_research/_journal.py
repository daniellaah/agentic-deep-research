"""Internal provider-call journal and retry adapters."""

import hashlib
import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import Lock
from typing import TypeVar, cast

from openai import APIConnectionError, APIStatusError

from .checkpoint import (
    CheckpointStore,
    EffectRecord,
    RunState,
    _cancelled_state,
    _decode_result,
    _to_jsonable,
    _transition,
)
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
    Source,
)
from .planning import ResearchPlanner
from .reporting import ReportAgent
from .runner import AgentRunner

_T = TypeVar("_T")


class _WaitingForApproval(Exception):
    pass


class _RunCancelled(Exception):
    pass


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


class _EffectJournal:
    def __init__(
        self,
        store: CheckpointStore,
        state: RunState,
        *,
        retry_policy: RetryPolicy,
        lease_owner: str,
        lease_ttl_seconds: float,
        sleeper: Callable[[float], None] = time.sleep,
        random_source: Callable[[], float] = random.random,
    ) -> None:
        self._store = store
        self._state = state
        self._retry_policy = retry_policy
        self._lease_owner = lease_owner
        self._lease_ttl_seconds = lease_ttl_seconds
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
            remaining_attempts = self._retry_policy.max_attempts - completed_attempts
            if remaining_attempts < 1:
                raise RuntimeError(f"retry limit exhausted for effect: {kind}")

        for attempt_offset in range(remaining_attempts):
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
                    retryable=_is_retryable(error),
                )
                with self._lock:
                    self._save_effect(failed)
                delay = _retry_delay(
                    error,
                    failed_attempt_number=started.attempts,
                    policy=self._retry_policy,
                    random_source=self._random_source,
                )
                if delay is None or attempt_offset + 1 >= remaining_attempts:
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

        self._state = self._store.update_owned(
            self._state.run_id,
            self._lease_owner,
            self._lease_ttl_seconds,
            mutation,
        )

    def _check_control(self, kind: str, *, enforce_approval: bool = True) -> None:
        with self._lock:
            state = self._store.update_owned(
                self._state.run_id,
                self._lease_owner,
                self._lease_ttl_seconds,
                lambda latest: latest,
            )
            if state.cancel_requested or state.status == "cancelled":
                self._state = self._store.update_owned(
                    state.run_id,
                    self._lease_owner,
                    self._lease_ttl_seconds,
                    _cancelled_state,
                )
                raise _RunCancelled
            if not enforce_approval or kind != "runner.run" or not state.requires_approval:
                self._state = state
                return
            if state.approval_status == "approved":
                self._state = state
                return
            if state.approval_status == "rejected":
                self._state = self._store.update_owned(
                    state.run_id,
                    self._lease_owner,
                    self._lease_ttl_seconds,
                    _cancelled_state,
                )
                raise _RunCancelled
            if state.approval_status == "not_requested":
                state = self._store.update_owned(
                    state.run_id,
                    self._lease_owner,
                    self._lease_ttl_seconds,
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
        return self._journal.invoke(
            kind="runner.run",
            component=self._inner,
            inputs={"instructions": instructions, "task": task, "budget": budget},
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
    for attribute in ("_model", "_max_output_tokens", "_corpus_sha256"):
        value = getattr(component, attribute, None)
        if isinstance(value, (str, int, float, bool)):
            fingerprint[attribute.removeprefix("_")] = value
    return fingerprint


def _component_model(component: object) -> str:
    model = getattr(component, "_model", "")
    return model if isinstance(model, str) else ""


def _engine_fingerprint(
    *,
    runner: AgentRunner,
    planner: ResearchPlanner | None,
    report_agent: ReportAgent | None,
    retry_policy: RetryPolicy,
) -> str:
    payload = {
        "workflow_version": 3,
        "runner": _component_fingerprint(runner),
        "planner": None if planner is None else _component_fingerprint(planner),
        "report_agent": (
            None if report_agent is None else _component_fingerprint(report_agent)
        ),
        "retry_policy": {
            "max_attempts": retry_policy.max_attempts,
            "initial_delay_seconds": retry_policy.initial_delay_seconds,
            "max_delay_seconds": retry_policy.max_delay_seconds,
            "max_retry_after_seconds": retry_policy.max_retry_after_seconds,
            "jitter_ratio": retry_policy.jitter_ratio,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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

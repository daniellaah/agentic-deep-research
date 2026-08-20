"""Safe public events projected from durable research state transitions.

The checkpoint contains prompts, model output, evidence, and raw adapter errors.  This
module deliberately projects only a small lifecycle view that is safe to poll or send
over an event stream.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .checkpoint import RunState

EventType = Literal[
    "run.snapshot",
    "run.created",
    "run.running",
    "scope.clarification_required",
    "scope.clarified",
    "effect.started",
    "effect.completed",
    "effect.failed",
    "plan.edited",
    "plan.approved",
    "plan.rejected",
    "approval.required",
    "cancel.requested",
    "run.action_required",
    "run.completed",
    "run.failed",
    "run.cancelled",
    "run.updated",
]

_EVENT_TYPES = {
    "run.snapshot",
    "run.created",
    "run.running",
    "scope.clarification_required",
    "scope.clarified",
    "effect.started",
    "effect.completed",
    "effect.failed",
    "plan.edited",
    "plan.approved",
    "plan.rejected",
    "approval.required",
    "cancel.requested",
    "run.action_required",
    "run.completed",
    "run.failed",
    "run.cancelled",
    "run.updated",
}
_RUN_STATUSES = {
    "created",
    "running",
    "waiting_for_human",
    "completed",
    "failed",
    "cancelled",
}
_RESULT_STATUSES = {"completed", "incomplete", "needs_review"}
_PUBLIC_EFFECT_KINDS = {
    "scope.resolve": "scope.resolve",
    "planner.plan": "planner.plan",
    "runner.run": "runner.run",
    "report.write": "report.write",
    "report.critique": "report.critique",
    "report.verify": "report.verify",
    "report.revise": "report.revise",
}
_PHASE_BY_EFFECT_KIND = {
    "scope.resolve": "clarification",
    "planner.plan": "planning",
    "runner.run": "researching",
    "report.write": "drafting",
    "report.critique": "reviewing",
    "report.verify": "verifying",
    "report.revise": "revising",
}
_PHASES = {
    "created",
    "planning",
    "researching",
    "drafting",
    "reviewing",
    "verifying",
    "revising",
    "clarification",
    "approval",
    "action_required",
    "completed",
    "failed",
    "cancelled",
    "control",
}


@dataclass(frozen=True)
class RunEvent:
    """A bounded, privacy-safe description of one durable state transition."""

    run_id: str
    sequence: int
    state_version: int
    event_type: EventType
    occurred_at: str
    status: str
    phase: str
    history_complete: bool = True
    effect_kind: str | None = None
    effect_status: str | None = None
    attempt: int | None = None
    retryable: bool | None = None
    result_status: str | None = None
    clarification_required: bool | None = None
    plan_revision: int | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if isinstance(self.sequence, bool) or self.sequence < 0:
            raise ValueError("sequence must not be negative")
        if isinstance(self.state_version, bool) or self.state_version < 0:
            raise ValueError("state_version must not be negative")
        if self.event_type not in _EVENT_TYPES:
            raise ValueError(f"unsupported event type: {self.event_type}")
        if self.status not in _RUN_STATUSES:
            raise ValueError(f"unsupported run status: {self.status}")
        if self.phase not in _PHASES:
            raise ValueError(f"unsupported event phase: {self.phase}")
        if self.schema_version != 1:
            raise ValueError(f"unsupported event schema: {self.schema_version}")
        if self.attempt is not None and (
            isinstance(self.attempt, bool) or self.attempt < 1
        ):
            raise ValueError("attempt must be at least 1")
        if self.plan_revision is not None and (
            isinstance(self.plan_revision, bool) or self.plan_revision < 0
        ):
            raise ValueError("plan_revision must not be negative")
        if self.result_status is not None and self.result_status not in _RESULT_STATUSES:
            raise ValueError(f"unsupported result status: {self.result_status}")
        if self.event_type.startswith("effect."):
            expected_status = self.event_type.removeprefix("effect.")
            if self.effect_kind is None or self.effect_status != expected_status:
                raise ValueError("effect events require matching kind and status")
            if self.attempt is None:
                raise ValueError("effect events require an attempt")
        elif any(
            value is not None
            for value in (
                self.effect_kind,
                self.effect_status,
                self.attempt,
                self.retryable,
            )
        ):
            raise ValueError("non-effect events must not contain effect metadata")


def project_event(
    previous: RunState | None,
    current: RunState,
    sequence: int,
    history_complete: bool = True,
) -> RunEvent:
    """Project a checkpoint transition without exposing checkpoint content.

    ``history_complete=False`` is intended for the one safe snapshot inserted when an
    older checkpoint database first gains event support.
    """

    if previous is not None and previous.run_id != current.run_id:
        raise ValueError("event transition must keep the same run_id")

    status = _safe_status(getattr(current, "status", ""))
    occurred_at = str(
        getattr(current, "updated_at", "") or getattr(current, "created_at", "")
    )

    if not history_complete:
        return _event(
            current,
            sequence,
            "run.snapshot",
            status=status,
            phase=_phase_for_state(current),
            occurred_at=occurred_at,
            history_complete=False,
        )
    if previous is None:
        return _event(
            current,
            sequence,
            "run.created",
            status=status,
            phase="created",
            occurred_at=occurred_at,
        )

    plan_control = _new_plan_control(previous, current)
    control_kind = _plan_control_kind(plan_control)
    if control_kind == "edit":
        return _event(
            current,
            sequence,
            "plan.edited",
            status=status,
            phase="approval",
            occurred_at=occurred_at,
            plan_revision=_plan_revision(plan_control, current),
        )
    if control_kind == "approve":
        return _event(
            current,
            sequence,
            "plan.approved",
            status=status,
            phase="approval",
            occurred_at=occurred_at,
            plan_revision=_plan_revision(plan_control, current),
        )

    previous_approval = str(getattr(previous, "approval_status", ""))
    current_approval = str(getattr(current, "approval_status", ""))
    if current_approval != previous_approval:
        if current_approval == "rejected":
            return _event(
                current,
                sequence,
                "plan.rejected",
                status=status,
                phase="approval",
                occurred_at=occurred_at,
                plan_revision=_plan_revision(None, current),
            )
        if current_approval == "approved":
            return _event(
                current,
                sequence,
                "plan.approved",
                status=status,
                phase="approval",
                occurred_at=occurred_at,
                plan_revision=_plan_revision(None, current),
            )

    clarification = _clarification_change(previous, current)
    if clarification is True:
        return _event(
            current,
            sequence,
            "scope.clarification_required",
            status=status,
            phase="clarification",
            occurred_at=occurred_at,
            clarification_required=True,
        )
    if clarification is False:
        return _event(
            current,
            sequence,
            "scope.clarified",
            status=status,
            phase="clarification",
            occurred_at=occurred_at,
            clarification_required=False,
        )

    if _is_ambiguous_action_required(current):
        return _event(
            current,
            sequence,
            "run.action_required",
            status=status,
            phase="action_required",
            occurred_at=occurred_at,
        )

    if current_approval == "pending" and current_approval != previous_approval:
        return _event(
            current,
            sequence,
            "approval.required",
            status=status,
            phase="approval",
            occurred_at=occurred_at,
            plan_revision=_plan_revision(None, current),
        )

    if (
        bool(getattr(current, "cancel_requested", False))
        and not bool(getattr(previous, "cancel_requested", False))
        and status != "cancelled"
    ):
        return _event(
            current,
            sequence,
            "cancel.requested",
            status=status,
            phase="control",
            occurred_at=occurred_at,
        )

    changed_effect = _changed_effect(previous, current)
    if changed_effect is not None:
        effect_status = str(getattr(changed_effect, "status", ""))
        if effect_status in {"started", "completed", "failed"}:
            public_kind = _public_effect_kind(changed_effect)
            return _event(
                current,
                sequence,
                f"effect.{effect_status}",  # type: ignore[arg-type]
                status=status,
                phase=_PHASE_BY_EFFECT_KIND.get(public_kind, "control"),
                occurred_at=occurred_at,
                effect_kind=public_kind,
                effect_status=effect_status,
                attempt=int(getattr(changed_effect, "attempts", 0)),
                retryable=(
                    bool(getattr(changed_effect, "retryable", False))
                    if effect_status == "failed"
                    else False
                ),
            )

    if status == "completed" and status != getattr(previous, "status", ""):
        result = getattr(current, "result", None)
        raw_result_status = getattr(result, "status", None)
        result_status = (
            str(raw_result_status) if raw_result_status in _RESULT_STATUSES else None
        )
        return _event(
            current,
            sequence,
            "run.completed",
            status=status,
            phase="completed",
            occurred_at=occurred_at,
            result_status=result_status,
        )
    if status == "failed" and status != getattr(previous, "status", ""):
        return _event(
            current,
            sequence,
            "run.failed",
            status=status,
            phase="failed",
            occurred_at=occurred_at,
        )
    if status == "cancelled" and status != getattr(previous, "status", ""):
        return _event(
            current,
            sequence,
            "run.cancelled",
            status=status,
            phase="cancelled",
            occurred_at=occurred_at,
        )
    if status == "running" and status != getattr(previous, "status", ""):
        return _event(
            current,
            sequence,
            "run.running",
            status=status,
            phase=_phase_for_state(current),
            occurred_at=occurred_at,
        )
    return _event(
        current,
        sequence,
        "run.updated",
        status=status,
        phase=_phase_for_state(current),
        occurred_at=occurred_at,
    )


def event_to_dict(event: RunEvent) -> dict[str, object]:
    """Return the versioned JSON-compatible public event projection."""

    data: dict[str, object] = {
        "status": event.status,
        "phase": event.phase,
    }
    optional = {
        "effect_kind": event.effect_kind,
        "effect_status": event.effect_status,
        "attempt": event.attempt,
        "retryable": event.retryable,
        "result_status": event.result_status,
        "clarification_required": event.clarification_required,
        "plan_revision": event.plan_revision,
    }
    data.update({key: value for key, value in optional.items() if value is not None})
    return {
        "schema_version": event.schema_version,
        "run_id": event.run_id,
        "sequence": event.sequence,
        "state_version": event.state_version,
        "type": event.event_type,
        "occurred_at": event.occurred_at,
        "history_complete": event.history_complete,
        "data": data,
    }


def event_to_json(event: RunEvent) -> str:
    """Serialize one event deterministically for storage, polling, or SSE data."""

    return json.dumps(
        event_to_dict(event),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def event_from_json(payload: str) -> RunEvent:
    """Restore one validated public event from durable JSON storage."""
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise TypeError("event payload must be a JSON object")
    data = decoded.get("data")
    if not isinstance(data, dict):
        raise TypeError("event data must be a JSON object")
    history_complete = decoded.get("history_complete")
    if not isinstance(history_complete, bool):
        raise TypeError("event history_complete must be a boolean")
    retryable = data.get("retryable")
    if retryable is not None and not isinstance(retryable, bool):
        raise TypeError("event retryable must be a boolean")
    clarification_required = data.get("clarification_required")
    if clarification_required is not None and not isinstance(
        clarification_required,
        bool,
    ):
        raise TypeError("event clarification_required must be a boolean")
    return RunEvent(
        run_id=str(decoded.get("run_id", "")),
        sequence=_strict_int(decoded.get("sequence"), "sequence"),
        state_version=_strict_int(decoded.get("state_version"), "state_version"),
        event_type=str(decoded.get("type", "")),  # type: ignore[arg-type]
        occurred_at=str(decoded.get("occurred_at", "")),
        status=str(data.get("status", "")),
        phase=str(data.get("phase", "")),
        history_complete=history_complete,
        effect_kind=_optional_string(data.get("effect_kind")),
        effect_status=_optional_string(data.get("effect_status")),
        attempt=_optional_int(data.get("attempt"), "attempt"),
        retryable=retryable,
        result_status=_optional_string(data.get("result_status")),
        clarification_required=clarification_required,
        plan_revision=_optional_int(data.get("plan_revision"), "plan_revision"),
        schema_version=_strict_int(decoded.get("schema_version"), "schema_version"),
    )


def _strict_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"event {name} must be an integer")
    return value


def _optional_int(value: object, name: str) -> int | None:
    return None if value is None else _strict_int(value, name)


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)


def _event(
    current: RunState,
    sequence: int,
    event_type: EventType,
    *,
    status: str,
    phase: str,
    occurred_at: str,
    history_complete: bool = True,
    effect_kind: str | None = None,
    effect_status: str | None = None,
    attempt: int | None = None,
    retryable: bool | None = None,
    result_status: str | None = None,
    clarification_required: bool | None = None,
    plan_revision: int | None = None,
) -> RunEvent:
    return RunEvent(
        run_id=str(getattr(current, "run_id", "")),
        sequence=sequence,
        state_version=int(getattr(current, "state_version", 0)),
        event_type=event_type,
        occurred_at=occurred_at,
        status=status,
        phase=phase,
        history_complete=history_complete,
        effect_kind=effect_kind,
        effect_status=effect_status,
        attempt=attempt,
        retryable=retryable,
        result_status=result_status,
        clarification_required=clarification_required,
        plan_revision=plan_revision,
    )


def _safe_status(value: object) -> str:
    status = str(value)
    if status not in _RUN_STATUSES:
        raise ValueError(f"unsupported run status: {status}")
    return status


def _changed_effect(previous: RunState, current: RunState) -> object | None:
    previous_effects = {
        str(getattr(effect, "effect_id", "")): effect
        for effect in getattr(previous, "effects", ())
    }
    for effect in reversed(tuple(getattr(current, "effects", ()))):
        effect_id = str(getattr(effect, "effect_id", ""))
        prior = previous_effects.get(effect_id)
        if prior is None or (
            getattr(prior, "status", None),
            getattr(prior, "attempts", None),
        ) != (
            getattr(effect, "status", None),
            getattr(effect, "attempts", None),
        ):
            return effect
    return None


def _public_effect_kind(effect: object) -> str:
    kind = str(getattr(effect, "kind", ""))
    return _PUBLIC_EFFECT_KINDS.get(kind, "other")


def _new_plan_control(previous: RunState, current: RunState) -> object | None:
    previous_controls = tuple(getattr(previous, "plan_controls", ()) or ())
    current_controls = tuple(getattr(current, "plan_controls", ()) or ())
    if current_controls == previous_controls:
        return None
    previous_ids = {
        str(getattr(control, "control_id", "")) for control in previous_controls
    }
    added = [
        control
        for control in current_controls
        if str(getattr(control, "control_id", "")) not in previous_ids
    ]
    return added[-1] if added else current_controls[-1] if current_controls else None


def _plan_control_kind(control: object | None) -> str | None:
    if control is None:
        return None
    kind = str(getattr(control, "kind", getattr(control, "action", ""))).casefold()
    if kind in {"edit", "edited", "plan_edited"}:
        return "edit"
    if kind in {"approve", "approved", "plan_approved"}:
        return "approve"
    return None


def _plan_revision(control: object | None, current: RunState) -> int | None:
    plan = getattr(control, "plan", None) if control is not None else None
    if plan is None:
        try:
            plan = getattr(current, "plan", None)
        except (TypeError, ValueError):
            return None
    revision = getattr(plan, "revision", None)
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        return None
    return revision


def _clarification_change(previous: RunState, current: RunState) -> bool | None:
    prior = getattr(previous, "clarification", None)
    latest = getattr(current, "clarification", None)
    if latest == prior:
        return None
    prior_required = getattr(prior, "needs_clarification", False) is True
    if latest is None and prior_required:
        return False
    required = getattr(latest, "needs_clarification", None)
    if required is True:
        return True
    if required is False and prior_required:
        return False
    return None


def _is_ambiguous_action_required(state: RunState) -> bool:
    return (
        getattr(state, "status", "") == "waiting_for_human"
        and getattr(state, "current_step", "") == "ambiguous_effect"
    )


def _phase_for_state(state: RunState) -> str:
    status = getattr(state, "status", "")
    if status in {"created", "completed", "failed", "cancelled"}:
        return str(status)
    step = str(getattr(state, "current_step", ""))
    if step == "approval":
        return "approval"
    if step in {"clarification", "scope_clarification"}:
        return "clarification"
    return _PHASE_BY_EFFECT_KIND.get(step, "control")

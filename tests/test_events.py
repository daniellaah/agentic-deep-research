import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agentic_deep_research.checkpoint import EffectRecord, RunState
from agentic_deep_research.events import (
    RunEvent,
    event_from_json,
    event_to_dict,
    event_to_json,
    project_event,
)
from agentic_deep_research.models import (
    ClarificationDecision,
    PlanControlRecord,
    ResearchPlan,
    ResearchQuestion,
    ResearchRequest,
)


def _state(**changes: object) -> RunState:
    state = RunState.create(
        ResearchRequest("private research topic"),
        run_id="run-events",
    )
    return replace(state, **changes)


def _next(previous: RunState, **changes: object) -> RunState:
    return replace(
        previous,
        state_version=previous.state_version + 1,
        updated_at="2026-08-19T12:00:00+00:00",
        **changes,
    )


def _effect(
    status: str,
    *,
    kind: str = "runner.run",
    attempts: int = 1,
) -> EffectRecord:
    return EffectRecord(
        effect_id="effect-secret-id",
        kind=kind,
        input_hash="secret-input-hash",
        status=status,
        attempts=attempts,
        result_type="AgentRun",
        result=(
            {"report": "secret generated report"}
            if status == "completed"
            else None
        ),
        error_type="SecretProviderError" if status == "failed" else None,
        error_message="secret raw provider error" if status == "failed" else None,
        retryable=status == "failed",
    )


def test_created_event_has_a_small_versioned_json_projection() -> None:
    state = _state(engine_fingerprint="secret-engine", model_name="secret-model")

    event = project_event(None, state, sequence=0)

    assert event.event_type == "run.created"
    assert event.phase == "created"
    assert event_to_dict(event) == {
        "schema_version": 1,
        "run_id": "run-events",
        "sequence": 0,
        "state_version": 0,
        "type": "run.created",
        "occurred_at": state.updated_at,
        "history_complete": True,
        "data": {"status": "created", "phase": "created"},
    }
    assert json.loads(event_to_json(event)) == event_to_dict(event)

    serialized = event_to_json(event)
    assert event_from_json(serialized) == event
    assert "private research topic" not in serialized
    assert "secret-engine" not in serialized
    assert "secret-model" not in serialized


def test_incomplete_history_projects_an_honest_snapshot() -> None:
    current = _state(status="running", current_step="runner.run", state_version=7)

    event = project_event(None, current, sequence=0, history_complete=False)

    assert event.event_type == "run.snapshot"
    assert event.history_complete is False
    assert event.state_version == 7
    assert event.phase == "researching"


def test_created_to_running_projects_the_runtime_transition() -> None:
    previous = _state()
    current = _next(previous, status="running", current_step="starting")

    event = project_event(previous, current, sequence=1)

    assert event.event_type == "run.running"
    assert event.status == "running"
    assert event.phase == "control"


@pytest.mark.parametrize("effect_status", ["started", "completed", "failed"])
def test_effect_events_expose_progress_but_not_effect_content(
    effect_status: str,
) -> None:
    previous = _state(status="running", current_step="starting")
    effect = _effect(effect_status)
    current = _next(
        previous,
        status="running",
        current_step="runner.run",
        effects=(effect,),
    )

    event = project_event(previous, current, sequence=2)
    serialized = event_to_json(event)

    assert event.event_type == f"effect.{effect_status}"
    assert event.effect_kind == "runner.run"
    assert event.effect_status == effect_status
    assert event.attempt == 1
    assert event.phase == "researching"
    assert "secret-input-hash" not in serialized
    assert "secret generated report" not in serialized
    assert "SecretProviderError" not in serialized
    assert "secret raw provider error" not in serialized


def test_unknown_effect_kind_is_not_reflected_to_public_clients() -> None:
    previous = _state(status="running")
    current = _next(
        previous,
        status="running",
        effects=(_effect("started", kind="private.secret.component"),),
    )

    event = project_event(previous, current, sequence=1)

    assert event.event_type == "effect.started"
    assert event.effect_kind == "other"
    assert "private.secret.component" not in event_to_json(event)


def test_scoping_effect_has_a_clarification_phase() -> None:
    previous = _state(status="running")
    current = _next(
        previous,
        status="running",
        current_step="scope.resolve",
        effects=(_effect("started", kind="scope.resolve"),),
    )

    event = project_event(previous, current, sequence=1)

    assert event.event_type == "effect.started"
    assert event.effect_kind == "scope.resolve"
    assert event.phase == "clarification"


def test_approval_lifecycle_uses_plan_events_without_exposing_reason() -> None:
    running = _state(status="running")
    pending = _next(
        running,
        status="waiting_for_human",
        current_step="approval",
        approval_status="pending",
        approval_reason="secret review reason",
    )
    approved = _next(
        pending,
        status="running",
        approval_status="approved",
        approval_reason=None,
    )
    rejected = _next(
        pending,
        status="cancelled",
        approval_status="rejected",
        approval_reason="secret rejection reason",
        termination_reason="secret cancellation reason",
    )

    required_event = project_event(running, pending, sequence=2)
    approved_event = project_event(pending, approved, sequence=3)
    rejected_event = project_event(pending, rejected, sequence=3)

    assert required_event.event_type == "approval.required"
    assert approved_event.event_type == "plan.approved"
    assert rejected_event.event_type == "plan.rejected"
    serialized = "".join(
        event_to_json(event)
        for event in (required_event, approved_event, rejected_event)
    )
    assert "secret review reason" not in serialized
    assert "secret rejection reason" not in serialized
    assert "secret cancellation reason" not in serialized


def test_plan_control_records_project_only_kind_and_revision() -> None:
    plan = ResearchPlan(
        objective="private objective",
        questions=(ResearchQuestion(id="q1", question="private question"),),
        revision=4,
    )
    previous = _state(status="waiting_for_human", plan_controls=())
    edited_control = PlanControlRecord(
        control_id=f"plan.edit:{'a' * 64}",
        kind="edit",
        input_hash="a" * 64,
        base_plan_hash="b" * 64,
        plan_hash="c" * 64,
        plan=plan,
        created_at="2026-08-19T12:00:00+00:00",
    )
    edited = _next(
        previous,
        plan_controls=(edited_control,),
    )
    approved_control = PlanControlRecord(
        control_id=f"plan.approve:{'d' * 64}",
        kind="approve",
        input_hash="d" * 64,
        base_plan_hash="c" * 64,
        plan_hash="c" * 64,
        plan=plan,
        created_at="2026-08-19T12:01:00+00:00",
    )
    approved = _next(
        edited,
        plan_controls=(edited_control, approved_control),
    )

    edited_event = project_event(previous, edited, sequence=1)
    approved_event = project_event(edited, approved, sequence=2)

    assert edited_event.event_type == "plan.edited"
    assert edited_event.plan_revision == 4
    assert approved_event.event_type == "plan.approved"
    assert approved_event.plan_revision == 4
    serialized = event_to_json(edited_event) + event_to_json(approved_event)
    assert "private objective" not in serialized
    assert "private question" not in serialized
    assert "a" * 64 not in serialized
    assert "b" * 64 not in serialized
    assert "c" * 64 not in serialized
    assert "d" * 64 not in serialized


def test_scope_clarification_never_projects_reason_or_question() -> None:
    previous = _state(status="created", clarification=None)
    decision = ClarificationDecision(
        needs_clarification=True,
        reason="secret ambiguity analysis",
        question="secret clarification question",
    )
    waiting = _next(
        previous,
        status="waiting_for_human",
        current_step="clarification",
        clarification=decision,
    )
    resolved = _next(
        waiting,
        status="created",
        current_step="clarification_answered",
        clarification=None,
    )

    required_event = project_event(previous, waiting, sequence=1)
    resolved_event = project_event(waiting, resolved, sequence=2)

    assert required_event.event_type == "scope.clarification_required"
    assert required_event.clarification_required is True
    assert resolved_event.event_type == "scope.clarified"
    assert resolved_event.clarification_required is False
    serialized = event_to_json(required_event) + event_to_json(resolved_event)
    assert "secret ambiguity analysis" not in serialized
    assert "secret clarification question" not in serialized


def test_cancel_and_ambiguous_action_transitions_are_distinct() -> None:
    running = _state(status="running")
    cancel_requested = _next(running, cancel_requested=True)
    cancelled = _next(cancel_requested, status="cancelled")
    ambiguous = _next(
        running,
        status="waiting_for_human",
        current_step="ambiguous_effect",
        error_type="AmbiguousEffect",
        error_message="secret ambiguous provider detail",
    )

    requested_event = project_event(running, cancel_requested, sequence=1)
    cancelled_event = project_event(cancel_requested, cancelled, sequence=2)
    action_event = project_event(running, ambiguous, sequence=1)

    assert requested_event.event_type == "cancel.requested"
    assert cancelled_event.event_type == "run.cancelled"
    assert action_event.event_type == "run.action_required"
    assert "secret ambiguous provider detail" not in event_to_json(action_event)


@pytest.mark.parametrize(
    ("status", "expected_type", "expected_phase"),
    [
        ("completed", "run.completed", "completed"),
        ("failed", "run.failed", "failed"),
        ("cancelled", "run.cancelled", "cancelled"),
    ],
)
def test_terminal_run_events(
    status: str,
    expected_type: str,
    expected_phase: str,
) -> None:
    previous = _state(status="running")
    result = SimpleNamespace(status="needs_review", report="secret final report")
    current = _next(
        previous,
        status=status,
        result=result if status == "completed" else None,
        error_message="secret terminal error" if status == "failed" else None,
        termination_reason="secret cancel reason" if status == "cancelled" else None,
    )

    event = project_event(previous, current, sequence=5)

    assert event.event_type == expected_type
    assert event.phase == expected_phase
    assert event.result_status == ("needs_review" if status == "completed" else None)
    serialized = event_to_json(event)
    assert "secret final report" not in serialized
    assert "secret terminal error" not in serialized
    assert "secret cancel reason" not in serialized


def test_cross_run_transition_and_invalid_event_metadata_are_rejected() -> None:
    previous = _state()
    other = replace(previous, run_id="run-other")

    with pytest.raises(ValueError, match="same run_id"):
        project_event(previous, other, sequence=1)
    with pytest.raises(ValueError, match="sequence"):
        project_event(None, previous, sequence=-1)
    with pytest.raises(ValueError, match="effect events require"):
        RunEvent(
            run_id="run-events",
            sequence=0,
            state_version=0,
            event_type="effect.started",
            occurred_at=previous.updated_at,
            status="running",
            phase="researching",
        )

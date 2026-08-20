"""Serializable run state and atomic local checkpoint storage."""

import hashlib
import json
import os
import re
import sqlite3
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from types import UnionType
from typing import Any, Protocol, Union, get_args, get_origin, get_type_hints
from uuid import uuid4

from .events import RunEvent, event_from_json, event_to_json, project_event
from .models import (
    AgentRun,
    CitationVerification,
    ClarificationDecision,
    ConversationMessage,
    PlanControlRecord,
    PlanningRun,
    ReportCritique,
    ReportDraft,
    ResearchBrief,
    ResearchPlan,
    ResearchRequest,
    ResearchResult,
    ScopingRun,
)

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
    "ScopingRun": ScopingRun,
}


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
    retryable: bool = False

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
    engine_fingerprint: str = ""
    model_name: str = ""
    status: str = "created"
    current_step: str = "created"
    completed_steps: tuple[str, ...] = ()
    effects: tuple[EffectRecord, ...] = ()
    conversation: tuple[ConversationMessage, ...] = ()
    clarification: ClarificationDecision | None = None
    plan_controls: tuple[PlanControlRecord, ...] = ()
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

    @property
    def plan(self) -> ResearchPlan | None:
        """Return the latest human edit, otherwise the latest generated plan."""
        for control in reversed(self.plan_controls):
            if control.kind == "edit":
                return control.plan
        for effect in self.effects:
            if effect.status == "completed" and effect.result_type == "PlanningRun":
                planning_run = _decode_result(effect.result_type, effect.result)
                if (
                    isinstance(planning_run, PlanningRun)
                    and planning_run.plan.revision == 0
                ):
                    return planning_run.plan
        return self.result.plan if self.result is not None else None

    @property
    def approved_plan(self) -> ResearchPlan | None:
        """Return the immutable plan snapshot most recently approved by a human."""
        for control in reversed(self.plan_controls):
            if control.kind == "approve":
                return control.plan
        return None

    @property
    def plan_hash(self) -> str | None:
        """Return the concurrency identity of the currently inspectable plan."""
        for control in reversed(self.plan_controls):
            if control.kind == "edit":
                return control.plan_hash
        plan = self.plan
        return None if plan is None else research_plan_hash(plan)

    @property
    def brief(self) -> ResearchBrief | None:
        """Return the single persisted research-brief source of truth."""
        return self.request.brief

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


class CheckpointStore(Protocol):
    """Persist and restore versioned research run state."""

    def create(self, state: RunState) -> None:
        """Persist a new run and reject duplicate identities."""
        ...

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

    def update_owned(
        self,
        run_id: str,
        owner_id: str,
        ttl_seconds: float,
        mutation: Callable[[RunState], RunState],
    ) -> RunState:
        """Mutate state only while the caller owns the execution lease."""
        ...

    def acquire_lease(self, run_id: str, owner_id: str, ttl_seconds: float) -> None:
        """Acquire exclusive execution ownership for a bounded period."""
        ...

    def renew_lease(self, run_id: str, owner_id: str, ttl_seconds: float) -> None:
        """Extend execution ownership or fail if it was lost."""
        ...

    def release_lease(self, run_id: str, owner_id: str) -> None:
        """Release execution ownership held by one runtime."""
        ...


class JsonCheckpointStore:
    """Store one atomic, human-inspectable JSON checkpoint per run."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._lock = RLock()
        self._leases: dict[str, tuple[str, float]] = {}

    def create(self, state: RunState) -> None:
        """Create a checkpoint only when its run ID is unused."""
        with self._lock:
            if self._path(state.run_id).exists():
                raise FileExistsError(f"checkpoint already exists: {state.run_id}")
            self.save(state)

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
                os.chmod(destination, 0o600)
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
        """Apply one thread-safe read-modify-write operation in this process."""
        with self._lock:
            updated = mutation(self.load(run_id))
            self.save(updated)
            return updated

    def update_owned(
        self,
        run_id: str,
        owner_id: str,
        ttl_seconds: float,
        mutation: Callable[[RunState], RunState],
    ) -> RunState:
        """Fence an in-process state update with its active lease."""
        with self._lock:
            current = self._leases.get(run_id)
            now = time.time()
            if current is None or current[0] != owner_id:
                raise RuntimeError(f"execution lease lost: {run_id}")
            updated = mutation(self.load(run_id))
            self.save(updated)
            self._leases[run_id] = (owner_id, now + ttl_seconds)
            return updated

    def acquire_lease(self, run_id: str, owner_id: str, ttl_seconds: float) -> None:
        """Acquire an in-process lease for the learning-oriented JSON store."""
        with self._lock:
            self.load(run_id)
            current = self._leases.get(run_id)
            now = time.time()
            if current is not None and current[0] != owner_id and current[1] > now:
                raise RuntimeError(f"run is already executing: {run_id}")
            self._leases[run_id] = (owner_id, now + ttl_seconds)

    def renew_lease(self, run_id: str, owner_id: str, ttl_seconds: float) -> None:
        """Renew an in-process JSON-store lease."""
        with self._lock:
            current = self._leases.get(run_id)
            if current is None or current[0] != owner_id:
                raise RuntimeError(f"execution lease lost: {run_id}")
            self._leases[run_id] = (owner_id, time.time() + ttl_seconds)

    def release_lease(self, run_id: str, owner_id: str) -> None:
        """Release an in-process JSON-store lease."""
        with self._lock:
            current = self._leases.get(run_id)
            if current is not None and current[0] == owner_id:
                del self._leases[run_id]

    def _path(self, run_id: str) -> Path:
        _validate_run_id(run_id)
        return self._root / f"{run_id}.json"


class SQLiteCheckpointStore:
    """Transactionally store checkpoints for concurrent local processes."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS research_runs (
                    run_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    state_version INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS research_run_leases (
                    run_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS research_run_events (
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    state_version INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, sequence)
                )
                """
            )
            _backfill_event_snapshots(connection)
        os.chmod(self._path, 0o600)

    def create(self, state: RunState) -> None:
        """Atomically insert a new run together with its first public event."""
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    INSERT INTO research_runs (run_id, state_json, state_version)
                    VALUES (?, ?, ?)
                    """,
                    (state.run_id, _serialize_state(state), state.state_version),
                )
            except sqlite3.IntegrityError as error:
                raise FileExistsError(
                    f"checkpoint already exists: {state.run_id}"
                ) from error
            _append_event(connection, None, state)
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def save(self, state: RunState) -> None:
        """Insert a state or commit one valid next-version transition."""
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state_json FROM research_runs WHERE run_id = ?",
                (state.run_id,),
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO research_runs (run_id, state_json, state_version)
                    VALUES (?, ?, ?)
                    """,
                    (state.run_id, _serialize_state(state), state.state_version),
                )
                _append_event(connection, None, state)
            else:
                previous = _deserialize_state(str(row[0]))
                _persist_transition(connection, previous, state)
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def load(self, run_id: str) -> RunState:
        """Load the latest committed state for a run."""
        _validate_run_id(run_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM research_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise FileNotFoundError(f"checkpoint not found: {run_id}")
        return _deserialize_state(str(row[0]))

    def list_events(
        self,
        run_id: str,
        *,
        after_sequence: int = -1,
        limit: int = 100,
    ) -> tuple[RunEvent, ...]:
        """Return durable public events after an exclusive per-run cursor."""
        _validate_run_id(run_id)
        if isinstance(after_sequence, bool) or after_sequence < -1:
            raise ValueError("after_sequence must be at least -1")
        if isinstance(limit, bool) or not 1 <= limit <= 100:
            raise ValueError("event limit must be between 1 and 100")
        with self._connect() as connection:
            if connection.execute(
                "SELECT 1 FROM research_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone() is None:
                raise FileNotFoundError(f"checkpoint not found: {run_id}")
            rows = connection.execute(
                """
                SELECT payload_json
                FROM research_run_events
                WHERE run_id = ? AND sequence > ?
                ORDER BY sequence
                LIMIT ?
                """,
                (run_id, after_sequence, limit),
            ).fetchall()
        return tuple(event_from_json(str(row[0])) for row in rows)

    def update(
        self,
        run_id: str,
        mutation: Callable[[RunState], RunState],
    ) -> RunState:
        """Serialize concurrent read-modify-write operations with a transaction."""
        _validate_run_id(run_id)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state_json FROM research_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise FileNotFoundError(f"checkpoint not found: {run_id}")
            previous = _deserialize_state(str(row[0]))
            updated = mutation(previous)
            _persist_transition(connection, previous, updated)
            connection.commit()
            return updated
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def update_owned(
        self,
        run_id: str,
        owner_id: str,
        ttl_seconds: float,
        mutation: Callable[[RunState], RunState],
    ) -> RunState:
        """Atomically fence, renew, and persist an execution-owned update."""
        _validate_run_id(run_id)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            lease = connection.execute(
                "SELECT owner_id, expires_at FROM research_run_leases WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            now = time.time()
            if (
                lease is None
                or str(lease[0]) != owner_id
            ):
                raise RuntimeError(f"execution lease lost: {run_id}")
            row = connection.execute(
                "SELECT state_json FROM research_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise FileNotFoundError(f"checkpoint not found: {run_id}")
            previous = _deserialize_state(str(row[0]))
            updated = mutation(previous)
            _persist_transition(connection, previous, updated)
            connection.execute(
                """
                UPDATE research_run_leases
                SET expires_at = ?
                WHERE run_id = ? AND owner_id = ?
                """,
                (now + ttl_seconds, run_id, owner_id),
            )
            connection.commit()
            return updated
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def acquire_lease(self, run_id: str, owner_id: str, ttl_seconds: float) -> None:
        """Acquire a cross-process lease in one SQLite transaction."""
        _validate_run_id(run_id)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM research_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone() is None:
                raise FileNotFoundError(f"checkpoint not found: {run_id}")
            current = connection.execute(
                "SELECT owner_id, expires_at FROM research_run_leases WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            now = time.time()
            if current is not None and str(current[0]) != owner_id and float(current[1]) > now:
                raise RuntimeError(f"run is already executing: {run_id}")
            connection.execute(
                """
                INSERT INTO research_run_leases (run_id, owner_id, expires_at)
                VALUES (?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    owner_id = excluded.owner_id,
                    expires_at = excluded.expires_at
                """,
                (run_id, owner_id, now + ttl_seconds),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def renew_lease(self, run_id: str, owner_id: str, ttl_seconds: float) -> None:
        """Renew a cross-process lease while preserving its owner."""
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE research_run_leases
                SET expires_at = ?
                WHERE run_id = ? AND owner_id = ?
                """,
                (time.time() + ttl_seconds, run_id, owner_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"execution lease lost: {run_id}")

    def release_lease(self, run_id: str, owner_id: str) -> None:
        """Release a cross-process lease only when the owner matches."""
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM research_run_leases WHERE run_id = ? AND owner_id = ?",
                (run_id, owner_id),
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=30)
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection


def _persist_transition(
    connection: sqlite3.Connection,
    previous: RunState,
    current: RunState,
) -> None:
    """Commit one state change and its public event inside the caller transaction."""
    if current.run_id != previous.run_id:
        raise ValueError("checkpoint mutation must not change run_id")
    previous_payload = _serialize_state(previous)
    current_payload = _serialize_state(current)
    if current_payload == previous_payload:
        return
    if current.state_version != previous.state_version + 1:
        raise ValueError("checkpoint mutation must increment state_version exactly once")
    connection.execute(
        """
        UPDATE research_runs
        SET state_json = ?, state_version = ?
        WHERE run_id = ?
        """,
        (current_payload, current.state_version, current.run_id),
    )
    _append_event(connection, previous, current)


def _append_event(
    connection: sqlite3.Connection,
    previous: RunState | None,
    current: RunState,
    *,
    history_complete: bool = True,
    ignore_conflict: bool = False,
) -> RunEvent:
    row = connection.execute(
        """
        SELECT COALESCE(MAX(sequence), -1) + 1
        FROM research_run_events
        WHERE run_id = ?
        """,
        (current.run_id,),
    ).fetchone()
    sequence = int(row[0]) if row is not None else 0
    event = project_event(
        previous,
        current,
        sequence,
        history_complete=history_complete,
    )
    insert = (
        "INSERT OR IGNORE INTO research_run_events"
        if ignore_conflict
        else "INSERT INTO research_run_events"
    )
    connection.execute(
        f"""
        {insert} (
            run_id,
            sequence,
            state_version,
            event_type,
            occurred_at,
            payload_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            event.run_id,
            event.sequence,
            event.state_version,
            event.event_type,
            event.occurred_at,
            event_to_json(event),
        ),
    )
    return event


def _backfill_event_snapshots(connection: sqlite3.Connection) -> None:
    """Give pre-event databases one honest snapshot without inventing history."""
    rows = connection.execute(
        """
        SELECT runs.state_json
        FROM research_runs AS runs
        WHERE NOT EXISTS (
            SELECT 1
            FROM research_run_events AS events
            WHERE events.run_id = runs.run_id
        )
        ORDER BY runs.run_id
        """
    ).fetchall()
    for row in rows:
        state = _deserialize_state(str(row[0]))
        _append_event(
            connection,
            None,
            state,
            history_complete=False,
            ignore_conflict=True,
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


def _state_to_dict(state: RunState) -> dict[str, object]:
    payload = asdict(state)
    payload["completed_steps"] = list(state.completed_steps)
    payload["effects"] = [asdict(effect) for effect in state.effects]
    payload["conversation"] = [asdict(message) for message in state.conversation]
    payload["plan_controls"] = [asdict(control) for control in state.plan_controls]
    return payload


def _serialize_state(state: RunState) -> str:
    return json.dumps(_state_to_dict(state), ensure_ascii=False, separators=(",", ":"))


def _deserialize_state(payload: str) -> RunState:
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise TypeError("checkpoint must contain a JSON object")
    return _state_from_dict(decoded)


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
    raw_conversation = payload.get("conversation", [])
    if not isinstance(raw_conversation, list):
        raise TypeError("checkpoint conversation must be a JSON array")
    conversation = tuple(
        _decode_dataclass(ConversationMessage, item) for item in raw_conversation
    )
    raw_plan_controls = payload.get("plan_controls", [])
    if not isinstance(raw_plan_controls, list):
        raise TypeError("checkpoint plan_controls must be a JSON array")
    plan_controls = tuple(
        _decode_dataclass(PlanControlRecord, item) for item in raw_plan_controls
    )
    state = RunState(
        run_id=str(payload.get("run_id", "")),
        request=request,
        engine_fingerprint=str(payload.get("engine_fingerprint", "")),
        model_name=str(payload.get("model_name", "")),
        status=str(payload.get("status", "")),
        current_step=str(payload.get("current_step", "")),
        completed_steps=tuple(str(item) for item in raw_completed_steps),
        effects=effects,
        conversation=conversation,
        clarification=(
            None
            if payload.get("clarification") is None
            else _decode_dataclass(
                ClarificationDecision,
                payload.get("clarification"),
            )
        ),
        plan_controls=plan_controls,
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
    _validate_loaded_plan_controls(state)
    return state


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


def research_plan_hash(plan: ResearchPlan) -> str:
    """Return a stable SHA-256 digest for optimistic plan controls."""
    encoded = json.dumps(
        _to_jsonable(plan),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def research_plan_version_hash(plan: ResearchPlan, base_plan_hash: str) -> str:
    """Bind a plan version to its parent so equal content cannot create ABA."""
    encoded = json.dumps(
        {
            "base_plan_hash": base_plan_hash,
            "plan": _to_jsonable(plan),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def plan_control_input_hash(
    *,
    run_id: str,
    kind: str,
    base_plan_hash: str,
    plan: ResearchPlan,
) -> str:
    """Return the stable identity of one append-only plan control command."""
    encoded = json.dumps(
        {
            "schema_version": 1,
            "run_id": run_id,
            "kind": kind,
            "base_plan_hash": base_plan_hash,
            "plan": _to_jsonable(plan),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_loaded_plan_controls(state: RunState) -> None:
    if not state.plan_controls:
        return
    generated_plan = _initial_generated_plan(state.effects)
    if generated_plan is None:
        raise ValueError("plan controls require an initial generated plan")
    current_plan = generated_plan
    current_hash = research_plan_hash(generated_plan)
    approved = False
    for control in state.plan_controls:
        if approved:
            raise ValueError("plan controls cannot follow approval")
        expected_input_hash = plan_control_input_hash(
            run_id=state.run_id,
            kind=control.kind,
            base_plan_hash=control.base_plan_hash,
            plan=control.plan,
        )
        if control.input_hash != expected_input_hash:
            raise ValueError("plan control has an invalid input hash")
        if control.base_plan_hash != current_hash:
            raise ValueError("plan control history has an invalid base plan hash")
        if control.kind == "edit":
            expected_hash = research_plan_version_hash(
                control.plan,
                control.base_plan_hash,
            )
            if control.plan_hash != expected_hash:
                raise ValueError("edited plan control has an invalid plan hash")
            current_plan = control.plan
            current_hash = control.plan_hash
            continue
        if control.plan != current_plan or control.plan_hash != current_hash:
            raise ValueError("approval does not match the current plan snapshot")
        approved = True


def _initial_generated_plan(
    effects: tuple[EffectRecord, ...],
) -> ResearchPlan | None:
    for effect in effects:
        if effect.status != "completed" or effect.result_type != "PlanningRun":
            continue
        planning_run = _decode_result(effect.result_type, effect.result)
        if isinstance(planning_run, PlanningRun) and planning_run.plan.revision == 0:
            return planning_run.plan
    return None


def _validate_run_id(run_id: str) -> None:
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must contain only letters, digits, hyphens, or underscores")

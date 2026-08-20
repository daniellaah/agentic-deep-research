import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from agentic_deep_research.checkpoint import (
    EffectRecord,
    RunState,
    SQLiteCheckpointStore,
    _serialize_state,
    _transition,
)
from agentic_deep_research.models import AgentRun, ResearchRequest


def _created_state(run_id: str = "event-run") -> RunState:
    return RunState.create(
        ResearchRequest(topic="Sensitive topic that must not enter events"),
        run_id=run_id,
    )


def test_sqlite_store_commits_ordered_events_with_state_transitions(tmp_path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.sqlite3")
    store.create(_created_state())

    running = store.update(
        "event-run",
        lambda state: _transition(
            state,
            status="running",
            current_step="runner.run",
        ),
    )
    started = EffectRecord(
        effect_id="runner.run:abc",
        kind="runner.run",
        input_hash="abc",
        status="started",
        attempts=1,
        result_type="AgentRun",
    )
    store.update(
        "event-run",
        lambda state: _transition(state, effects=(*state.effects, started)),
    )
    completed = EffectRecord(
        effect_id=started.effect_id,
        kind=started.kind,
        input_hash=started.input_hash,
        status="completed",
        attempts=1,
        result_type="AgentRun",
        result=AgentRun(report="Provider output that must stay private"),
    )
    store.update(
        "event-run",
        lambda state: _transition(state, effects=(completed,)),
    )

    events = store.list_events("event-run", after_sequence=-1, limit=100)

    assert [event.sequence for event in events] == [0, 1, 2, 3]
    assert [event.event_type for event in events] == [
        "run.created",
        "run.running",
        "effect.started",
        "effect.completed",
    ]
    assert events[1].state_version == running.state_version
    assert events[2].effect_kind == "runner.run"
    assert events[3].effect_status == "completed"
    assert all("Sensitive topic" not in event.event_type for event in events)


def test_sqlite_store_does_not_emit_an_event_for_a_noop_update(tmp_path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.sqlite3")
    state = _created_state()
    store.create(state)

    returned = store.update(state.run_id, lambda current: current)

    assert returned == state
    assert len(store.list_events(state.run_id, after_sequence=-1, limit=100)) == 1


def test_event_insert_failure_rolls_back_the_checkpoint_update(tmp_path) -> None:
    path = tmp_path / "checkpoints.sqlite3"
    store = SQLiteCheckpointStore(path)
    initial = _created_state()
    store.create(initial)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_later_events
            BEFORE INSERT ON research_run_events
            WHEN NEW.sequence > 0
            BEGIN
                SELECT RAISE(ABORT, 'event rejected');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="event rejected"):
        store.update(
            initial.run_id,
            lambda state: _transition(state, status="running"),
        )

    assert store.load(initial.run_id) == initial
    assert len(store.list_events(initial.run_id, after_sequence=-1, limit=100)) == 1


def test_existing_database_receives_one_incomplete_history_snapshot(tmp_path) -> None:
    path = tmp_path / "legacy.sqlite3"
    state = _created_state("legacy-run")
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE research_runs (
                run_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                state_version INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO research_runs (run_id, state_json, state_version)
            VALUES (?, ?, ?)
            """,
            (state.run_id, _serialize_state(state), state.state_version),
        )

    store = SQLiteCheckpointStore(path)
    events = store.list_events(state.run_id, after_sequence=-1, limit=100)

    assert len(events) == 1
    assert events[0].event_type == "run.snapshot"
    assert events[0].history_complete is False


def test_concurrent_legacy_store_initialization_keeps_one_snapshot(tmp_path) -> None:
    path = tmp_path / "legacy-concurrent.sqlite3"
    state = _created_state("legacy-concurrent")
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE research_runs (
                run_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                state_version INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO research_runs (run_id, state_json, state_version)
            VALUES (?, ?, ?)
            """,
            (state.run_id, _serialize_state(state), state.state_version),
        )
    barrier = Barrier(4)

    def initialize(_: int) -> None:
        barrier.wait(timeout=5)
        SQLiteCheckpointStore(path)

    with ThreadPoolExecutor(max_workers=4) as executor:
        tuple(executor.map(initialize, range(4)))

    events = SQLiteCheckpointStore(path).list_events(
        state.run_id,
        after_sequence=-1,
        limit=100,
    )
    assert len(events) == 1
    assert events[0].event_type == "run.snapshot"


def test_concurrent_updates_receive_unique_contiguous_event_sequences(tmp_path) -> None:
    path = tmp_path / "checkpoints.sqlite3"
    store = SQLiteCheckpointStore(path)
    state = _created_state("parallel-events")
    store.create(state)

    def advance(index: int) -> None:
        SQLiteCheckpointStore(path).update(
            state.run_id,
            lambda current: _transition(current, current_step=f"step-{index}"),
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        tuple(executor.map(advance, range(4)))

    events = store.list_events(state.run_id, after_sequence=-1, limit=100)

    assert [event.sequence for event in events] == [0, 1, 2, 3, 4]
    assert store.load(state.run_id).state_version == 4
    assert [event.sequence for event in store.list_events(
        state.run_id,
        after_sequence=2,
        limit=2,
    )] == [3, 4]

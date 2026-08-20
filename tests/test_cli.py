from agentic_deep_research import ResearchRequest
from agentic_deep_research.cli import main
from agentic_deep_research.runtime import RunState, SQLiteCheckpointStore


def test_cancel_command_updates_a_checkpoint_without_model_configuration(
    tmp_path,
    capsys,
) -> None:
    database = tmp_path / "checkpoints.sqlite3"
    store = SQLiteCheckpointStore(database)
    store.save(
        RunState.create(
            ResearchRequest(topic="Cancel from the CLI"),
            run_id="cli-cancel",
        )
    )

    main(
        [
            "--checkpoint-db",
            str(database),
            "--cancel",
            "cli-cancel",
            "--reason",
            "User stopped it",
        ]
    )

    state = store.load("cli-cancel")
    assert state.status == "cancelled"
    assert state.termination_reason == "User stopped it"
    assert "run_id=cli-cancel, status=cancelled" in capsys.readouterr().out

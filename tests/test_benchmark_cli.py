import json
from dataclasses import replace

from agentic_deep_research import ResearchRequest, ResearchResult, TokenUsage, benchmark_cli
from agentic_deep_research.benchmarks import BenchmarkCase
from agentic_deep_research.checkpoint import RunState, SQLiteCheckpointStore
from agentic_deep_research.corpus import JsonlCorpusBackend
from agentic_deep_research.evaluation import (
    DurableCaseExecutor,
    EvaluationManifest,
    EvaluationSummary,
    create_or_validate_manifest,
    deterministic_case_run_id,
)
from agentic_deep_research.models import ResearchBudget
from agentic_deep_research.runtime import RetryPolicy


def _write_frames(path, *, question: str = "Private question?", answer: str = "Answer"):
    path.write_text(
        json.dumps({"id": "case-1", "question": question, "answer": answer}) + "\n",
        encoding="utf-8",
    )


def _summary() -> EvaluationSummary:
    return EvaluationSummary(
        experiment_id="exp-1",
        benchmark="frames",
        total=1,
        completed=0,
        correct=0,
        research_errors=1,
        judge_errors=0,
        waiting=0,
        accuracy=0.0,
        accuracy_on_completed=None,
        average_score=0.0,
        average_score_on_completed=None,
        average_tool_calls=0.0,
        average_sources=0.0,
        average_tokens=0.0,
    )


def test_run_cli_creates_manifest_before_openai_and_uses_durable_executor(
    tmp_path,
    monkeypatch,
) -> None:
    data = tmp_path / "frames.jsonl"
    experiment = tmp_path / "experiment"
    _write_frames(data)
    client_calls: list[int] = []
    captured: dict[str, object] = {}

    def fake_openai(*, max_retries: int):
        assert (experiment / "manifest.json").exists()
        client_calls.append(max_retries)
        return object()

    class FakeOrchestrator:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def run(self) -> EvaluationSummary:
            return _summary()

    monkeypatch.setattr(benchmark_cli, "OpenAI", fake_openai)
    monkeypatch.setattr(benchmark_cli, "EvaluationOrchestrator", FakeOrchestrator)

    benchmark_cli.main(
        [
            "run",
            "frames",
            "--data",
            str(data),
            "--experiment-dir",
            str(experiment),
            "--experiment-id",
            "exp-1",
            "--model",
            "research-model",
            "--judge",
            "exact",
            "--limit",
            "1",
        ]
    )

    manifest = json.loads((experiment / "manifest.json").read_text())
    assert manifest["language"] == "English"
    assert manifest["min_sources"] == 2
    assert manifest["grader_version"] == "exact-match-v1"
    assert client_calls == [0]
    assert isinstance(captured["executor"], DurableCaseExecutor)
    assert captured["output_path"] == experiment / "records.jsonl"


def test_run_cli_binds_fixed_corpus_and_disables_web_verification(
    tmp_path,
    monkeypatch,
) -> None:
    data = tmp_path / "frames.jsonl"
    corpus = tmp_path / "corpus.jsonl"
    experiment = tmp_path / "fixed-experiment"
    _write_frames(data)
    corpus.write_text(
        json.dumps(
            {
                "id": "doc-1",
                "title": "Reliability evaluation",
                "text": "Evaluation makes agent reliability measurable.",
                "url": "https://example.test/reliability",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class FakeOrchestrator:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def run(self) -> EvaluationSummary:
            return _summary()

    monkeypatch.setattr(benchmark_cli, "OpenAI", lambda **_: object())
    monkeypatch.setattr(benchmark_cli, "EvaluationOrchestrator", FakeOrchestrator)

    benchmark_cli.main(
        [
            "run",
            "frames",
            "--data",
            str(data),
            "--experiment-dir",
            str(experiment),
            "--experiment-id",
            "fixed-exp-1",
            "--model",
            "research-model",
            "--judge",
            "exact",
            "--report-judge",
            "openai",
            "--search-protocol",
            "fixed-corpus",
            "--corpus",
            str(corpus),
            "--limit",
            "1",
        ]
    )

    manifest = json.loads((experiment / "manifest.json").read_text())
    assert manifest["search_protocol"] == "fixed-corpus-lexical-v1"
    assert manifest["corpus_sha256"] == JsonlCorpusBackend(corpus).corpus_sha256
    assert manifest["report_judge_model"] == "research-model"
    executor = captured["executor"]
    assert isinstance(executor, DurableCaseExecutor)
    assert executor._runtime.report_model_name == ""


def test_replay_cli_uses_checkpoint_without_constructing_openai(
    tmp_path,
    monkeypatch,
) -> None:
    question = "HIGHLY PRIVATE QUESTION"
    answer = "Answer"
    data = tmp_path / "frames.jsonl"
    experiment = tmp_path / "experiment"
    _write_frames(data, question=question, answer=answer)
    case = BenchmarkCase(id="case-1", question=question, answer=answer)
    manifest = EvaluationManifest.create(
        experiment_id="exp-1",
        benchmark="frames",
        cases=(case,),
        research_model="research-model",
        judge_model="deterministic",
        budget=ResearchBudget(),
        retry_policy=RetryPolicy(),
        search_protocol="verified-adaptive-live-web-v4",
        grader_version="exact-match-v1",
    )
    create_or_validate_manifest(experiment / "manifest.json", manifest)
    result = ResearchResult(
        topic=question,
        report=answer,
        raw_report=answer,
        sources=(),
        citations=(),
        evidence=(),
        trace=(),
        status="completed",
        stop_reason="completed",
        usage=TokenUsage(),
    )
    run_id = deterministic_case_run_id(manifest.experiment_id, case.id)
    state = replace(
        RunState.create(ResearchRequest(topic=question), run_id=run_id),
        status="completed",
        current_step="completed",
        result=result,
    )
    SQLiteCheckpointStore(experiment / "checkpoints.sqlite3").create(state)

    def fail_openai(**_: object) -> object:
        raise AssertionError("strict offline replay must not construct an OpenAI client")

    monkeypatch.setattr(benchmark_cli, "OpenAI", fail_openai)

    benchmark_cli.main(
        [
            "replay",
            "frames",
            "--data",
            str(data),
            "--experiment-dir",
            str(experiment),
        ]
    )

    serialized = (experiment / "replay-records.jsonl").read_text()
    assert question not in serialized
    assert answer not in serialized
    assert json.loads(serialized)["judgment"]["correct"] is True

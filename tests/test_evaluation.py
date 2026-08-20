import json
import stat
from dataclasses import replace

import pytest

from agentic_deep_research.benchmarks import BenchmarkCase, Judgment
from agentic_deep_research.checkpoint import EffectRecord, JsonCheckpointStore, RunState
from agentic_deep_research.evaluation import (
    CaseExecution,
    DurableCaseExecutor,
    EvaluationManifest,
    EvaluationOrchestrator,
    ReplayCaseExecutor,
    TrajectoryMetrics,
    _experiment_lock,
    benchmark_dataset_sha256,
    create_or_validate_manifest,
    deterministic_case_run_id,
)
from agentic_deep_research.models import (
    Citation,
    CitationCheck,
    Evidence,
    ResearchBudget,
    ResearchRequest,
    ResearchResult,
    ResearchStep,
    Source,
    TokenUsage,
)
from agentic_deep_research.report_evaluation import DeterministicReportJudge
from agentic_deep_research.runtime import RetryPolicy, RuntimeOutcome


class FakeRuntime:
    def __init__(
        self,
        *,
        start_state: RunState,
        resume_state: RunState | None = None,
    ) -> None:
        self._start_state = start_state
        self._resume_state = resume_state
        self.states: dict[str, RunState] = {}
        self.start_calls = 0
        self.resume_calls = 0
        self.get_calls = 0
        self.model_name = "research-model"
        self.planner_model_name = "research-model"
        self.report_model_name = "research-model"
        self.corpus_sha256 = None
        self.retry_policy = RetryPolicy(max_attempts=2, jitter_ratio=0)

    def get(self, run_id: str) -> RunState:
        self.get_calls += 1
        try:
            return self.states[run_id]
        except KeyError as error:
            raise FileNotFoundError(run_id) from error

    def start(
        self,
        request: ResearchRequest,
        *,
        run_id: str,
    ) -> RuntimeOutcome:
        assert request.topic
        self.start_calls += 1
        state = replace(self._start_state, run_id=run_id)
        self.states[run_id] = state
        return RuntimeOutcome(state=state, result=state.result)

    def resume(self, run_id: str) -> RuntimeOutcome:
        self.resume_calls += 1
        assert self._resume_state is not None
        state = replace(self._resume_state, run_id=run_id)
        self.states[run_id] = state
        return RuntimeOutcome(state=state, result=state.result)


class CountingJudge:
    def __init__(self, *, fail_once: bool = False) -> None:
        self.calls = 0
        self._fail_once = fail_once

    @property
    def grader(self) -> str:
        return "test-grader-v1"

    @property
    def model_name(self) -> str:
        return "judge-model"

    def judge(self, *, question: str, reference: str, prediction: str) -> Judgment:
        self.calls += 1
        if self._fail_once:
            self._fail_once = False
            raise RuntimeError("temporary grader failure with sensitive context")
        return Judgment(
            correct=prediction == reference,
            score=1.0 if prediction == reference else 0.0,
            reason=f"Private rationale about {question} and {reference}",
            grader="test-grader-v1",
        )


def _manifest(
    cases: tuple[BenchmarkCase, ...],
    *,
    research_model: str = "research-model",
) -> EvaluationManifest:
    return EvaluationManifest.create(
        experiment_id="experiment-001",
        benchmark="test-benchmark",
        cases=cases,
        research_model=research_model,
        judge_model="judge-model",
        budget=ResearchBudget(max_tool_calls=3, max_research_steps=2),
        retry_policy=RetryPolicy(max_attempts=2, jitter_ratio=0),
        search_protocol="verified-adaptive-live-web-v3",
        grader_version="test-grader-v1",
    )


def _result(topic: str, report: str) -> ResearchResult:
    source = Source(
        title="Secret source title",
        url="https://example.com/private-source",
    )
    citation = Citation(source=source, start_index=0, end_index=3)
    evidence = Evidence(
        id="ev-secret",
        claim="Secret evidence claim",
        source=source,
        verification_status="supported",
    )
    check = CitationCheck(
        claim="Secret checked claim",
        source=source,
        status="supported",
        reason="Secret citation reason",
        evidence_id=evidence.id,
    )
    return ResearchResult(
        topic=topic,
        report=report,
        raw_report=report,
        sources=(source,),
        citations=(citation,),
        evidence=(evidence,),
        trace=(
            ResearchStep(action="search", detail="Secret query", kind="tool"),
            ResearchStep(action="reason", detail="Secret thought", kind="reasoning"),
        ),
        status="completed",
        stop_reason="completed",
        usage=TokenUsage(input_tokens=11, output_tokens=7, total_tokens=18),
        citation_checks=(check,),
        revision_count=2,
    )


def _state(
    *,
    status: str,
    result: ResearchResult | None = None,
    error_type: str | None = None,
) -> RunState:
    state = RunState.create(ResearchRequest(topic="placeholder"), run_id="placeholder")
    effect = EffectRecord(
        effect_id="effect-safe-digest",
        kind="worker",
        input_hash="private-input-digest",
        status="completed" if status == "completed" else "failed",
        attempts=3,
        result_type="AgentRun",
        result=None,
        error_type=error_type,
        error_message="Secret provider error",
        retryable=True,
    )
    return replace(
        state,
        status=status,
        current_step="completed" if status == "completed" else "worker",
        completed_steps=("planner",),
        effects=(effect,),
        result=result,
        error_type=error_type,
        error_message="Secret runtime error" if error_type else None,
    )


def test_manifest_create_or_validate_rejects_protocol_or_dataset_drift(tmp_path) -> None:
    cases = (BenchmarkCase(id="one", question="Question?", answer="Answer"),)
    manifest = _manifest(cases)
    path = tmp_path / "manifest.json"

    assert create_or_validate_manifest(path, manifest) == manifest
    assert create_or_validate_manifest(path, manifest) == manifest
    assert json.loads(path.read_text())["dataset_sha256"] == benchmark_dataset_sha256(cases)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    with pytest.raises(ValueError, match="does not match"):
        create_or_validate_manifest(path, _manifest(cases, research_model="other-model"))

    changed_cases = (BenchmarkCase(id="one", question="Question?", answer="Changed"),)
    assert benchmark_dataset_sha256(changed_cases) != manifest.dataset_sha256

    with pytest.raises(ValueError, match="fixed-corpus protocols require"):
        replace(manifest, search_protocol="fixed-corpus-v1")
    with pytest.raises(ValueError, match="does not match"):
        create_or_validate_manifest(path, replace(manifest, min_sources=3))


def test_experiment_lock_rejects_a_concurrent_writer(tmp_path) -> None:
    with _experiment_lock(tmp_path), pytest.raises(
        RuntimeError,
        match="already running",
    ), _experiment_lock(tmp_path):
        pass

    assert stat.S_IMODE((tmp_path / ".evaluation.lock").stat().st_mode) == 0o600


def test_executor_rejects_runtime_configuration_before_research() -> None:
    case = BenchmarkCase(id="one", question="Question?", answer="Answer")
    manifest = _manifest((case,), research_model="different-model")
    runtime = FakeRuntime(start_state=_state(status="failed"))

    with pytest.raises(ValueError, match="runtime model"):
        DurableCaseExecutor(runtime).execute(case, manifest)  # type: ignore[arg-type]

    assert runtime.start_calls == 0

    fixed_manifest = replace(
        _manifest((case,)),
        search_protocol="fixed-corpus-lexical-v1",
        corpus_sha256="a" * 64,
    )
    runtime.model_name = fixed_manifest.research_model
    runtime.corpus_sha256 = "b" * 64
    with pytest.raises(ValueError, match="runtime corpus"):
        DurableCaseExecutor(
            runtime,  # type: ignore[arg-type]
            search_protocol="fixed-corpus-lexical-v1",
        ).execute(case, fixed_manifest)

    assert runtime.start_calls == 0


def test_research_error_resumes_then_completed_record_is_skipped(tmp_path) -> None:
    case = BenchmarkCase(id="one", question="Private question?", answer="Private answer")
    cases = (case,)
    manifest = _manifest(cases)
    failed = _state(status="failed", error_type="ConnectionError")
    completed = _state(
        status="completed",
        result=_result(case.question, case.answer),
    )
    runtime = FakeRuntime(start_state=failed, resume_state=completed)
    judge = CountingJudge()
    output_path = tmp_path / "records.jsonl"
    evaluator = EvaluationOrchestrator(
        manifest=manifest,
        cases=cases,
        executor=DurableCaseExecutor(runtime),  # type: ignore[arg-type]
        judge=judge,
        output_path=output_path,
    )

    first = evaluator.run()
    assert first.research_errors == 1
    assert runtime.start_calls == 1
    assert runtime.resume_calls == 0

    second = evaluator.run()
    assert second.completed == 1
    assert second.correct == 1
    assert runtime.start_calls == 1
    assert runtime.resume_calls == 1
    assert judge.calls == 1

    third = evaluator.run()
    assert third == second
    assert runtime.start_calls == 1
    assert runtime.resume_calls == 1
    assert judge.calls == 1
    assert len(output_path.read_text().splitlines()) == 2


def test_judge_error_only_rejudges_and_jsonl_never_copies_artifacts(tmp_path) -> None:
    case = BenchmarkCase(
        id="secret-case-id",
        question="HIGHLY PRIVATE QUESTION",
        answer="HIGHLY PRIVATE ANSWER",
    )
    cases = (case,)
    manifest = _manifest(cases)
    completed = _state(
        status="completed",
        result=_result(case.question, case.answer),
    )
    runtime = FakeRuntime(start_state=completed)
    judge = CountingJudge(fail_once=True)
    output_path = tmp_path / "records.jsonl"
    evaluator = EvaluationOrchestrator(
        manifest=manifest,
        cases=cases,
        executor=DurableCaseExecutor(runtime),  # type: ignore[arg-type]
        judge=judge,
        output_path=output_path,
    )

    first = evaluator.run()
    assert first.judge_errors == 1
    assert runtime.start_calls == 1
    assert runtime.resume_calls == 0

    second = evaluator.run()
    assert second.completed == 1
    assert runtime.start_calls == 1
    assert runtime.resume_calls == 0
    assert judge.calls == 2

    serialized = output_path.read_text()
    forbidden = (
        case.question,
        case.answer,
        "Secret source title",
        "private-source",
        "Secret evidence claim",
        "Secret checked claim",
        "Secret citation reason",
        "Secret query",
        "Secret thought",
        "Secret provider error",
        "Secret runtime error",
        "Private rationale",
        "temporary grader failure with sensitive context",
    )
    assert not any(value in serialized for value in forbidden)

    latest = json.loads(serialized.splitlines()[-1])
    assert set(latest) == {
        "benchmark",
        "case_id",
        "error_type",
        "experiment_id",
        "judgment",
        "report_judgment",
        "run_id",
        "schema_version",
        "status",
        "trajectory",
    }
    assert "reason" not in latest["judgment"]
    assert len(latest["judgment"]["reason_sha256"]) == 64


def test_trace_v2_metrics_count_retries_tools_sources_tokens_and_citations() -> None:
    result = _result("Topic", "Answer")
    state = _state(status="completed", result=result)
    case = BenchmarkCase(id="one", question="Topic", answer="Answer")
    manifest = _manifest((case,))
    runtime = FakeRuntime(start_state=state)
    case_execution = DurableCaseExecutor(runtime).execute(  # type: ignore[arg-type]
        case,
        manifest,
    )

    metrics = TrajectoryMetrics.from_execution(case_execution)

    assert metrics == TrajectoryMetrics(
        provider_effects=1,
        completed_effects=1,
        failed_effects=0,
        retry_attempts=2,
        tool_calls=1,
        source_count=1,
        evidence_count=1,
        citation_count=1,
        supported_citations=1,
        unsupported_citations=0,
        uncertain_citations=0,
        unverified_evidence=0,
        supported_evidence=1,
        unsupported_evidence=0,
        uncertain_evidence=0,
        citation_support_rate=1.0,
        evidence_yield_per_tool_call=1.0,
        planned_questions=0,
        completed_findings=0,
        input_tokens=11,
        output_tokens=7,
        total_tokens=18,
        revision_count=2,
    )


def test_trajectory_does_not_count_an_unbound_supported_check() -> None:
    result = _result("Topic", "Answer")
    result = replace(
        result,
        citation_checks=(replace(result.citation_checks[0], evidence_id=""),),
    )
    execution = CaseExecution(
        case_id="one",
        run_id="run-one",
        status="completed",
        state=_state(status="completed", result=result),
        result=result,
    )

    metrics = TrajectoryMetrics.from_execution(execution)

    assert metrics.citation_count == 1
    assert metrics.supported_citations == 0
    assert metrics.citation_support_rate == 0.0


def test_waiting_checkpoint_is_reused_and_replay_executor_only_loads(tmp_path) -> None:
    case = BenchmarkCase(id="one", question="Question?", answer="Answer")
    cases = (case,)
    manifest = _manifest(cases)
    waiting = _state(status="waiting_for_human", error_type="AmbiguousEffect")
    runtime = FakeRuntime(start_state=waiting)
    executor = DurableCaseExecutor(runtime)  # type: ignore[arg-type]

    assert executor.execute(case, manifest).status == "waiting"
    assert executor.execute(case, manifest).status == "waiting"
    assert runtime.start_calls == 1
    assert runtime.resume_calls == 0

    store = JsonCheckpointStore(tmp_path / "checkpoints")
    run_id = deterministic_case_run_id(manifest.experiment_id, case.id)
    completed = replace(
        _state(status="completed", result=_result(case.question, case.answer)),
        run_id=run_id,
    )
    store.create(completed)
    replay = ReplayCaseExecutor(store)

    assert replay.execute(case, manifest).status == "completed"
    missing_case = BenchmarkCase(id="missing", question="Missing?", answer="Missing")
    missing_manifest = _manifest((missing_case,))
    missing = replay.execute(missing_case, missing_manifest)
    assert missing.status == "research_error"
    assert missing.error_type == "FileNotFoundError"


def test_quality_gate_failure_is_scored_once_without_calling_the_judge(tmp_path) -> None:
    case = BenchmarkCase(id="one", question="Question?", answer="Answer")
    cases = (case,)
    result = replace(
        _result(case.question, case.answer),
        status="needs_review",
        stop_reason="insufficient_sources",
    )
    runtime = FakeRuntime(start_state=_state(status="completed", result=result))
    judge = CountingJudge()
    output_path = tmp_path / "records.jsonl"
    evaluator = EvaluationOrchestrator(
        manifest=_manifest(cases),
        cases=cases,
        executor=DurableCaseExecutor(runtime),  # type: ignore[arg-type]
        judge=judge,
        output_path=output_path,
    )

    summary = evaluator.run()
    record = json.loads(output_path.read_text())

    assert summary.completed == 1
    assert summary.correct == 0
    assert judge.calls == 0
    assert record["judgment"]["grader"] == "research-quality-gate-v1"
    assert record["trajectory"]["research_status"] == "needs_review"
    assert record["trajectory"]["stop_reason"] == "insufficient_sources"


def test_invalid_judge_object_becomes_resumable_judge_error(tmp_path) -> None:
    class InvalidJudge:
        grader = "test-grader-v1"
        model_name = "judge-model"

        def judge(self, **_: object) -> object:
            return object()

    case = BenchmarkCase(id="one", question="Question?", answer="Answer")
    cases = (case,)
    runtime = FakeRuntime(
        start_state=_state(
            status="completed",
            result=_result(case.question, case.answer),
        )
    )
    output_path = tmp_path / "records.jsonl"

    summary = EvaluationOrchestrator(
        manifest=_manifest(cases),
        cases=cases,
        executor=DurableCaseExecutor(runtime),  # type: ignore[arg-type]
        judge=InvalidJudge(),  # type: ignore[arg-type]
        output_path=output_path,
    ).run()

    assert summary.judge_errors == 1
    assert json.loads(output_path.read_text())["error_type"] == "AttributeError"


def test_waiting_record_is_not_duplicated_and_jsonl_is_private(tmp_path) -> None:
    case = BenchmarkCase(id="one", question="Question?", answer="Answer")
    cases = (case,)
    runtime = FakeRuntime(start_state=_state(status="waiting_for_human"))
    output_path = tmp_path / "records.jsonl"
    evaluator = EvaluationOrchestrator(
        manifest=_manifest(cases),
        cases=cases,
        executor=DurableCaseExecutor(runtime),  # type: ignore[arg-type]
        judge=CountingJudge(),
        output_path=output_path,
    )

    evaluator.run()
    evaluator.run()

    assert len(output_path.read_text().splitlines()) == 1
    assert stat.S_IMODE(output_path.stat().st_mode) == 0o600


def test_long_report_rubric_is_persisted_and_summarized(tmp_path) -> None:
    case = BenchmarkCase(id="one", question="Question?", answer="Answer")
    cases = (case,)
    report_judge = DeterministicReportJudge()
    manifest = replace(
        _manifest(cases),
        report_grader_version=report_judge.grader,
        report_judge_model=report_judge.model_name,
    )
    runtime = FakeRuntime(
        start_state=_state(
            status="completed",
            result=_result(case.question, case.answer),
        )
    )
    output_path = tmp_path / "records.jsonl"

    summary = EvaluationOrchestrator(
        manifest=manifest,
        cases=cases,
        executor=DurableCaseExecutor(runtime),  # type: ignore[arg-type]
        judge=CountingJudge(),
        report_judge=report_judge,
        output_path=output_path,
    ).run()
    record = json.loads(output_path.read_text())

    assert summary.reports_evaluated == 1
    assert summary.reports_passed == 1
    assert summary.average_report_score == 0.9
    assert record["report_judgment"]["grader"] == report_judge.grader
    assert all("reason" not in score for score in record["report_judgment"]["scores"])

    with pytest.raises(ValueError, match="report judge does not match"):
        EvaluationOrchestrator(
            manifest=manifest,
            cases=cases,
            executor=DurableCaseExecutor(runtime),  # type: ignore[arg-type]
            judge=CountingJudge(),
            report_judge=DeterministicReportJudge(pass_threshold=0.8),
            output_path=tmp_path / "other.jsonl",
        )

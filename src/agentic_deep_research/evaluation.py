"""Durable, privacy-conscious orchestration for repeatable research evaluations."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4

from .benchmarks import AnswerJudge, BenchmarkCase, Judgment
from .checkpoint import CheckpointStore, EffectRecord, RunState
from .models import ResearchBudget, ResearchRequest, ResearchResult
from .report_evaluation import ReportJudge, ReportJudgment
from .runtime import ResearchRuntime, RetryPolicy, RuntimeOutcome

EvaluationStatus = Literal[
    "completed",
    "waiting",
    "research_error",
    "judge_error",
]


@dataclass(frozen=True)
class EvaluationManifest:
    """Immutable identity and protocol configuration for one experiment."""

    experiment_id: str
    benchmark: str
    dataset_sha256: str
    selected_case_ids: tuple[str, ...]
    research_model: str
    judge_model: str
    budget: ResearchBudget
    retry_policy: RetryPolicy
    search_protocol: str
    grader_version: str
    language: str = "English"
    min_sources: int = 2
    require_citations: bool = True
    corpus_sha256: str | None = None
    report_grader_version: str | None = None
    report_judge_model: str | None = None
    protocol_version: str = "research-eval-v2"
    schema_version: int = 1

    def __post_init__(self) -> None:
        text_fields = {
            "experiment_id": self.experiment_id,
            "benchmark": self.benchmark,
            "research_model": self.research_model,
            "judge_model": self.judge_model,
            "search_protocol": self.search_protocol,
            "grader_version": self.grader_version,
            "language": self.language,
            "protocol_version": self.protocol_version,
        }
        for name, value in text_fields.items():
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        if self.min_sources < 1:
            raise ValueError("min_sources must be at least 1")
        if type(self.require_citations) is not bool:
            raise TypeError("require_citations must be a boolean")
        if len(self.dataset_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.dataset_sha256
        ):
            raise ValueError("dataset_sha256 must be a lowercase SHA-256 digest")
        if not self.selected_case_ids:
            raise ValueError("selected_case_ids must not be empty")
        if len(self.selected_case_ids) != len(set(self.selected_case_ids)):
            raise ValueError("selected_case_ids must be unique")
        if any(not case_id.strip() for case_id in self.selected_case_ids):
            raise ValueError("selected case IDs must not be empty")
        if self.schema_version != 1:
            raise ValueError(f"unsupported evaluation manifest schema: {self.schema_version}")
        if self.corpus_sha256 is not None and (
            len(self.corpus_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.corpus_sha256)
        ):
            raise ValueError("corpus_sha256 must be a lowercase SHA-256 digest")
        if self.search_protocol.startswith("fixed-corpus") and self.corpus_sha256 is None:
            raise ValueError("fixed-corpus protocols require corpus_sha256")
        if (self.report_grader_version is None) != (self.report_judge_model is None):
            raise ValueError(
                "report_grader_version and report_judge_model must be configured together"
            )

    @classmethod
    def create(
        cls,
        *,
        experiment_id: str,
        benchmark: str,
        cases: Iterable[BenchmarkCase],
        research_model: str,
        judge_model: str,
        budget: ResearchBudget,
        retry_policy: RetryPolicy,
        search_protocol: str,
        grader_version: str,
        language: str = "English",
        min_sources: int = 2,
        require_citations: bool = True,
        corpus_sha256: str | None = None,
        report_grader_version: str | None = None,
        report_judge_model: str | None = None,
        protocol_version: str = "research-eval-v2",
    ) -> EvaluationManifest:
        """Build a manifest whose dataset digest covers all selected case content."""
        selected = tuple(cases)
        return cls(
            experiment_id=experiment_id,
            benchmark=benchmark,
            dataset_sha256=benchmark_dataset_sha256(selected),
            selected_case_ids=tuple(case.id for case in selected),
            research_model=research_model,
            judge_model=judge_model,
            budget=budget,
            retry_policy=retry_policy,
            search_protocol=search_protocol,
            grader_version=grader_version,
            language=language,
            min_sources=min_sources,
            require_citations=require_citations,
            corpus_sha256=corpus_sha256,
            report_grader_version=report_grader_version,
            report_judge_model=report_judge_model,
            protocol_version=protocol_version,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON-compatible representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> EvaluationManifest:
        """Strictly restore a manifest from its JSON representation."""
        expected_fields = {
            "experiment_id",
            "benchmark",
            "dataset_sha256",
            "selected_case_ids",
            "research_model",
            "judge_model",
            "budget",
            "retry_policy",
            "search_protocol",
            "grader_version",
            "language",
            "min_sources",
            "require_citations",
            "corpus_sha256",
            "report_grader_version",
            "report_judge_model",
            "protocol_version",
            "schema_version",
        }
        if set(payload) != expected_fields:
            raise ValueError("evaluation manifest fields do not match schema version 1")
        budget_payload = payload["budget"]
        retry_payload = payload["retry_policy"]
        if not isinstance(budget_payload, Mapping) or not isinstance(retry_payload, Mapping):
            raise TypeError("manifest budget and retry_policy must be objects")
        selected_ids = payload["selected_case_ids"]
        if not isinstance(selected_ids, list):
            raise TypeError("manifest selected_case_ids must be an array")
        return cls(
            experiment_id=str(payload["experiment_id"]),
            benchmark=str(payload["benchmark"]),
            dataset_sha256=str(payload["dataset_sha256"]),
            selected_case_ids=tuple(str(case_id) for case_id in selected_ids),
            research_model=str(payload["research_model"]),
            judge_model=str(payload["judge_model"]),
            budget=ResearchBudget(**dict(budget_payload)),
            retry_policy=RetryPolicy(**dict(retry_payload)),
            search_protocol=str(payload["search_protocol"]),
            grader_version=str(payload["grader_version"]),
            language=str(payload["language"]),
            min_sources=int(payload["min_sources"]),
            require_citations=payload["require_citations"],  # type: ignore[arg-type]
            corpus_sha256=(
                str(payload["corpus_sha256"])
                if payload["corpus_sha256"] is not None
                else None
            ),
            report_grader_version=(
                str(payload["report_grader_version"])
                if payload["report_grader_version"] is not None
                else None
            ),
            report_judge_model=(
                str(payload["report_judge_model"])
                if payload["report_judge_model"] is not None
                else None
            ),
            protocol_version=str(payload["protocol_version"]),
            schema_version=int(payload["schema_version"]),
        )


def benchmark_dataset_sha256(cases: Iterable[BenchmarkCase]) -> str:
    """Hash benchmark content without persisting that sensitive content."""
    rows = [
        {
            "id": case.id,
            "question": case.question,
            "answer": case.answer,
            "metadata": case.metadata,
        }
        for case in cases
    ]
    encoded = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def create_or_validate_manifest(
    path: Path,
    expected: EvaluationManifest,
) -> EvaluationManifest:
    """Create the experiment manifest once or reject incompatible resumption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        expected.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    temporary = _write_private_temporary(path, serialized)
    try:
        os.link(temporary, path)
        _fsync_directory(path.parent)
        return expected
    except FileExistsError:
        os.chmod(path, 0o600)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("evaluation manifest must contain a JSON object")
        existing = EvaluationManifest.from_dict(payload)
        if existing != expected:
            raise ValueError("existing evaluation manifest does not match this experiment")
        return existing
    finally:
        temporary.unlink(missing_ok=True)


def deterministic_case_run_id(experiment_id: str, case_id: str) -> str:
    """Derive a stable, checkpoint-safe run identity without exposing the case ID."""
    if not experiment_id.strip() or not case_id.strip():
        raise ValueError("experiment_id and case_id must not be empty")
    digest = hashlib.sha256(f"{experiment_id}\0{case_id}".encode()).hexdigest()
    return f"eval-{digest[:40]}"


@dataclass(frozen=True)
class CaseExecution:
    """A checkpoint-backed research outcome for one benchmark case."""

    case_id: str
    run_id: str
    status: Literal["completed", "waiting", "research_error"]
    state: RunState | None = field(default=None, repr=False, compare=False)
    result: ResearchResult | None = field(default=None, repr=False, compare=False)
    error_type: str | None = None


class CaseExecutor(Protocol):
    """Execute or load a benchmark case through durable research state."""

    def execute(
        self,
        case: BenchmarkCase,
        manifest: EvaluationManifest,
    ) -> CaseExecution:
        """Start, resume, or reuse one case as appropriate."""
        ...

    def load(
        self,
        case: BenchmarkCase,
        manifest: EvaluationManifest,
    ) -> CaseExecution:
        """Load one case without performing provider-facing work."""
        ...


class DurableCaseExecutor:
    """Bridge benchmark cases to :class:`ResearchRuntime` checkpoints."""

    def __init__(
        self,
        runtime: ResearchRuntime,
        *,
        request_factory: Callable[[BenchmarkCase, EvaluationManifest], ResearchRequest]
        | None = None,
        search_protocol: str = "verified-adaptive-live-web-v2",
    ) -> None:
        self._runtime = runtime
        self._request_factory = request_factory or _default_request
        self._search_protocol = search_protocol

    def execute(
        self,
        case: BenchmarkCase,
        manifest: EvaluationManifest,
    ) -> CaseExecution:
        self._validate_configuration(manifest)
        run_id = deterministic_case_run_id(manifest.experiment_id, case.id)
        try:
            state = self._runtime.get(run_id)
        except FileNotFoundError:
            try:
                request = self._request_factory(case, manifest)
                _validate_request(request, case, manifest)
                outcome = self._runtime.start(
                    request,
                    run_id=run_id,
                )
            except FileExistsError:
                return self.load(case, manifest)
            except Exception as error:  # noqa: BLE001 - convert runtime failures to records.
                return _execution_error(case.id, run_id, error)
            return _case_execution(case.id, outcome)
        except Exception as error:  # noqa: BLE001 - checkpoint adapters may vary.
            return _execution_error(case.id, run_id, error)

        if state.status in {"completed", "waiting_for_human", "cancelled"}:
            return _case_execution(
                case.id,
                RuntimeOutcome(state=state, result=state.result),
            )
        try:
            return _case_execution(case.id, self._runtime.resume(run_id))
        except Exception as error:  # noqa: BLE001 - convert runtime failures to records.
            return _execution_error(case.id, run_id, error, state=state)

    def load(
        self,
        case: BenchmarkCase,
        manifest: EvaluationManifest,
    ) -> CaseExecution:
        self._validate_configuration(manifest)
        run_id = deterministic_case_run_id(manifest.experiment_id, case.id)
        try:
            state = self._runtime.get(run_id)
        except Exception as error:  # noqa: BLE001 - replay must remain recordable.
            return _execution_error(case.id, run_id, error)
        return _case_execution(
            case.id,
            RuntimeOutcome(state=state, result=state.result),
        )

    def _validate_configuration(self, manifest: EvaluationManifest) -> None:
        if self._runtime.model_name != manifest.research_model:
            raise ValueError("runtime model does not match the evaluation manifest")
        component_models = (
            self._runtime.planner_model_name,
            self._runtime.report_model_name,
        )
        if any(model and model != manifest.research_model for model in component_models):
            raise ValueError("runtime component model does not match the manifest")
        if self._runtime.retry_policy != manifest.retry_policy:
            raise ValueError("runtime retry policy does not match the evaluation manifest")
        if self._search_protocol != manifest.search_protocol:
            raise ValueError("search protocol does not match the evaluation manifest")
        if self._runtime.corpus_sha256 != manifest.corpus_sha256:
            raise ValueError("runtime corpus does not match the evaluation manifest")


class ReplayCaseExecutor:
    """Read checkpointed cases for offline replay without executing research."""

    def __init__(self, store: CheckpointStore) -> None:
        self._store = store

    def execute(
        self,
        case: BenchmarkCase,
        manifest: EvaluationManifest,
    ) -> CaseExecution:
        return self.load(case, manifest)

    def load(
        self,
        case: BenchmarkCase,
        manifest: EvaluationManifest,
    ) -> CaseExecution:
        run_id = deterministic_case_run_id(manifest.experiment_id, case.id)
        try:
            state = self._store.load(run_id)
        except Exception as error:  # noqa: BLE001 - missing replay inputs are data errors.
            return _execution_error(case.id, run_id, error)
        return _case_execution(
            case.id,
            RuntimeOutcome(state=state, result=state.result),
        )


@dataclass(frozen=True)
class EffectSnapshot:
    """Safe control-plane projection of a provider effect."""

    effect_id: str
    kind: str
    status: str
    attempts: int
    result_type: str
    error_type: str | None

    def __post_init__(self) -> None:
        if self.status not in {"started", "completed", "failed"}:
            raise ValueError(f"unsupported effect snapshot status: {self.status}")
        if self.attempts < 1:
            raise ValueError("effect snapshot attempts must be at least 1")

    @classmethod
    def from_effect(cls, effect: EffectRecord) -> EffectSnapshot:
        """Drop effect inputs, results, and error messages from the projection."""
        return cls(
            effect_id=effect.effect_id,
            kind=effect.kind,
            status=effect.status,
            attempts=effect.attempts,
            result_type=effect.result_type,
            error_type=effect.error_type,
        )


@dataclass(frozen=True)
class TrajectoryMetrics:
    """Deterministic metrics derived from a durable trajectory and final result."""

    provider_effects: int = 0
    completed_effects: int = 0
    failed_effects: int = 0
    retry_attempts: int = 0
    tool_calls: int = 0
    source_count: int = 0
    evidence_count: int = 0
    citation_count: int = 0
    supported_citations: int = 0
    unsupported_citations: int = 0
    uncertain_citations: int = 0
    unverified_evidence: int = 0
    supported_evidence: int = 0
    unsupported_evidence: int = 0
    uncertain_evidence: int = 0
    citation_support_rate: float | None = None
    evidence_yield_per_tool_call: float | None = None
    planned_questions: int = 0
    completed_findings: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    revision_count: int = 0

    @classmethod
    def from_execution(cls, execution: CaseExecution) -> TrajectoryMetrics:
        """Calculate metrics without retaining any research content."""
        effects = execution.state.effects if execution.state is not None else ()
        result = execution.result
        if result is None:
            return cls(
                provider_effects=len(effects),
                completed_effects=sum(effect.status == "completed" for effect in effects),
                failed_effects=sum(effect.status == "failed" for effect in effects),
                retry_attempts=sum(max(0, effect.attempts - 1) for effect in effects),
            )
        evidence_ids = {item.id for item in result.evidence if item.id}
        supported_checks = [
            check
            for check in result.citation_checks
            if check.status == "supported"
            and check.evidence_id
            and check.evidence_id in evidence_ids
        ]
        unsupported_checks = [
            check
            for check in result.citation_checks
            if check.status == "unsupported"
            and check.evidence_id
            and check.evidence_id in evidence_ids
        ]
        uncertain_checks = [
            check
            for check in result.citation_checks
            if check.status == "uncertain"
            and check.evidence_id
            and check.evidence_id in evidence_ids
        ]
        evidence_statuses = [item.verification_status for item in result.evidence]
        tool_calls = sum(step.kind == "tool" for step in result.trace)
        citation_count = len(result.citations)
        return cls(
            provider_effects=len(effects),
            completed_effects=sum(effect.status == "completed" for effect in effects),
            failed_effects=sum(effect.status == "failed" for effect in effects),
            retry_attempts=sum(max(0, effect.attempts - 1) for effect in effects),
            tool_calls=tool_calls,
            source_count=len(result.sources),
            evidence_count=len(result.evidence),
            citation_count=citation_count,
            supported_citations=len(supported_checks),
            unsupported_citations=len(unsupported_checks),
            uncertain_citations=len(uncertain_checks),
            unverified_evidence=evidence_statuses.count("unverified"),
            supported_evidence=evidence_statuses.count("supported"),
            unsupported_evidence=evidence_statuses.count("unsupported"),
            uncertain_evidence=evidence_statuses.count("uncertain"),
            citation_support_rate=(
                len(supported_checks) / citation_count if citation_count else None
            ),
            evidence_yield_per_tool_call=(
                len(result.evidence) / tool_calls if tool_calls else None
            ),
            planned_questions=(len(result.plan.questions) if result.plan is not None else 0),
            completed_findings=sum(
                finding.status == "completed" for finding in result.findings
            ),
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            total_tokens=result.usage.total_tokens,
            revision_count=result.revision_count,
        )


@dataclass(frozen=True)
class TrajectorySnapshot:
    """Versioned, artifact-free control trace suitable for evaluation JSONL."""

    run_status: str
    current_step: str
    completed_steps: tuple[str, ...]
    research_status: str
    stop_reason: str
    effects: tuple[EffectSnapshot, ...]
    metrics: TrajectoryMetrics
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError(f"unsupported trajectory snapshot schema: {self.schema_version}")

    @classmethod
    def from_execution(cls, execution: CaseExecution) -> TrajectorySnapshot:
        """Project checkpoint state and metrics into the trace-v2 schema."""
        state = execution.state
        return cls(
            run_status=state.status if state is not None else "unavailable",
            current_step=state.current_step if state is not None else "unavailable",
            completed_steps=state.completed_steps if state is not None else (),
            research_status=(
                execution.result.status if execution.result is not None else "unavailable"
            ),
            stop_reason=(
                execution.result.stop_reason
                if execution.result is not None
                else (state.termination_reason if state is not None else None)
                or execution.error_type
                or "unavailable"
            ),
            effects=(
                tuple(EffectSnapshot.from_effect(effect) for effect in state.effects)
                if state is not None
                else ()
            ),
            metrics=TrajectoryMetrics.from_execution(execution),
        )


@dataclass(frozen=True)
class JudgmentSnapshot:
    """Privacy-safe judgment that retains the score but hashes free-form rationale."""

    correct: bool
    score: float
    grader: str
    reason_sha256: str

    def __post_init__(self) -> None:
        if type(self.correct) is not bool:
            raise TypeError("judgment snapshot correct must be a boolean")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("judgment snapshot score must be numeric")
        if not 0 <= self.score <= 1:
            raise ValueError("judgment snapshot score must be between 0 and 1")
        if not self.grader.strip():
            raise ValueError("judgment snapshot grader must not be empty")
        if len(self.reason_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.reason_sha256
        ):
            raise ValueError("reason_sha256 must be a lowercase SHA-256 digest")

    @classmethod
    def from_judgment(cls, judgment: Judgment) -> JudgmentSnapshot:
        """Replace potentially revealing grader prose with a stable digest."""
        return cls(
            correct=judgment.correct,
            score=judgment.score,
            grader=judgment.grader,
            reason_sha256=hashlib.sha256(judgment.reason.encode()).hexdigest(),
        )


@dataclass(frozen=True)
class RubricScoreSnapshot:
    """Privacy-safe projection of one long-report rubric score."""

    criterion_id: str
    score: float
    reason_sha256: str

    def __post_init__(self) -> None:
        if not self.criterion_id.strip():
            raise ValueError("rubric snapshot criterion_id must not be empty")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("rubric snapshot score must be numeric")
        if not 0 <= self.score <= 1:
            raise ValueError("rubric snapshot score must be between 0 and 1")
        if len(self.reason_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.reason_sha256
        ):
            raise ValueError("rubric reason_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class ReportJudgmentSnapshot:
    """Content-free long-report rubric result persisted with a case record."""

    overall_score: float
    passed: bool
    grader: str
    scores: tuple[RubricScoreSnapshot, ...]

    def __post_init__(self) -> None:
        if isinstance(self.overall_score, bool) or not isinstance(
            self.overall_score, (int, float)
        ):
            raise TypeError("report snapshot overall_score must be numeric")
        if not 0 <= self.overall_score <= 1:
            raise ValueError("report snapshot overall_score must be between 0 and 1")
        if type(self.passed) is not bool:
            raise TypeError("report snapshot passed must be a boolean")
        if not self.grader.strip() or not self.scores:
            raise ValueError("report snapshot requires a grader and scores")
        ids = [score.criterion_id for score in self.scores]
        if len(ids) != len(set(ids)):
            raise ValueError("report snapshot criterion IDs must be unique")

    @classmethod
    def from_judgment(cls, judgment: ReportJudgment) -> ReportJudgmentSnapshot:
        return cls(
            overall_score=judgment.overall_score,
            passed=judgment.passed,
            grader=judgment.grader,
            scores=tuple(
                RubricScoreSnapshot(
                    criterion_id=score.criterion_id,
                    score=score.score,
                    reason_sha256=hashlib.sha256(score.reason.encode()).hexdigest(),
                )
                for score in judgment.scores
            ),
        )


@dataclass(frozen=True)
class EvaluationRecord:
    """One append-only, content-free evaluation outcome."""

    experiment_id: str
    benchmark: str
    case_id: str
    run_id: str
    status: EvaluationStatus
    trajectory: TrajectorySnapshot
    judgment: JudgmentSnapshot | None = None
    report_judgment: ReportJudgmentSnapshot | None = None
    error_type: str | None = None
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError(f"unsupported evaluation record schema: {self.schema_version}")
        if self.status not in {"completed", "waiting", "research_error", "judge_error"}:
            raise ValueError(f"unsupported evaluation status: {self.status}")
        if self.status == "completed" and self.judgment is None:
            raise ValueError("completed evaluation records require a judgment")
        if self.status != "completed" and self.judgment is not None:
            raise ValueError("incomplete evaluation records must not contain a judgment")
        if self.status != "completed" and self.report_judgment is not None:
            raise ValueError("incomplete records must not contain a report judgment")

    @property
    def correct(self) -> bool:
        """Return scored correctness, or false for incomplete records."""
        return self.judgment.correct if self.judgment is not None else False


@dataclass(frozen=True)
class EvaluationSummary:
    """Deterministic aggregate over the latest record for every selected case."""

    experiment_id: str
    benchmark: str
    total: int
    completed: int
    correct: int
    research_errors: int
    judge_errors: int
    waiting: int
    accuracy: float
    accuracy_on_completed: float | None
    average_score: float
    average_score_on_completed: float | None
    average_tool_calls: float
    average_sources: float
    average_tokens: float
    reports_evaluated: int = 0
    reports_passed: int = 0
    average_report_score: float | None = None
    schema_version: int = 2


class EvaluationOrchestrator:
    """Resume research and judging independently using append-only case records."""

    def __init__(
        self,
        *,
        manifest: EvaluationManifest,
        cases: Iterable[BenchmarkCase],
        executor: CaseExecutor,
        judge: AnswerJudge,
        output_path: Path,
        manifest_path: Path | None = None,
        summary_path: Path | None = None,
        report_judge: ReportJudge | None = None,
    ) -> None:
        self._manifest = manifest
        self._cases = _select_cases(cases, manifest.selected_case_ids)
        self._executor = executor
        self._judge = judge
        self._output_path = output_path
        self._manifest_path = manifest_path or output_path.with_suffix(".manifest.json")
        self._summary_path = summary_path or output_path.with_suffix(".summary.json")
        self._report_judge = report_judge
        if judge.grader != manifest.grader_version:
            raise ValueError("judge grader does not match the evaluation manifest")
        if judge.model_name != manifest.judge_model:
            raise ValueError("judge model does not match the evaluation manifest")
        if report_judge is None:
            if manifest.report_grader_version is not None:
                raise ValueError("manifest requires a report judge")
        elif (
            report_judge.grader != manifest.report_grader_version
            or report_judge.model_name != manifest.report_judge_model
        ):
            raise ValueError("report judge does not match the evaluation manifest")

    def run(self) -> EvaluationSummary:
        """Advance every non-completed case and persist deterministic metrics."""
        with _experiment_lock(self._output_path.parent):
            return self._run_locked()

    def _run_locked(self) -> EvaluationSummary:
        create_or_validate_manifest(self._manifest_path, self._manifest)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        records = _load_evaluation_records(self._output_path, self._manifest)

        for case in self._cases:
            previous = records.get(case.id)
            if previous is not None and previous.status == "completed":
                continue
            if previous is not None and previous.status == "judge_error":
                execution = self._executor.load(case, self._manifest)
            else:
                execution = self._executor.execute(case, self._manifest)
            record = self._record(case, execution)
            if record != previous:
                _append_record_atomic(self._output_path, record)
            records[case.id] = record

        summary = _summarize(self._manifest, records)
        _write_summary(self._summary_path, summary)
        return summary

    def _record(
        self,
        case: BenchmarkCase,
        execution: CaseExecution,
    ) -> EvaluationRecord:
        trajectory = TrajectorySnapshot.from_execution(execution)
        if execution.status != "completed" or execution.result is None:
            return EvaluationRecord(
                experiment_id=self._manifest.experiment_id,
                benchmark=self._manifest.benchmark,
                case_id=case.id,
                run_id=execution.run_id,
                status=execution.status,
                trajectory=trajectory,
                error_type=execution.error_type,
            )
        if execution.result.status != "completed":
            judgment = Judgment(
                correct=False,
                score=0.0,
                reason=(
                    "research did not pass the quality gate: "
                    f"{execution.result.stop_reason}"
                ),
                grader="research-quality-gate-v1",
            )
            return EvaluationRecord(
                experiment_id=self._manifest.experiment_id,
                benchmark=self._manifest.benchmark,
                case_id=case.id,
                run_id=execution.run_id,
                status="completed",
                trajectory=trajectory,
                judgment=JudgmentSnapshot.from_judgment(judgment),
            )
        try:
            judgment = self._judge.judge(
                question=case.question,
                reference=case.answer,
                prediction=execution.result.raw_report,
            )
            if judgment.grader != self._manifest.grader_version:
                raise ValueError("judge returned an unexpected grader identity")
            snapshot = JudgmentSnapshot.from_judgment(judgment)
            report_snapshot = None
            if self._report_judge is not None:
                report_judgment = self._report_judge.judge(
                    report=execution.result.raw_report,
                    evidence=execution.result.evidence,
                    citation_checks=execution.result.citation_checks,
                )
                if report_judgment.grader != self._manifest.report_grader_version:
                    raise ValueError("report judge returned an unexpected grader identity")
                report_snapshot = ReportJudgmentSnapshot.from_judgment(report_judgment)
        except Exception as error:  # noqa: BLE001 - judging must be independently resumable.
            return EvaluationRecord(
                experiment_id=self._manifest.experiment_id,
                benchmark=self._manifest.benchmark,
                case_id=case.id,
                run_id=execution.run_id,
                status="judge_error",
                trajectory=trajectory,
                error_type=type(error).__name__,
            )
        return EvaluationRecord(
            experiment_id=self._manifest.experiment_id,
            benchmark=self._manifest.benchmark,
            case_id=case.id,
            run_id=execution.run_id,
            status="completed",
            trajectory=trajectory,
            judgment=snapshot,
            report_judgment=report_snapshot,
        )


def _default_request(
    case: BenchmarkCase,
    manifest: EvaluationManifest,
) -> ResearchRequest:
    return ResearchRequest(
        topic=case.question,
        language=manifest.language,
        budget=manifest.budget,
        min_sources=manifest.min_sources,
        require_citations=manifest.require_citations,
    )


def _validate_request(
    request: ResearchRequest,
    case: BenchmarkCase,
    manifest: EvaluationManifest,
) -> None:
    if request.topic != case.question:
        raise ValueError("evaluation request topic must match the benchmark case")
    expected = _default_request(case, manifest)
    if request != expected:
        raise ValueError("evaluation request configuration does not match the manifest")


def _case_execution(case_id: str, outcome: RuntimeOutcome) -> CaseExecution:
    state = outcome.state
    if state.status == "completed" and outcome.result is not None:
        status: Literal["completed", "waiting", "research_error"] = "completed"
        error_type = None
    elif state.status == "waiting_for_human":
        status = "waiting"
        error_type = state.error_type
    else:
        status = "research_error"
        error_type = state.error_type or (
            "MissingResearchResult" if state.status == "completed" else state.status
        )
    return CaseExecution(
        case_id=case_id,
        run_id=state.run_id,
        status=status,
        state=state,
        result=outcome.result,
        error_type=error_type,
    )


def _execution_error(
    case_id: str,
    run_id: str,
    error: Exception,
    *,
    state: RunState | None = None,
) -> CaseExecution:
    return CaseExecution(
        case_id=case_id,
        run_id=run_id,
        status="research_error",
        state=state,
        error_type=type(error).__name__,
    )


def _select_cases(
    cases: Iterable[BenchmarkCase],
    selected_ids: tuple[str, ...],
) -> tuple[BenchmarkCase, ...]:
    available: dict[str, BenchmarkCase] = {}
    for case in cases:
        if case.id in available:
            raise ValueError("benchmark case IDs must be unique")
        available[case.id] = case
    missing = [case_id for case_id in selected_ids if case_id not in available]
    if missing:
        raise ValueError(f"selected benchmark cases are missing: {', '.join(missing)}")
    return tuple(available[case_id] for case_id in selected_ids)


def _load_evaluation_records(
    path: Path,
    manifest: EvaluationManifest,
) -> dict[str, EvaluationRecord]:
    if not path.exists():
        return {}
    selected_ids = set(manifest.selected_case_ids)
    records: dict[str, EvaluationRecord] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError(f"evaluation record on line {line_number} must be an object")
        record = _record_from_dict(payload)
        if (
            record.experiment_id != manifest.experiment_id
            or record.benchmark != manifest.benchmark
            or record.case_id not in selected_ids
            or record.run_id
            != deterministic_case_run_id(manifest.experiment_id, record.case_id)
        ):
            raise ValueError("existing evaluation record does not match the manifest")
        records[record.case_id] = record
    return records


def _record_from_dict(payload: Mapping[str, object]) -> EvaluationRecord:
    trajectory_payload = payload.get("trajectory")
    if not isinstance(trajectory_payload, Mapping):
        raise TypeError("evaluation trajectory must be an object")
    metrics_payload = trajectory_payload.get("metrics")
    effects_payload = trajectory_payload.get("effects")
    if not isinstance(metrics_payload, Mapping) or not isinstance(effects_payload, list):
        raise TypeError("evaluation trajectory metrics/effects have invalid types")
    trajectory = TrajectorySnapshot(
        run_status=str(trajectory_payload["run_status"]),
        current_step=str(trajectory_payload["current_step"]),
        completed_steps=tuple(str(item) for item in trajectory_payload["completed_steps"]),
        research_status=str(trajectory_payload["research_status"]),
        stop_reason=str(trajectory_payload["stop_reason"]),
        effects=tuple(EffectSnapshot(**effect) for effect in effects_payload),
        metrics=TrajectoryMetrics(**dict(metrics_payload)),
        schema_version=int(trajectory_payload["schema_version"]),
    )
    judgment_payload = payload.get("judgment")
    if judgment_payload is not None and not isinstance(judgment_payload, Mapping):
        raise TypeError("evaluation judgment must be an object or null")
    judgment = (
        JudgmentSnapshot(**dict(judgment_payload))
        if judgment_payload is not None
        else None
    )
    report_payload = payload.get("report_judgment")
    if report_payload is not None and not isinstance(report_payload, Mapping):
        raise TypeError("evaluation report_judgment must be an object or null")
    if report_payload is None:
        report_judgment = None
    else:
        raw_scores = report_payload.get("scores")
        if not isinstance(raw_scores, list):
            raise TypeError("report_judgment scores must be an array")
        report_judgment = ReportJudgmentSnapshot(
            overall_score=report_payload["overall_score"],  # type: ignore[arg-type]
            passed=report_payload["passed"],  # type: ignore[arg-type]
            grader=str(report_payload["grader"]),
            scores=tuple(RubricScoreSnapshot(**score) for score in raw_scores),
        )
    return EvaluationRecord(
        experiment_id=str(payload["experiment_id"]),
        benchmark=str(payload["benchmark"]),
        case_id=str(payload["case_id"]),
        run_id=str(payload["run_id"]),
        status=str(payload["status"]),  # type: ignore[arg-type]
        trajectory=trajectory,
        judgment=judgment,
        report_judgment=report_judgment,
        error_type=(str(payload["error_type"]) if payload.get("error_type") else None),
        schema_version=int(payload["schema_version"]),
    )


def _summarize(
    manifest: EvaluationManifest,
    records: Mapping[str, EvaluationRecord],
) -> EvaluationSummary:
    selected = [records[case_id] for case_id in manifest.selected_case_ids]
    total = len(selected)
    completed_records = [record for record in selected if record.status == "completed"]
    report_records = [
        record for record in completed_records if record.report_judgment is not None
    ]

    def average(value: Callable[[EvaluationRecord], float]) -> float:
        return sum(value(record) for record in selected) / total if total else 0.0

    return EvaluationSummary(
        experiment_id=manifest.experiment_id,
        benchmark=manifest.benchmark,
        total=total,
        completed=sum(record.status == "completed" for record in selected),
        correct=sum(record.correct for record in selected),
        research_errors=sum(record.status == "research_error" for record in selected),
        judge_errors=sum(record.status == "judge_error" for record in selected),
        waiting=sum(record.status == "waiting" for record in selected),
        accuracy=average(lambda record: float(record.correct)),
        accuracy_on_completed=(
            sum(record.correct for record in completed_records) / len(completed_records)
            if completed_records
            else None
        ),
        average_score=average(
            lambda record: record.judgment.score if record.judgment is not None else 0.0
        ),
        average_score_on_completed=(
            sum(
                record.judgment.score
                for record in completed_records
                if record.judgment is not None
            )
            / len(completed_records)
            if completed_records
            else None
        ),
        average_tool_calls=average(
            lambda record: float(record.trajectory.metrics.tool_calls)
        ),
        average_sources=average(
            lambda record: float(record.trajectory.metrics.source_count)
        ),
        average_tokens=average(
            lambda record: float(record.trajectory.metrics.total_tokens)
        ),
        reports_evaluated=len(report_records),
        reports_passed=sum(record.report_judgment.passed for record in report_records),
        average_report_score=(
            sum(record.report_judgment.overall_score for record in report_records)
            / len(report_records)
            if report_records
            else None
        ),
    )


def _write_summary(path: Path, summary: EvaluationSummary) -> None:
    serialized = json.dumps(
        asdict(summary),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    _atomic_replace_private(path, serialized)


def _append_record_atomic(path: Path, record: EvaluationRecord) -> None:
    """Append logically while atomically replacing the durable JSONL snapshot."""
    serialized = json.dumps(
        asdict(record),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    _atomic_replace_private(path, existing + serialized)


def _atomic_replace_private(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _write_private_temporary(path, content)
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_private_temporary(path: Path, content: str) -> Path:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                descriptor = -1
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _experiment_lock(directory: Path) -> Iterator[None]:
    """Prevent concurrent writers from losing evaluation records."""
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".evaluation.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    os.chmod(lock_path, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"evaluation experiment is already running: {directory}"
            ) from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

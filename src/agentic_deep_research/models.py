"""Domain models for a bounded web-research run."""

from dataclasses import dataclass, field, replace
from typing import Literal

VerificationStatus = Literal[
    "unverified",
    "supported",
    "unsupported",
    "uncertain",
]
ContextPurpose = Literal["worker", "planner"]


@dataclass(frozen=True)
class BudgetSnapshot:
    """Dynamic global capacity at one supervisor decision point."""

    tool_calls_remaining: int
    research_steps_remaining: int

    def __post_init__(self) -> None:
        if self.tool_calls_remaining < 0:
            raise ValueError("tool_calls_remaining must not be negative")
        if self.research_steps_remaining < 0:
            raise ValueError("research_steps_remaining must not be negative")


@dataclass(frozen=True)
class ContextPack:
    """Bounded context plus observable selection metadata."""

    purpose: ContextPurpose
    text: str
    selected_evidence_ids: tuple[str, ...]
    omitted_evidence_count: int
    included_conflict_count: int
    omitted_conflict_count: int
    used_chars: int

    def __post_init__(self) -> None:
        if self.purpose not in {"worker", "planner"}:
            raise ValueError("unsupported context purpose")
        if len(set(self.selected_evidence_ids)) != len(self.selected_evidence_ids):
            raise ValueError("selected_evidence_ids must be unique")
        if self.omitted_evidence_count < 0:
            raise ValueError("omitted_evidence_count must not be negative")
        if self.included_conflict_count < 0:
            raise ValueError("included_conflict_count must not be negative")
        if self.omitted_conflict_count < 0:
            raise ValueError("omitted_conflict_count must not be negative")
        if self.used_chars != len(self.text):
            raise ValueError("used_chars must equal len(text)")


@dataclass(frozen=True)
class ResearchBudget:
    """Hard limits passed to the model runtime."""

    max_tool_calls: int = 8
    max_output_tokens: int = 20_000
    max_research_steps: int = 4
    max_parallel_workers: int = 2
    max_context_chars: int = 8_000
    max_verification_tool_calls: int = 2
    max_revision_rounds: int = 2

    def __post_init__(self) -> None:
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")
        if self.max_research_steps < 1:
            raise ValueError("max_research_steps must be at least 1")
        if self.max_parallel_workers < 1:
            raise ValueError("max_parallel_workers must be at least 1")
        if self.max_context_chars < 1:
            raise ValueError("max_context_chars must be at least 1")
        if self.max_verification_tool_calls < 0:
            raise ValueError("max_verification_tool_calls must not be negative")
        if self.max_revision_rounds < 0:
            raise ValueError("max_revision_rounds must not be negative")


@dataclass(frozen=True)
class ResearchRequest:
    """A topic and the constraints for researching it."""

    topic: str
    language: str = "the same language as the request"
    budget: ResearchBudget = field(default_factory=ResearchBudget)
    min_sources: int = 2
    require_citations: bool = True

    def __post_init__(self) -> None:
        if not self.topic.strip():
            raise ValueError("research topic must not be empty")
        if not self.language.strip():
            raise ValueError("report language must not be empty")
        if self.min_sources < 1:
            raise ValueError("min_sources must be at least 1")


@dataclass(frozen=True)
class Source:
    """A web source consulted during research."""

    title: str
    url: str
    quality: str = "unknown"
    id: str = field(default="", compare=False)
    canonical_url: str = field(default="", compare=False)


@dataclass(frozen=True)
class ResearchQuestion:
    """One independently researchable question in an adaptive plan."""

    id: str
    question: str
    rationale: str = ""
    priority: int = 1


@dataclass(frozen=True)
class ResearchPlan:
    """An explicit, inspectable plan for answering a research request."""

    objective: str
    questions: tuple[ResearchQuestion, ...]
    revision: int = 0


@dataclass(frozen=True)
class Citation:
    """A source annotation attached to a span of the generated report."""

    source: Source
    start_index: int
    end_index: int
    evidence_id: str = field(default="", compare=False)


@dataclass(frozen=True)
class CitationClaim:
    """A final-report claim paired with its controlled source."""

    claim: str
    source: Source
    marker: str
    claim_id: str = ""
    evidence_id: str = field(default="", compare=False)


@dataclass(frozen=True)
class CitationCheck:
    """Semantic support judgment for one claim-source pair."""

    claim: str
    source: Source
    status: str
    reason: str
    claim_id: str = ""
    evidence_id: str = field(default="", compare=False)


@dataclass(frozen=True)
class Evidence:
    """A report claim linked to the web source cited for it."""

    claim: str
    source: Source
    question_id: str = ""
    excerpt: str = field(default="", compare=False)
    confidence: str = "medium"
    id: str = field(default="", compare=False)
    verification_status: VerificationStatus = field(
        default="unverified",
        compare=False,
    )
    origin_artifact_id: str = field(default="", compare=False)
    origin_start_index: int = field(default=-1, compare=False)
    origin_end_index: int = field(default=-1, compare=False)
    corroboration_count: int = field(default=1, compare=False)


@dataclass(frozen=True)
class EvidenceConflict:
    """A disagreement that must remain visible in the final research state."""

    question_id: str
    description: str
    sources: tuple[Source, ...] = ()


@dataclass(frozen=True)
class EvidenceLedger:
    """Immutable provenance snapshot accumulated during one research run."""

    schema_version: int = 1
    sources: tuple[Source, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    conflicts: tuple[EvidenceConflict, ...] = ()
    checks: tuple[CitationCheck, ...] = ()

    @property
    def entries(self) -> tuple[Evidence, ...]:
        """Return evidence records using ledger terminology."""
        return self.evidence

    def with_checks(self, checks: tuple[CitationCheck, ...]) -> "EvidenceLedger":
        """Return a snapshot with citation checks linked to evidence records."""
        status_priority = {
            "supported": 1,
            "uncertain": 2,
            "unsupported": 3,
        }
        statuses: dict[str, str] = {}
        for check in checks:
            if not check.evidence_id:
                continue
            status = (
                check.status
                if check.status in status_priority
                else "uncertain"
            )
            current = statuses.get(check.evidence_id)
            if current is None or status_priority[status] > status_priority[current]:
                statuses[check.evidence_id] = status

        evidence = tuple(
            replace(
                item,
                verification_status=statuses.get(
                    item.id,
                    item.verification_status,
                ),
            )
            for item in self.evidence
        )
        return replace(self, evidence=evidence, checks=checks)


@dataclass(frozen=True)
class ResearchPacket:
    """Normalized output of ingesting one worker run into the evidence ledger."""

    question_id: str
    answer: str
    evidence: tuple[Evidence, ...] = ()
    conflicts: tuple[EvidenceConflict, ...] = ()
    artifact_id: str = ""
    status: str = "completed"
    stop_reason: str = "completed"


@dataclass(frozen=True)
class ResearchStep:
    """An observable web action performed during research."""

    action: str
    detail: str
    kind: str = "tool"


@dataclass(frozen=True)
class TokenUsage:
    """Token counts reported by the model provider."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class AgentRun:
    """Provider-neutral output returned by an agent runner."""

    report: str
    sources: tuple[Source, ...] = ()
    citations: tuple[Citation, ...] = ()
    trace: tuple[ResearchStep, ...] = ()
    status: str = "completed"
    stop_reason: str = "completed"
    usage: TokenUsage = field(default_factory=TokenUsage)
    conflicts: tuple[EvidenceConflict, ...] = ()
    evidence: tuple[Evidence, ...] = ()


@dataclass(frozen=True)
class PlanningRun:
    """A planner response plus provider execution metadata."""

    plan: ResearchPlan
    status: str = "completed"
    stop_reason: str = "completed"
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass(frozen=True)
class ReportDraft:
    """A synthesized report that cites controlled source markers."""

    report: str
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass(frozen=True)
class ReportCritique:
    """Structured quality feedback about a synthesized report."""

    coverage_gaps: tuple[str, ...] = ()
    unsupported_claims: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    clarity_issues: tuple[str, ...] = ()
    revision_instructions: tuple[str, ...] = ()
    needs_more_research: bool = False
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass(frozen=True)
class CitationVerification:
    """All citation judgments and observable verifier execution metadata."""

    checks: tuple[CitationCheck, ...] = ()
    trace: tuple[ResearchStep, ...] = ()
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass(frozen=True)
class ResearchFinding:
    """The isolated output produced for one planned research question."""

    question_id: str
    question: str
    answer: str
    sources: tuple[Source, ...] = ()
    citations: tuple[Citation, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    status: str = "completed"
    stop_reason: str = "completed"
    conflicts: tuple[EvidenceConflict, ...] = ()


@dataclass(frozen=True)
class ResearchArtifact:
    """A full research product kept outside the model's working context."""

    id: str
    kind: str
    content: str
    question_id: str = ""


@dataclass(frozen=True)
class ResearchResult:
    """Public result of a completed or bounded research workflow."""

    topic: str
    report: str
    raw_report: str
    sources: tuple[Source, ...]
    citations: tuple[Citation, ...]
    evidence: tuple[Evidence, ...]
    trace: tuple[ResearchStep, ...]
    status: str
    stop_reason: str
    usage: TokenUsage
    plan: ResearchPlan | None = None
    findings: tuple[ResearchFinding, ...] = ()
    conflicts: tuple[EvidenceConflict, ...] = ()
    artifacts: tuple[ResearchArtifact, ...] = ()
    critique: ReportCritique | None = None
    citation_checks: tuple[CitationCheck, ...] = ()
    revision_count: int = 0
    ledger: EvidenceLedger | None = None

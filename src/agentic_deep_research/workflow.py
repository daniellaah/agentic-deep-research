"""Public entry point for the adaptive, evidence-aware research workflow."""

from .models import (
    AgentRun,
    Citation,
    Evidence,
    EvidenceConflict,
    ResearchFinding,
    ResearchRequest,
    ResearchResult,
    Source,
    TokenUsage,
)
from .planning import ResearchPlanner, TopicPlanner
from .runner import AgentRunner
from .supervisor import ResearchSupervisor


def run_research(
    request: str | ResearchRequest,
    *,
    runner: AgentRunner,
    planner: ResearchPlanner | None = None,
) -> ResearchResult:
    """Research a topic on the web within an explicit execution budget."""
    if isinstance(request, str):
        request = ResearchRequest(topic=request)

    supervision = ResearchSupervisor(
        planner=planner or TopicPlanner(),
        runner=runner,
    ).run(request)
    plan = supervision.plan
    runs = [(question.id, question.question, run) for question, run in supervision.runs]

    raw_report, citations, findings = _combine_findings(
        request.topic.strip(),
        runs,
        supervision.sources,
        supervision.evidence,
        supervision.conflicts,
    )
    sources = supervision.sources
    evidence = supervision.evidence
    status, stop_reason = _quality_status(
        request,
        supervision.status,
        supervision.stop_reason,
        raw_report,
        evidence,
        sources,
    )

    return ResearchResult(
        topic=request.topic.strip(),
        report=_render_citations(raw_report, citations),
        raw_report=raw_report,
        sources=sources,
        citations=citations,
        evidence=evidence,
        trace=supervision.trace,
        status=status,
        stop_reason=stop_reason,
        usage=_sum_usage(supervision.planning_usage, *(run.usage for _, _, run in runs)),
        plan=plan,
        findings=findings,
        conflicts=supervision.conflicts,
        artifacts=supervision.artifacts,
    )


def _combine_findings(
    topic: str,
    runs: list[tuple[str, str, AgentRun]],
    all_sources: tuple[Source, ...],
    all_evidence: tuple[Evidence, ...],
    all_conflicts: tuple[EvidenceConflict, ...],
) -> tuple[str, tuple[Citation, ...], tuple[ResearchFinding, ...]]:
    sources_by_url = {source.url: source for source in all_sources}
    if len(runs) == 1 and runs[0][1] == topic:
        question_id, question, run = runs[0]
        sources = _canonical_sources(run.sources, sources_by_url)
        citations = _canonical_citations(run.citations, sources_by_url)
        evidence = _for_question(all_evidence, question_id)
        conflicts = _conflicts_for_question(all_conflicts, question_id)
        finding = _finding(
            question_id,
            question,
            run,
            sources,
            citations,
            evidence,
            conflicts,
        )
        return run.report, citations, (finding,)

    report = f"# {topic}\n"
    citations: list[Citation] = []
    findings: list[ResearchFinding] = []
    for question_id, question, run in runs:
        sources = _canonical_sources(run.sources, sources_by_url)
        run_citations = _canonical_citations(run.citations, sources_by_url)
        report += f"\n## {question}\n\n"
        offset = len(report)
        report += run.report.rstrip() + "\n"
        shifted = tuple(
            Citation(
                source=citation.source,
                start_index=citation.start_index + offset,
                end_index=citation.end_index + offset,
            )
            for citation in run_citations
        )
        citations.extend(shifted)
        evidence = _for_question(all_evidence, question_id)
        conflicts = _conflicts_for_question(all_conflicts, question_id)
        findings.append(
            _finding(
                question_id,
                question,
                run,
                sources,
                run_citations,
                evidence,
                conflicts,
            )
        )
    return report.rstrip(), tuple(citations), tuple(findings)


def _finding(
    question_id: str,
    question: str,
    run: AgentRun,
    sources: tuple[Source, ...],
    citations: tuple[Citation, ...],
    evidence: tuple[Evidence, ...],
    conflicts: tuple[EvidenceConflict, ...],
) -> ResearchFinding:
    return ResearchFinding(
        question_id=question_id,
        question=question,
        answer=run.report,
        sources=sources,
        citations=citations,
        evidence=evidence,
        status=run.status,
        stop_reason=run.stop_reason,
        conflicts=conflicts,
    )


def _canonical_sources(
    sources: tuple[Source, ...],
    sources_by_url: dict[str, Source],
) -> tuple[Source, ...]:
    return tuple(sources_by_url.get(source.url, source) for source in sources)


def _canonical_citations(
    citations: tuple[Citation, ...],
    sources_by_url: dict[str, Source],
) -> tuple[Citation, ...]:
    normalized: list[Citation] = []
    seen: set[tuple[str, int, int]] = set()
    for citation in citations:
        key = (citation.source.url, citation.start_index, citation.end_index)
        if key in seen:
            continue
        normalized.append(
            Citation(
                source=sources_by_url.get(citation.source.url, citation.source),
                start_index=citation.start_index,
                end_index=citation.end_index,
            )
        )
        seen.add(key)
    return tuple(normalized)


def _for_question(
    evidence: tuple[Evidence, ...],
    question_id: str,
) -> tuple[Evidence, ...]:
    return tuple(item for item in evidence if item.question_id == question_id)


def _conflicts_for_question(
    conflicts: tuple[EvidenceConflict, ...],
    question_id: str,
) -> tuple[EvidenceConflict, ...]:
    return tuple(item for item in conflicts if item.question_id == question_id)


def _sum_usage(*items: TokenUsage) -> TokenUsage:
    return TokenUsage(
        input_tokens=sum(item.input_tokens for item in items),
        output_tokens=sum(item.output_tokens for item in items),
        total_tokens=sum(item.total_tokens for item in items),
    )


def _quality_status(
    request: ResearchRequest,
    provider_status: str,
    provider_stop_reason: str,
    report: str,
    evidence: tuple[Evidence, ...],
    sources: tuple[object, ...],
) -> tuple[str, str]:
    if provider_status != "completed":
        return provider_status, provider_stop_reason
    if not report.strip():
        return "needs_review", "empty_report"
    if len(sources) < request.min_sources:
        return "needs_review", "insufficient_sources"
    if request.require_citations and not evidence:
        return "needs_review", "missing_citations"
    return "completed", "completed"


def _render_citations(report: str, citations: tuple[Citation, ...]) -> str:
    rendered = report
    for citation in sorted(citations, key=lambda item: item.start_index, reverse=True):
        if not _valid_citation(report, citation):
            continue
        link = f"[{citation.source.title}]({citation.source.url})"
        rendered = rendered[: citation.start_index] + link + rendered[citation.end_index :]
    return rendered


def _valid_citation(report: str, citation: Citation) -> bool:
    return 0 <= citation.start_index < citation.end_index <= len(report) and bool(
        citation.source.url
    )

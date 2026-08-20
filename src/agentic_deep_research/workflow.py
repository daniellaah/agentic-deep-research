"""Public entry point for the adaptive, evidence-aware research workflow."""

import re
from dataclasses import dataclass, replace

from .evidence import EvidenceStore
from .models import (
    AgentRun,
    Citation,
    CitationCheck,
    CitationClaim,
    Evidence,
    EvidenceConflict,
    PlanningRun,
    ReportCritique,
    ResearchArtifact,
    ResearchFinding,
    ResearchPlan,
    ResearchQuestion,
    ResearchRequest,
    ResearchResult,
    ResearchStep,
    Source,
    TokenUsage,
)
from .planning import ResearchPlanner, TopicPlanner
from .reporting import ReportAgent
from .runner import AgentRunner
from .supervisor import ResearchSupervisor, SupervisorResult


@dataclass(frozen=True)
class _ResearchMaterial:
    plan: ResearchPlan
    fallback_report: str
    fallback_citations: tuple[Citation, ...]
    findings: tuple[ResearchFinding, ...]
    sources: tuple[Source, ...]
    evidence: tuple[Evidence, ...]
    conflicts: tuple[EvidenceConflict, ...]
    status: str
    stop_reason: str
    usage: TokenUsage


class _GapPlanner:
    def __init__(self, questions: tuple[str, ...], round_number: int) -> None:
        self._questions = questions
        self._round_number = round_number

    def plan(
        self,
        *,
        request: ResearchRequest,
        context: str,
        completed_questions: tuple[str, ...],
        max_questions: int,
        revision: int,
    ) -> PlanningRun:
        del context, completed_questions, revision
        questions = tuple(
            ResearchQuestion(
                id=f"g{self._round_number}q{index}",
                question=question,
                rationale="Resolve a report-quality or citation gap.",
                priority=index,
            )
            for index, question in enumerate(self._questions[:max_questions], start=1)
        )
        return PlanningRun(
            plan=ResearchPlan(
                objective=request.topic.strip(),
                questions=questions,
                revision=self._round_number,
            )
        )


def run_research(
    request: str | ResearchRequest,
    *,
    runner: AgentRunner,
    planner: ResearchPlanner | None = None,
    report_agent: ReportAgent | None = None,
) -> ResearchResult:
    """Research a topic on the web within an explicit execution budget."""
    if isinstance(request, str):
        request = ResearchRequest(topic=request)

    initial_supervision = ResearchSupervisor(
        planner=planner or TopicPlanner(),
        runner=runner,
    ).run(request)
    supervisions = [initial_supervision]
    material = _build_material(request.topic.strip(), supervisions)
    trace = list(initial_supervision.trace)
    artifacts = list(initial_supervision.artifacts)
    reporting_usage: list[TokenUsage] = []
    critique = None
    citation_checks = ()
    revision_count = 0
    pipeline_stop_reason: str | None = None

    if report_agent is None:
        raw_report = material.fallback_report
        citations = material.fallback_citations
    else:
        draft = report_agent.write(
            request=request,
            findings=material.findings,
            evidence=material.evidence,
            conflicts=material.conflicts,
            sources=material.sources,
        )
        raw_report = draft.report
        citations = _source_marker_citations(raw_report, material.sources)
        reporting_usage.append(draft.usage)
        artifacts.append(
            ResearchArtifact(
                id="report:draft",
                kind="report_draft",
                content=raw_report,
            )
        )
        trace.append(
            ResearchStep(
                action="report_written",
                detail=f"citations={len(citations)}",
                kind="control",
            )
        )
        verification_tool_calls = 0
        while True:
            citations = _source_marker_citations(raw_report, material.sources)
            critique = report_agent.critique(
                request=request,
                report=raw_report,
                findings=material.findings,
                evidence=material.evidence,
                conflicts=material.conflicts,
                sources=material.sources,
            )
            reporting_usage.append(critique.usage)
            trace.append(
                ResearchStep(
                    action="report_critiqued",
                    detail=(
                        f"coverage_gaps={len(critique.coverage_gaps)}, "
                        f"unsupported_claims={len(critique.unsupported_claims)}"
                    ),
                    kind="control",
                )
            )

            remaining_verification_calls = max(
                0,
                request.budget.max_verification_tool_calls - verification_tool_calls,
            )
            verification = report_agent.verify(
                request=request,
                claims=_citation_claims(raw_report, citations),
                max_tool_calls=remaining_verification_calls,
            )
            citation_checks = verification.checks
            reporting_usage.append(verification.usage)
            trace.extend(verification.trace)
            verification_tool_calls += sum(step.kind == "tool" for step in verification.trace)
            trace.append(_verification_step(citation_checks))

            if not _needs_revision(critique, citation_checks, len(citations)):
                break
            if request.budget.max_revision_rounds == 0:
                break
            if revision_count >= request.budget.max_revision_rounds:
                pipeline_stop_reason = "max_revision_rounds"
                break

            gap_questions = _gap_questions(critique, citation_checks)
            if gap_questions:
                remaining_steps = request.budget.max_research_steps - sum(
                    len(item.runs) for item in supervisions
                )
                remaining_tool_calls = request.budget.max_tool_calls - sum(
                    step.kind == "tool" for item in supervisions for step in item.trace
                )
                if remaining_steps < 1 or remaining_tool_calls < 1:
                    pipeline_stop_reason = "research_budget_exhausted"
                    break
                selected_gaps = gap_questions[:remaining_steps]
                trace.append(
                    ResearchStep(
                        action="gap_search_started",
                        detail=f"questions={len(selected_gaps)}",
                        kind="control",
                    )
                )
                gap_request = replace(
                    request,
                    budget=replace(
                        request.budget,
                        max_tool_calls=remaining_tool_calls,
                        max_research_steps=remaining_steps,
                    ),
                )
                gap_supervision = ResearchSupervisor(
                    planner=_GapPlanner(selected_gaps, revision_count + 1),
                    runner=runner,
                ).run(gap_request)
                supervisions.append(gap_supervision)
                trace.extend(gap_supervision.trace)
                artifacts.extend(gap_supervision.artifacts)
                material = _build_material(request.topic.strip(), supervisions)

            revised = report_agent.revise(
                request=request,
                report=raw_report,
                critique=critique,
                verification=verification,
                findings=material.findings,
                evidence=material.evidence,
                conflicts=material.conflicts,
                sources=material.sources,
            )
            revision_count += 1
            raw_report = revised.report
            reporting_usage.append(revised.usage)
            artifacts.append(
                ResearchArtifact(
                    id=f"report:revision:{revision_count}",
                    kind="report_revision",
                    content=raw_report,
                )
            )
            trace.append(
                ResearchStep(
                    action="report_revised",
                    detail=f"revision={revision_count}",
                    kind="control",
                )
            )

    citations = (
        material.fallback_citations
        if report_agent is None
        else _source_marker_citations(raw_report, material.sources)
    )
    status, stop_reason = _quality_status(
        request,
        material.status,
        material.stop_reason,
        raw_report,
        material.evidence,
        material.sources,
        citations,
    )
    if status == "completed" and pipeline_stop_reason is not None:
        status, stop_reason = "needs_review", pipeline_stop_reason
    elif status == "completed" and report_agent is not None:
        if any(item.status == "unsupported" for item in citation_checks):
            status, stop_reason = "needs_review", "unsupported_citations"
        elif len(citation_checks) < len(citations) or any(
            item.status == "uncertain" for item in citation_checks
        ):
            status, stop_reason = "needs_review", "unverified_citations"
        elif critique is not None and (critique.needs_more_research or critique.coverage_gaps):
            status, stop_reason = "needs_review", "coverage_gaps"
        elif critique is not None and critique.unsupported_claims:
            status, stop_reason = "needs_review", "unsupported_claims"
        elif critique is not None and (
            critique.contradictions or critique.clarity_issues or critique.revision_instructions
        ):
            status, stop_reason = "needs_review", "revision_required"

    return ResearchResult(
        topic=request.topic.strip(),
        report=_render_citations(raw_report, citations),
        raw_report=raw_report,
        sources=material.sources,
        citations=citations,
        evidence=material.evidence,
        trace=tuple(trace),
        status=status,
        stop_reason=stop_reason,
        usage=_sum_usage(material.usage, *reporting_usage),
        plan=material.plan,
        findings=material.findings,
        conflicts=material.conflicts,
        artifacts=tuple(artifacts),
        critique=critique,
        citation_checks=citation_checks,
        revision_count=revision_count,
    )


def _build_material(
    topic: str,
    supervisions: list[SupervisorResult],
) -> _ResearchMaterial:
    evidence_store = EvidenceStore()
    runs: list[tuple[str, str, AgentRun]] = []
    questions: list[ResearchQuestion] = []
    usage: list[TokenUsage] = []
    status = "completed"
    stop_reason = "completed"
    for supervision in supervisions:
        questions.extend(supervision.plan.questions)
        usage.append(supervision.planning_usage)
        if supervision.status != "completed" and status == "completed":
            status = supervision.status
            stop_reason = supervision.stop_reason
        for question, run in supervision.runs:
            evidence_store.add_run(question.id, run)
            runs.append((question.id, question.question, run))
            usage.append(run.usage)

    fallback_report, fallback_citations, findings = _combine_findings(
        topic,
        runs,
        evidence_store.sources,
        evidence_store.evidence,
        evidence_store.conflicts,
    )
    return _ResearchMaterial(
        plan=ResearchPlan(
            objective=supervisions[-1].plan.objective,
            questions=tuple(questions),
            revision=max(item.plan.revision for item in supervisions),
        ),
        fallback_report=fallback_report,
        fallback_citations=fallback_citations,
        findings=findings,
        sources=evidence_store.sources,
        evidence=evidence_store.evidence,
        conflicts=evidence_store.conflicts,
        status=status,
        stop_reason=stop_reason,
        usage=_sum_usage(*usage),
    )


def _verification_step(checks: tuple[CitationCheck, ...]) -> ResearchStep:
    return ResearchStep(
        action="citations_verified",
        detail=(
            f"supported={sum(item.status == 'supported' for item in checks)}, "
            f"unsupported={sum(item.status == 'unsupported' for item in checks)}, "
            f"uncertain={sum(item.status == 'uncertain' for item in checks)}"
        ),
        kind="control",
    )


def _needs_revision(
    critique: ReportCritique,
    checks: tuple[CitationCheck, ...],
    citation_count: int,
) -> bool:
    return bool(
        critique.needs_more_research
        or critique.coverage_gaps
        or critique.unsupported_claims
        or critique.contradictions
        or critique.clarity_issues
        or critique.revision_instructions
        or len(checks) < citation_count
        or any(item.status != "supported" for item in checks)
    )


def _gap_questions(
    critique: ReportCritique,
    checks: tuple[CitationCheck, ...],
) -> tuple[str, ...]:
    candidates = [
        *critique.coverage_gaps,
        *critique.unsupported_claims,
        *(f"Resolve conflicting evidence: {item}" for item in critique.contradictions),
        *(
            f"Find reliable source support for: {item.claim}"
            for item in checks
            if item.status != "supported"
        ),
    ]
    questions: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.strip().casefold()
        if key and key not in seen:
            questions.append(candidate.strip())
            seen.add(key)
    return tuple(questions)


def _source_marker_citations(
    report: str,
    sources: tuple[Source, ...],
) -> tuple[Citation, ...]:
    citations: list[Citation] = []
    for match in re.finditer(r"\[S([1-9][0-9]*)\]", report):
        source_index = int(match.group(1)) - 1
        if source_index >= len(sources):
            continue
        citations.append(
            Citation(
                source=sources[source_index],
                start_index=match.start(),
                end_index=match.end(),
            )
        )
    return tuple(citations)


def _citation_claims(
    report: str,
    citations: tuple[Citation, ...],
) -> tuple[CitationClaim, ...]:
    return tuple(
        CitationClaim(
            claim=_claim_before(report, citation.start_index),
            source=citation.source,
            marker=report[citation.start_index : citation.end_index],
            claim_id=f"C{index}",
        )
        for index, citation in enumerate(citations, start=1)
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
    citations: tuple[Citation, ...],
) -> tuple[str, str]:
    if provider_status != "completed":
        return provider_status, provider_stop_reason
    if not report.strip():
        return "needs_review", "empty_report"
    if len(sources) < request.min_sources:
        return "needs_review", "insufficient_sources"
    if request.require_citations and (not evidence or not citations):
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


def _claim_before(report: str, citation_start: int) -> str:
    prefix = report[:citation_start].rstrip()
    if not prefix:
        return ""
    search_end = len(prefix) - 1 if prefix[-1] in ".!?。！？" else len(prefix)
    boundary = max(
        prefix.rfind("\n", 0, search_end),
        prefix.rfind(".", 0, search_end),
        prefix.rfind("!", 0, search_end),
        prefix.rfind("?", 0, search_end),
        prefix.rfind("。", 0, search_end),
        prefix.rfind("！", 0, search_end),
        prefix.rfind("？", 0, search_end),
    )
    return prefix[boundary + 1 :].strip()

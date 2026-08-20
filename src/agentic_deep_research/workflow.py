"""Public entry point for the adaptive, evidence-aware research workflow."""

import re
from dataclasses import dataclass, replace

from .evidence import canonicalize_url
from .models import (
    AgentRun,
    Citation,
    CitationCheck,
    CitationClaim,
    Evidence,
    EvidenceConflict,
    EvidenceLedger,
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

_CITATION_MARKER_PATTERN = re.compile(r"\[(E|S)([1-9][0-9]*)\]")
_MARKER_LIKE_PATTERN = re.compile(r"\[(?:E|S)[^\]\n]*\]")


@dataclass(frozen=True)
class _ResearchMaterial:
    plan: ResearchPlan
    fallback_report: str
    fallback_citations: tuple[Citation, ...]
    findings: tuple[ResearchFinding, ...]
    ledger: EvidenceLedger
    status: str
    stop_reason: str
    usage: TokenUsage

    @property
    def sources(self) -> tuple[Source, ...]:
        return self.ledger.sources

    @property
    def evidence(self) -> tuple[Evidence, ...]:
        return self.ledger.evidence

    @property
    def conflicts(self) -> tuple[EvidenceConflict, ...]:
        return self.ledger.conflicts


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
        citations = _ledger_marker_citations(raw_report, material.ledger)
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
            citations = _ledger_marker_citations(raw_report, material.ledger)
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
            claims = _citation_claims(raw_report, citations)
            verification = report_agent.verify(
                request=request,
                claims=claims,
                max_tool_calls=remaining_verification_calls,
            )
            citation_checks = _bind_citation_checks(verification.checks, claims)
            verification = replace(verification, checks=citation_checks)
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
        else _ledger_marker_citations(raw_report, material.ledger)
    )
    status, stop_reason = _quality_status(
        request,
        material.status,
        material.stop_reason,
        raw_report,
        material.ledger.evidence,
        material.ledger.sources,
        citations,
        has_unbound_markers=_has_unbound_markers(raw_report, material.ledger),
        has_unbound_citations=any(not item.evidence_id for item in citations),
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

    final_ledger = material.ledger.with_checks(citation_checks)
    final_findings = _ledger_findings(material.findings, final_ledger)
    return ResearchResult(
        topic=request.topic.strip(),
        report=_render_citations(raw_report, citations),
        raw_report=raw_report,
        sources=final_ledger.sources,
        citations=citations,
        evidence=final_ledger.evidence,
        trace=tuple(trace),
        status=status,
        stop_reason=stop_reason,
        usage=_sum_usage(material.usage, *reporting_usage),
        plan=material.plan,
        findings=final_findings,
        conflicts=final_ledger.conflicts,
        artifacts=tuple(artifacts),
        critique=critique,
        citation_checks=final_ledger.checks,
        revision_count=revision_count,
        ledger=final_ledger,
    )


def _build_material(
    topic: str,
    supervisions: list[SupervisorResult],
) -> _ResearchMaterial:
    ledger = _merge_supervision_ledgers(supervisions)
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
            runs.append((question.id, question.question, run))
            usage.append(run.usage)

    fallback_report, fallback_citations, findings = _combine_findings(
        topic,
        runs,
        ledger.sources,
        ledger.evidence,
        ledger.conflicts,
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
        ledger=ledger,
        status=status,
        stop_reason=stop_reason,
        usage=_sum_usage(*usage),
    )


def _merge_supervision_ledgers(
    supervisions: list[SupervisorResult],
) -> EvidenceLedger:
    """Merge immutable supervisor snapshots without regenerating stable IDs."""
    sources: list[Source] = []
    sources_by_key: dict[str, Source] = {}
    for supervision in supervisions:
        for source in supervision.ledger.sources:
            key = _source_key(source)
            if key not in sources_by_key:
                sources_by_key[key] = source
                sources.append(source)

    evidence: list[Evidence] = []
    evidence_keys: set[str] = set()
    conflicts: list[EvidenceConflict] = []
    conflict_keys: set[tuple[str, str, tuple[str, ...]]] = set()
    checks: list[CitationCheck] = []
    check_keys: set[tuple[str, str, str, str]] = set()
    schema_version = 1
    for supervision in supervisions:
        ledger = supervision.ledger
        schema_version = max(schema_version, ledger.schema_version)
        for item in ledger.evidence:
            key = item.id or "\n".join(
                (item.question_id, _claim_key(item.claim), _source_key(item.source))
            )
            if key in evidence_keys:
                continue
            evidence.append(
                replace(
                    item,
                    source=sources_by_key.get(_source_key(item.source), item.source),
                )
            )
            evidence_keys.add(key)
        for conflict in ledger.conflicts:
            normalized_sources = tuple(
                sources_by_key.get(_source_key(source), source)
                for source in conflict.sources
            )
            key = (
                conflict.question_id,
                conflict.description.strip().casefold(),
                tuple(sorted(_source_key(source) for source in normalized_sources)),
            )
            if key in conflict_keys:
                continue
            conflicts.append(replace(conflict, sources=normalized_sources))
            conflict_keys.add(key)
        for check in ledger.checks:
            key = (check.claim_id, check.evidence_id, check.status, check.reason)
            if key in check_keys:
                continue
            checks.append(check)
            check_keys.add(key)

    source_ids_by_claim: dict[str, set[str]] = {}
    for item in evidence:
        source_ids_by_claim.setdefault(_claim_key(item.claim), set()).add(
            _source_key(item.source)
        )
    evidence = [
        replace(
            item,
            confidence=(
                "high"
                if len(source_ids_by_claim[_claim_key(item.claim)]) >= 2
                else "medium"
            ),
            corroboration_count=len(source_ids_by_claim[_claim_key(item.claim)]),
        )
        for item in evidence
    ]
    return EvidenceLedger(
        schema_version=schema_version,
        sources=tuple(sources),
        evidence=tuple(evidence),
        conflicts=tuple(conflicts),
        checks=tuple(checks),
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


def _ledger_marker_citations(
    report: str,
    ledger: EvidenceLedger,
) -> tuple[Citation, ...]:
    citations: list[Citation] = []
    for match in _CITATION_MARKER_PATTERN.finditer(report):
        item_index = int(match.group(2)) - 1
        if item_index < 0:
            continue
        if match.group(1) == "E":
            if item_index >= len(ledger.evidence):
                continue
            evidence = ledger.evidence[item_index]
            source = evidence.source
            evidence_id = (
                evidence.id
                if _claim_matches_evidence(
                    _claim_before(report, match.start()),
                    evidence,
                )
                else ""
            )
        else:
            if item_index >= len(ledger.sources):
                continue
            source = ledger.sources[item_index]
            evidence_id = _matching_evidence_id(
                claim=_claim_before(report, match.start()),
                source=source,
                evidence=ledger.evidence,
            )
        citations.append(
            Citation(
                source=source,
                start_index=match.start(),
                end_index=match.end(),
                evidence_id=evidence_id,
            )
        )
    return tuple(citations)


def _has_unbound_markers(report: str, ledger: EvidenceLedger) -> bool:
    """Reject syntactically valid markers that do not bind to ledger evidence."""
    if any(
        _CITATION_MARKER_PATTERN.fullmatch(match.group()) is None
        for match in _MARKER_LIKE_PATTERN.finditer(report)
    ):
        return True
    for match in _CITATION_MARKER_PATTERN.finditer(report):
        item_index = int(match.group(2)) - 1
        if item_index < 0:
            return True
        if match.group(1) == "E":
            if item_index >= len(ledger.evidence):
                return True
            evidence = ledger.evidence[item_index]
            if not evidence.id or not _claim_matches_evidence(
                _claim_before(report, match.start()),
                evidence,
            ):
                return True
            continue
        if item_index >= len(ledger.sources):
            return True
        if not _matching_evidence_id(
            claim=_claim_before(report, match.start()),
            source=ledger.sources[item_index],
            evidence=ledger.evidence,
        ):
            return True
    return False


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
            evidence_id=citation.evidence_id,
        )
        for index, citation in enumerate(citations, start=1)
    )


def _bind_citation_checks(
    checks: tuple[CitationCheck, ...],
    claims: tuple[CitationClaim, ...],
) -> tuple[CitationCheck, ...]:
    """Bind exactly one conservative verifier judgment to every expected claim."""
    bound: list[CitationCheck] = []
    for claim in claims:
        exact = [
            check
            for check in checks
            if check.claim_id == claim.claim_id
            and check.evidence_id == claim.evidence_id
            and check.claim_id
            and check.evidence_id
            and _source_key(check.source) == _source_key(claim.source)
            and _claim_key(check.claim) == _claim_key(claim.claim)
        ]
        compatible = [
            check
            for check in checks
            if not check.claim_id
            and not check.evidence_id
            and _source_key(check.source) == _source_key(claim.source)
            and _claim_key(check.claim) == _claim_key(claim.claim)
        ]
        candidates = [*exact, *compatible]
        if len(candidates) != 1:
            reason = (
                "citation verifier returned conflicting checks"
                if len(candidates) > 1
                else "citation verifier omitted or mismatched this evidence binding"
            )
            bound.append(
                CitationCheck(
                    claim=claim.claim,
                    source=claim.source,
                    status="uncertain",
                    reason=reason,
                    claim_id=claim.claim_id,
                    evidence_id=claim.evidence_id,
                )
            )
            continue
        check = candidates[0]
        if check.status not in {"supported", "unsupported", "uncertain"}:
            bound.append(
                CitationCheck(
                    claim=claim.claim,
                    source=claim.source,
                    status="uncertain",
                    reason="citation verifier returned an invalid status",
                    claim_id=claim.claim_id,
                    evidence_id=claim.evidence_id,
                )
            )
            continue
        if not check.reason.strip():
            bound.append(
                CitationCheck(
                    claim=claim.claim,
                    source=claim.source,
                    status="uncertain",
                    reason="citation verifier returned no reason",
                    claim_id=claim.claim_id,
                    evidence_id=claim.evidence_id,
                )
            )
            continue
        bound.append(
            replace(
                check,
                claim=claim.claim,
                source=claim.source,
                claim_id=claim.claim_id,
                evidence_id=claim.evidence_id,
            )
        )
    return tuple(bound)


def _combine_findings(
    topic: str,
    runs: list[tuple[str, str, AgentRun]],
    all_sources: tuple[Source, ...],
    all_evidence: tuple[Evidence, ...],
    all_conflicts: tuple[EvidenceConflict, ...],
) -> tuple[str, tuple[Citation, ...], tuple[ResearchFinding, ...]]:
    sources_by_key = _source_lookup(all_sources)
    if len(runs) == 1 and runs[0][1] == topic:
        question_id, question, run = runs[0]
        sources = _canonical_sources(run.sources, sources_by_key)
        evidence = _for_question(all_evidence, question_id)
        citations = _bind_citations_to_evidence(
            run.report,
            _canonical_citations(run.citations, sources_by_key),
            evidence,
        )
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
        sources = _canonical_sources(run.sources, sources_by_key)
        evidence = _for_question(all_evidence, question_id)
        run_citations = _bind_citations_to_evidence(
            run.report,
            _canonical_citations(run.citations, sources_by_key),
            evidence,
        )
        report += f"\n## {question}\n\n"
        offset = len(report)
        report += run.report.rstrip() + "\n"
        shifted = tuple(
            Citation(
                source=citation.source,
                start_index=citation.start_index + offset,
                end_index=citation.end_index + offset,
                evidence_id=citation.evidence_id,
            )
            for citation in run_citations
        )
        citations.extend(shifted)
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
    sources_by_key: dict[str, Source],
) -> tuple[Source, ...]:
    return tuple(sources_by_key.get(_source_key(source), source) for source in sources)


def _ledger_findings(
    findings: tuple[ResearchFinding, ...],
    ledger: EvidenceLedger,
) -> tuple[ResearchFinding, ...]:
    """Project final ledger state back into the compatibility finding views."""
    sources_by_key = _source_lookup(ledger.sources)
    return tuple(
        replace(
            finding,
            sources=_canonical_sources(finding.sources, sources_by_key),
            citations=_canonical_citations(finding.citations, sources_by_key),
            evidence=_for_question(ledger.evidence, finding.question_id),
            conflicts=_conflicts_for_question(ledger.conflicts, finding.question_id),
        )
        for finding in findings
    )


def _canonical_citations(
    citations: tuple[Citation, ...],
    sources_by_key: dict[str, Source],
) -> tuple[Citation, ...]:
    normalized: list[Citation] = []
    seen: set[tuple[str, int, int]] = set()
    for citation in citations:
        source_key = _source_key(citation.source)
        key = (source_key, citation.start_index, citation.end_index)
        if key in seen:
            continue
        normalized.append(
            Citation(
                source=sources_by_key.get(source_key, citation.source),
                start_index=citation.start_index,
                end_index=citation.end_index,
                evidence_id=citation.evidence_id,
            )
        )
        seen.add(key)
    return tuple(normalized)


def _bind_citations_to_evidence(
    report: str,
    citations: tuple[Citation, ...],
    evidence: tuple[Evidence, ...],
) -> tuple[Citation, ...]:
    return tuple(
        replace(
            citation,
            evidence_id=_matching_evidence_id(
                claim=_claim_before(report, citation.start_index),
                source=citation.source,
                evidence=evidence,
            ),
        )
        for citation in citations
    )


def _matching_evidence_id(
    *,
    claim: str,
    source: Source,
    evidence: tuple[Evidence, ...],
) -> str:
    candidates = [
        item for item in evidence if _source_key(item.source) == _source_key(source)
    ]
    if not candidates:
        return ""
    claim_key = _claim_key(claim)
    exact = [item for item in candidates if _claim_key(item.claim) == claim_key]
    if len(exact) == 1:
        return exact[0].id
    return ""


def _claim_matches_evidence(claim: str, evidence: Evidence) -> bool:
    claim_key = _claim_key(claim)
    evidence_key = _claim_key(evidence.claim)
    return bool(claim_key and claim_key == evidence_key)


def _source_key(source: Source) -> str:
    if source.id:
        return source.id
    url = source.canonical_url or source.url
    return canonicalize_url(url) if url else ""


def _source_lookup(sources: tuple[Source, ...]) -> dict[str, Source]:
    lookup: dict[str, Source] = {}
    for source in sources:
        keys = (
            source.id,
            canonicalize_url(source.canonical_url) if source.canonical_url else "",
            canonicalize_url(source.url) if source.url else "",
        )
        for key in keys:
            if key:
                lookup.setdefault(key, source)
    return lookup


def _claim_key(claim: str) -> str:
    return re.sub(r"\s+", " ", claim).strip().rstrip(".!?。！？").casefold()


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
    *,
    has_unbound_markers: bool = False,
    has_unbound_citations: bool = False,
) -> tuple[str, str]:
    if provider_status != "completed":
        return provider_status, provider_stop_reason
    if not report.strip():
        return "needs_review", "empty_report"
    if len(sources) < request.min_sources:
        return "needs_review", "insufficient_sources"
    if request.require_citations and (has_unbound_markers or has_unbound_citations):
        return "needs_review", "unbound_citations"
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

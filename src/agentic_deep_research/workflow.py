"""Bounded, evidence-aware web-research workflow."""

from .models import Citation, Evidence, ResearchRequest, ResearchResult
from .runner import AgentRunner

_RESEARCH_INSTRUCTIONS = """You are a rigorous web research agent.
Search the web iteratively, open the most useful pages, and continue until the question is
answered with sufficient evidence or the tool budget is exhausted. Prefer primary and recent
sources, resolve important contradictions, and do not make unsupported factual claims.
Write a clear report with visible inline citations for factual claims.
"""


def run_research(
    request: str | ResearchRequest,
    *,
    runner: AgentRunner,
) -> ResearchResult:
    """Research a topic on the web within an explicit execution budget."""
    if isinstance(request, str):
        request = ResearchRequest(topic=request)

    run = runner.run(
        instructions=_RESEARCH_INSTRUCTIONS,
        task=(
            f"Research topic:\n{request.topic.strip()}\n\n"
            f"Write the final report in {request.language}. Consult at least "
            f"{request.min_sources} distinct useful sources and cite every factual claim."
        ),
        budget=request.budget,
    )
    evidence = _extract_evidence(run.report, run.citations)
    status, stop_reason = _quality_status(
        request,
        run.status,
        run.stop_reason,
        run.report,
        evidence,
        run.sources,
    )

    return ResearchResult(
        topic=request.topic.strip(),
        report=_render_citations(run.report, run.citations),
        raw_report=run.report,
        sources=run.sources,
        citations=run.citations,
        evidence=evidence,
        trace=run.trace,
        status=status,
        stop_reason=stop_reason,
        usage=run.usage,
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


def _extract_evidence(
    report: str,
    citations: tuple[Citation, ...],
) -> tuple[Evidence, ...]:
    evidence: list[Evidence] = []
    seen: set[tuple[str, str]] = set()
    for citation in citations:
        if not _valid_citation(report, citation):
            continue
        claim = _claim_before(report, citation.start_index)
        key = (claim, citation.source.url)
        if claim and key not in seen:
            evidence.append(Evidence(claim=claim, source=citation.source))
            seen.add(key)
    return tuple(evidence)


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

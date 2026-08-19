"""Bounded web-research workflow."""

from .models import ResearchRequest, ResearchResult
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
            f"Write the final report in {request.language}."
        ),
        budget=request.budget,
    )

    return ResearchResult(
        topic=request.topic.strip(),
        report=run.report,
        sources=run.sources,
        citations=run.citations,
        trace=run.trace,
        status=run.status,
        stop_reason=run.stop_reason,
        usage=run.usage,
    )

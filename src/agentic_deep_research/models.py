"""Domain models for research workflows."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchResult:
    """Structured output from a research workflow."""

    topic: str
    plan: str
    research_notes: str
    draft_report: str
    editorial_feedback: str
    final_report: str

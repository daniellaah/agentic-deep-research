"""Static deep research workflow."""

from .models import ResearchResult
from .runner import AgentRunner

_RESEARCH_PLANNER_INSTRUCTIONS = (
    "You are a research planner. "
    "Create a concise research plan for the given topic."
)

_RESEARCHER_INSTRUCTIONS = (
    "You are a research analyst. "
    "Develop detailed research notes from the topic and plan. "
    "Clearly distinguish assumptions from supported statements."
)

_WRITER_INSTRUCTIONS = (
    "You are a report writer. "
    "Create a clear draft report from the research plan and notes."
)

_EDITOR_INSTRUCTIONS = (
    "You are a report editor. "
    "Review the draft for clarity, completeness, and logical consistency. "
    "Return actionable editorial feedback."
)

_REVISER_INSTRUCTIONS = (
    "You are a report writer. "
    "Revise the draft using the editorial feedback. "
    "Return only the final report."
)


def run_research(topic: str, *, runner: AgentRunner) -> ResearchResult:
    """Generate a reviewed final report for a non-empty topic."""
    if not topic.strip():
        raise ValueError("research topic must not be empty")

    plan = runner.run(
        instructions=_RESEARCH_PLANNER_INSTRUCTIONS,
        task=topic,
    )

    research_notes = runner.run(
        instructions=_RESEARCHER_INSTRUCTIONS,
        task=(
            f"Topic:\n{topic}\n\n"
            f"Research plan:\n{plan}"
        ),
    )

    draft_report = runner.run(
        instructions=_WRITER_INSTRUCTIONS,
        task=(
            f"Topic:\n{topic}\n\n"
            f"Research plan:\n{plan}\n\n"
            f"Research notes:\n{research_notes}"
        ),
    )

    editorial_feedback = runner.run(
        instructions=_EDITOR_INSTRUCTIONS,
        task=(
            f"Topic:\n{topic}\n\n"
            f"Draft report:\n{draft_report}"
        ),
    )

    final_report = runner.run(
        instructions=_REVISER_INSTRUCTIONS,
        task=(
            f"Topic:\n{topic}\n\n"
            f"Draft report:\n{draft_report}\n\n"
            f"Editorial feedback:\n{editorial_feedback}"
        ),
    )

    return ResearchResult(
        topic=topic,
        plan=plan,
        research_notes=research_notes,
        draft_report=draft_report,
        editorial_feedback=editorial_feedback,
        final_report=final_report,
    )

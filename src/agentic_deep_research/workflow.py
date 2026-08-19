"""Static deep research workflow."""

from .runner import AgentRunner

_RESEARCH_PLANNER_INSTRUCTIONS = (
    "You are a research planner. "
    "Create a concise research plan for the given topic."
)


def run_research(topic: str, *, runner: AgentRunner) -> str:
    """Generate a research plan for a non-empty topic."""
    if not topic.strip():
        raise ValueError("research topic must not be empty")

    return runner.run(
        instructions=_RESEARCH_PLANNER_INSTRUCTIONS,
        task=topic,
    )

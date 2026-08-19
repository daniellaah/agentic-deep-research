import pytest

from agentic_deep_research import run_research


class FakeAgentRunner:
    def run(self, *, instructions: str, task: str) -> str:
        return f"Research plan for {task}"


def test_run_research_rejects_blank_topic() -> None:
    with pytest.raises(ValueError, match="research topic must not be empty"):
        run_research("   ", runner=FakeAgentRunner())


def test_run_research_returns_generated_plan() -> None:
    result = run_research(
        "Reliable LLM agents",
        runner=FakeAgentRunner(),
    )

    assert result == "Research plan for Reliable LLM agents"

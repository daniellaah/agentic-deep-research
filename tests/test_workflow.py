import pytest

from agentic_deep_research import (
    AgentRun,
    Citation,
    ResearchBudget,
    ResearchRequest,
    ResearchStep,
    Source,
    TokenUsage,
    run_research,
)


class FakeAgentRunner:
    def __init__(self, result: AgentRun) -> None:
        self._result = result
        self.calls: list[tuple[str, str, ResearchBudget]] = []

    def run(
        self,
        *,
        instructions: str,
        task: str,
        budget: ResearchBudget,
    ) -> AgentRun:
        self.calls.append((instructions, task, budget))
        return self._result


def test_run_research_rejects_blank_topic() -> None:
    result = AgentRun(report="unused")

    with pytest.raises(ValueError, match="research topic must not be empty"):
        run_research("   ", runner=FakeAgentRunner(result))


def test_run_research_returns_grounded_report_and_execution_metadata() -> None:
    source = Source(title="Agent systems", url="https://example.com/agents")
    citation = Citation(source=source, start_index=42, end_index=45)
    step = ResearchStep(action="search", detail="reliable research agents")
    usage = TokenUsage(input_tokens=120, output_tokens=80, total_tokens=200)
    runner = FakeAgentRunner(
        AgentRun(
            report="Reliable agents need evaluation.[1]",
            sources=(source,),
            citations=(citation,),
            trace=(step,),
            status="completed",
            stop_reason="completed",
            usage=usage,
        )
    )
    request = ResearchRequest(
        topic="Reliable research agents",
        language="English",
        budget=ResearchBudget(max_tool_calls=4, max_output_tokens=2_000),
    )

    result = run_research(request, runner=runner)

    assert result.topic == "Reliable research agents"
    assert result.report == "Reliable agents need evaluation.[1]"
    assert result.sources == (source,)
    assert result.citations == (citation,)
    assert result.trace == (step,)
    assert result.status == "completed"
    assert result.stop_reason == "completed"
    assert result.usage == usage

    assert len(runner.calls) == 1
    instructions, task, budget = runner.calls[0]
    assert "web research agent" in instructions
    assert "inline citations" in instructions
    assert "Reliable research agents" in task
    assert "English" in task
    assert budget == request.budget

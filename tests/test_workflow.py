import pytest

from agentic_deep_research import (
    AgentRun,
    Citation,
    Evidence,
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
    raw_report = "Reliable agents need evaluation. [1]"
    citation_start = raw_report.index("[1]")
    citation = Citation(
        source=source,
        start_index=citation_start,
        end_index=citation_start + len("[1]"),
    )
    step = ResearchStep(action="search", detail="reliable research agents")
    usage = TokenUsage(input_tokens=120, output_tokens=80, total_tokens=200)
    runner = FakeAgentRunner(
        AgentRun(
            report=raw_report,
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
        min_sources=1,
    )

    result = run_research(request, runner=runner)

    assert result.topic == "Reliable research agents"
    assert result.report == (
        "Reliable agents need evaluation. [Agent systems](https://example.com/agents)"
    )
    assert result.sources == (source,)
    assert result.citations == (citation,)
    assert result.evidence == (
        Evidence(
            claim="Reliable agents need evaluation.",
            source=source,
            question_id="r1q1",
            excerpt="Reliable agents need evaluation.",
        ),
    )
    assert step in result.trace
    assert any(item.action == "plan_created" for item in result.trace)
    assert result.status == "completed"
    assert result.stop_reason == "completed"
    assert result.usage == usage

    assert len(runner.calls) == 1
    instructions, task, budget = runner.calls[0]
    assert "web research worker" in instructions
    assert "cite every factual claim" in instructions
    assert "Reliable research agents" in task
    assert "English" in task
    assert budget.max_tool_calls <= request.budget.max_tool_calls
    assert budget.max_output_tokens == request.budget.max_output_tokens


def test_run_research_requires_enough_sources_before_marking_success() -> None:
    runner = FakeAgentRunner(
        AgentRun(
            report="An unsupported report.",
            status="completed",
            stop_reason="completed",
        )
    )
    request = ResearchRequest(topic="Reliable agents", min_sources=2)

    result = run_research(request, runner=runner)

    assert result.status == "needs_review"
    assert result.stop_reason == "insufficient_sources"


def test_run_research_rejects_an_empty_final_report() -> None:
    sources = (
        Source(title="One", url="https://example.com/one"),
        Source(title="Two", url="https://example.com/two"),
    )
    runner = FakeAgentRunner(AgentRun(report="", sources=sources))

    result = run_research("Reliable agents", runner=runner)

    assert result.status == "needs_review"
    assert result.stop_reason == "empty_report"

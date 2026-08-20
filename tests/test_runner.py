from types import SimpleNamespace
from unittest.mock import Mock

from agentic_deep_research import ResearchBudget
from agentic_deep_research.runner import OpenAIAgentRunner


def test_run_executes_bounded_web_research_and_parses_metadata() -> None:
    client = Mock()
    source = SimpleNamespace(
        title="Reliable agent evaluation",
        url="https://example.com/evaluation",
    )
    client.responses.create.return_value = SimpleNamespace(
        output_text="Evaluation makes agents more reliable.[1]",
        status="completed",
        incomplete_details=None,
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        ),
        output=[
            SimpleNamespace(
                type="web_search_call",
                action=SimpleNamespace(
                    type="search",
                    queries=["reliable agents", "agent evaluation"],
                    sources=[source],
                ),
            ),
            SimpleNamespace(
                type="web_search_call",
                action=SimpleNamespace(
                    type="open_page",
                    url="https://example.com/evaluation",
                ),
            ),
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(
                        type="output_text",
                        annotations=[
                            SimpleNamespace(
                                type="url_citation",
                                title=source.title,
                                url=source.url,
                                start_index=39,
                                end_index=42,
                            )
                        ],
                    )
                ],
            ),
        ],
    )
    runner = OpenAIAgentRunner(client=client, model="test-model")
    budget = ResearchBudget(max_tool_calls=4, max_output_tokens=2_000)

    result = runner.run(
        instructions="You are a web research agent.",
        task="Research AI agents.",
        budget=budget,
    )

    client.responses.create.assert_called_once_with(
        model="test-model",
        instructions="You are a web research agent.",
        input="Research AI agents.",
        tools=[{"type": "web_search", "search_context_size": "medium"}],
        tool_choice="required",
        include=["web_search_call.action.sources"],
        max_tool_calls=4,
        max_output_tokens=2_000,
    )
    assert result.report == "Evaluation makes agents more reliable.[1]"
    assert [(source.title, source.url) for source in result.sources] == [
        ("Reliable agent evaluation", "https://example.com/evaluation")
    ]
    assert result.citations[0].source == result.sources[0]
    assert result.citations[0].start_index == 39
    assert result.citations[0].end_index == 42
    assert [(step.action, step.detail) for step in result.trace] == [
        ("search", "reliable agents | agent evaluation"),
        ("open_page", "https://example.com/evaluation"),
    ]
    assert result.status == "completed"
    assert result.stop_reason == "completed"
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 50
    assert result.usage.total_tokens == 150


def test_run_exposes_the_provider_stop_reason() -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text="Partial report.",
        output=[],
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_tool_calls"),
        usage=None,
    )
    runner = OpenAIAgentRunner(client=client, model="test-model")

    result = runner.run(
        instructions="Research carefully.",
        task="Research AI agents.",
        budget=ResearchBudget(max_tool_calls=1, max_output_tokens=100),
    )

    assert result.report == "Partial report."
    assert result.status == "incomplete"
    assert result.stop_reason == "max_tool_calls"


def test_run_parses_explicitly_reported_source_conflicts() -> None:
    client = Mock()
    report = "CONFLICT: Studies report opposite effects. [1] [2]"
    first_start = report.index("[1]")
    second_start = report.index("[2]")
    annotations = [
        SimpleNamespace(
            type="url_citation",
            title="Study one",
            url="https://example.com/one",
            start_index=first_start,
            end_index=first_start + 3,
        ),
        SimpleNamespace(
            type="url_citation",
            title="Study two",
            url="https://example.com/two",
            start_index=second_start,
            end_index=second_start + 3,
        ),
    ]
    client.responses.create.return_value = SimpleNamespace(
        output_text=report,
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", annotations=annotations)],
            )
        ],
        status="completed",
        incomplete_details=None,
        usage=None,
    )
    runner = OpenAIAgentRunner(client=client, model="test-model")

    result = runner.run(
        instructions="Research carefully.",
        task="Compare the studies.",
        budget=ResearchBudget(max_tool_calls=2, max_output_tokens=1_000),
    )

    assert len(result.conflicts) == 1
    assert result.conflicts[0].description == "Studies report opposite effects."
    assert [source.url for source in result.conflicts[0].sources] == [
        "https://example.com/one",
        "https://example.com/two",
    ]

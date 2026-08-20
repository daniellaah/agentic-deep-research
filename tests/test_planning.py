from types import SimpleNamespace
from unittest.mock import Mock

from agentic_deep_research import ResearchRequest
from agentic_deep_research.planning import OpenAIAdaptivePlanner


def test_openai_planner_returns_structured_prioritized_questions() -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text=(
            '{"objective":"Evaluate research-agent reliability",'
            '"questions":['
            '{"question":"How is answer quality measured?",'
            '"rationale":"Define success","priority":1},'
            '{"question":"How are failures observed?",'
            '"rationale":"Find operational gaps","priority":2}'
            "]}"
        ),
        status="completed",
        incomplete_details=None,
        usage=SimpleNamespace(input_tokens=30, output_tokens=20, total_tokens=50),
    )
    planner = OpenAIAdaptivePlanner(client=client, model="planner-model")

    result = planner.plan(
        request=ResearchRequest(topic="Reliable research agents"),
        context="Prior evidence: evaluation matters.",
        completed_questions=("What is an agent?",),
        max_questions=2,
        revision=1,
    )

    assert result.plan.objective == "Evaluate research-agent reliability"
    assert [question.id for question in result.plan.questions] == ["r2q1", "r2q2"]
    assert [question.priority for question in result.plan.questions] == [1, 2]
    assert result.plan.revision == 1
    assert result.usage.total_tokens == 50

    call = client.responses.create.call_args.kwargs
    assert call["model"] == "planner-model"
    assert call["text"]["format"]["type"] == "json_schema"
    assert call["text"]["format"]["schema"]["properties"]["questions"]["maxItems"] == 2
    assert "Prior evidence: evaluation matters." in call["input"]
    assert "What is an agent?" in call["input"]

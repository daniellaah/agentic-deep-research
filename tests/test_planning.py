from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agentic_deep_research import ResearchBrief, ResearchRequest
from agentic_deep_research.planning import OpenAIAdaptivePlanner, TopicPlanner


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


def test_openai_planner_reports_an_incomplete_structured_response() -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text='{"objective":"Incomplete plan",',
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        usage=SimpleNamespace(input_tokens=20, output_tokens=5, total_tokens=25),
    )
    planner = OpenAIAdaptivePlanner(client=client, model="planner-model")

    with pytest.raises(
        RuntimeError,
        match="planner response incomplete: max_output_tokens",
    ):
        planner.plan(
            request=ResearchRequest(topic="Reliable research agents"),
            context="",
            completed_questions=(),
            max_questions=2,
            revision=0,
        )


def test_planners_use_the_scoped_brief_as_the_research_contract() -> None:
    brief = ResearchBrief(
        research_question="Which agent reliability patterns work?",
        objective="Help engineers compare implementation trade-offs.",
        scope_inclusions=("Evaluation", "Observability"),
        constraints=("Use primary sources",),
        deliverable="A cited engineering report.",
        success_criteria=("Compare mechanisms and limitations",),
    )
    request = ResearchRequest(topic="Tell me about reliable agents", brief=brief)

    fallback = TopicPlanner().plan(
        request=request,
        context="",
        completed_questions=(),
        max_questions=1,
        revision=0,
    )

    assert fallback.plan.objective == brief.objective
    assert fallback.plan.questions[0].question == brief.research_question

    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text=(
            '{"objective":"Compare patterns","questions":['
            '{"question":"How does evaluation help?",'
            '"rationale":"Measure quality","priority":1}]}'
        ),
        status="completed",
        incomplete_details=None,
        usage=None,
    )
    OpenAIAdaptivePlanner(client=client, model="planner-model").plan(
        request=request,
        context="",
        completed_questions=(),
        max_questions=1,
        revision=0,
    )

    planner_input = client.responses.create.call_args.kwargs["input"]
    assert '"research_question": "Which agent reliability patterns work?"' in planner_input
    assert '"constraints": ["Use primary sources"]' in planner_input

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agentic_deep_research.models import (
    ClarificationDecision,
    ConversationMessage,
    ResearchBrief,
    ResearchRequest,
    ScopingRun,
)
from agentic_deep_research.scoping import OpenAIResearchScoper


def test_openai_scoper_returns_a_structured_research_brief() -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text=json.dumps(
            {
                "needs_clarification": False,
                "reason": "The request is sufficiently scoped.",
                "question": None,
                "research_brief": {
                    "research_question": "How do reliable research agents work?",
                    "objective": "Help engineers choose reliable agent patterns.",
                    "scope_inclusions": ["Planning", "Evidence verification"],
                    "scope_exclusions": [],
                    "constraints": ["Prefer primary sources"],
                    "deliverable": "A cited Markdown report.",
                    "success_criteria": ["Compare major patterns"],
                },
            }
        ),
        status="completed",
        incomplete_details=None,
        usage=SimpleNamespace(input_tokens=10, output_tokens=20, total_tokens=30),
    )
    scoper = OpenAIResearchScoper(client=client, model="scope-model")

    result = scoper.scope(
        messages=(
            ConversationMessage(
                role="user",
                content="Research reliable agents for software engineers.",
            ),
        )
    )

    assert result.brief is not None
    assert result.brief.research_question == "How do reliable research agents work?"
    assert result.usage.total_tokens == 30
    call = client.responses.create.call_args.kwargs
    assert call["model"] == "scope-model"
    assert call["text"]["format"]["strict"] is True
    assert json.loads(call["input"])["conversation"][0]["role"] == "user"


def test_openai_scoper_returns_one_material_clarification() -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text=json.dumps(
            {
                "needs_clarification": True,
                "reason": "The comparison target changes the research scope.",
                "question": "Which products should be compared?",
                "research_brief": None,
            }
        ),
        status="completed",
        incomplete_details=None,
        usage=None,
    )

    result = OpenAIResearchScoper(client=client, model="scope-model").scope(
        messages=(ConversationMessage(role="user", content="Compare them."),)
    )

    assert result.brief is None
    assert result.clarification.needs_clarification is True
    assert result.clarification.question == "Which products should be compared?"


def test_openai_scoper_rejects_an_incomplete_response() -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text="",
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
    )

    with pytest.raises(RuntimeError, match="scoping response incomplete: max_output_tokens"):
        OpenAIResearchScoper(client=client, model="scope-model").scope(
            messages=(ConversationMessage(role="user", content="Research agents."),)
        )


def test_scoping_contract_rejects_inconsistent_clarification_and_brief() -> None:
    clarification = ClarificationDecision(
        needs_clarification=True,
        reason="The target is missing.",
        question="What target should be researched?",
    )
    brief = ResearchBrief(
        research_question="A question",
        objective="An objective",
    )

    with pytest.raises(ValueError, match="brief must be None"):
        ScopingRun(clarification=clarification, brief=brief)


def test_clarification_contract_requires_exactly_one_question_when_needed() -> None:
    with pytest.raises(ValueError, match="question is required"):
        ClarificationDecision(needs_clarification=True, reason="Ambiguous")

    with pytest.raises(ValueError, match="must be None"):
        ClarificationDecision(
            needs_clarification=False,
            reason="Clear",
            question="Unnecessary question",
        )


def test_scoping_models_reject_noncanonical_container_and_boolean_types() -> None:
    with pytest.raises(TypeError, match="scope_inclusions must be a tuple"):
        ResearchBrief(
            research_question="A question",
            objective="An objective",
            scope_inclusions=["A scope"],  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="needs_clarification must be a bool"):
        ClarificationDecision(
            needs_clarification=1,  # type: ignore[arg-type]
            reason="Invalid flag type",
            question="What is the intended scope?",
        )

    with pytest.raises(TypeError, match="brief must be a ResearchBrief"):
        ResearchRequest(topic="A topic", brief="invalid")  # type: ignore[arg-type]

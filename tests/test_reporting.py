from types import SimpleNamespace
from unittest.mock import Mock

from agentic_deep_research import (
    CitationClaim,
    Evidence,
    ReportCritique,
    ResearchFinding,
    ResearchRequest,
    Source,
)
from agentic_deep_research.reporting import OpenAIReportAgent


def _response(output_text: str, *, output: list[object] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        output_text=output_text,
        output=output or [],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
    )


def test_openai_report_agent_runs_structured_writer_critic_and_reviser() -> None:
    client = Mock()
    client.responses.create.side_effect = [
        _response('{"report":"A synthesized report [S1]."}'),
        _response(
            "{"
            '"coverage_gaps":[],"unsupported_claims":[],"contradictions":[],'
            '"clarity_issues":[],"revision_instructions":[],'
            '"needs_more_research":false}'
        ),
        _response('{"report":"A revised report [S1]."}'),
    ]
    source = Source("Evaluation", "https://example.com/evaluation")
    request = ResearchRequest(topic="Reliable agents", min_sources=1)
    finding = ResearchFinding(
        question_id="q1",
        question="How are agents evaluated?",
        answer="Agents are evaluated with benchmarks.",
        sources=(source,),
    )
    evidence = Evidence("Benchmarks measure reliability.", source, question_id="q1")
    agent = OpenAIReportAgent(client=client, model="report-model")

    draft = agent.write(
        request=request,
        findings=(finding,),
        evidence=(evidence,),
        conflicts=(),
        sources=(source,),
    )
    critique = agent.critique(
        request=request,
        report=draft.report,
        findings=(finding,),
        evidence=(evidence,),
        conflicts=(),
        sources=(source,),
    )
    revised = agent.revise(
        request=request,
        report=draft.report,
        critique=critique,
        verification=SimpleNamespace(checks=()),
        findings=(finding,),
        evidence=(evidence,),
        conflicts=(),
        sources=(source,),
    )

    assert draft.report == "A synthesized report [S1]."
    assert critique == ReportCritique(usage=critique.usage)
    assert revised.report == "A revised report [S1]."
    assert all(
        call.kwargs["text"]["format"]["type"] == "json_schema"
        for call in client.responses.create.call_args_list
    )
    assert "[S1]" in client.responses.create.call_args_list[0].kwargs["input"]


def test_openai_report_agent_verifies_claims_with_web_search() -> None:
    client = Mock()
    client.responses.create.return_value = _response(
        (
            '{"checks":[{"claim_id":"C1","status":"supported",'
            '"reason":"The source reports the measured gain."}]}'
        ),
        output=[
            SimpleNamespace(
                type="web_search_call",
                action=SimpleNamespace(type="open_page", url="https://example.com/evaluation"),
            )
        ],
    )
    source = Source("Evaluation", "https://example.com/evaluation")
    agent = OpenAIReportAgent(client=client, model="report-model")

    result = agent.verify(
        request=ResearchRequest(topic="Reliable agents", min_sources=1),
        claims=(
            CitationClaim(
                claim="Evaluation improves reliability.",
                source=source,
                marker="[S1]",
                claim_id="C1",
            ),
        ),
        max_tool_calls=2,
    )

    assert result.checks[0].status == "supported"
    assert result.checks[0].source == source
    assert result.trace[0].action == "open_page"
    call = client.responses.create.call_args.kwargs
    assert call["tools"] == [{"type": "web_search", "search_context_size": "medium"}]
    assert call["tool_choice"] == "required"
    assert call["max_tool_calls"] == 2

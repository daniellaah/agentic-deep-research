import json
from types import SimpleNamespace
from unittest.mock import Mock

from agentic_deep_research import (
    CitationClaim,
    Evidence,
    ReportCritique,
    ResearchBudget,
    ResearchFinding,
    ResearchRequest,
    Source,
)
from agentic_deep_research.reporting import OpenAIReportAgent, _report_input


def _response(output_text: str, *, output: list[object] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        output_text=output_text,
        output=output or [],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
    )


def _source(index: int = 1) -> Source:
    return Source(
        title="Evaluation",
        url=f"https://example.com/evaluation?source={index}",
        id=f"S{index}",
        canonical_url=f"https://example.com/evaluation-{index}",
    )


def _evidence(source: Source, index: int = 1) -> Evidence:
    return Evidence(
        claim="Benchmarks measure reliability.",
        source=source,
        question_id="q1",
        excerpt="The benchmark reports a measurable reliability score.",
        confidence="high",
        id=f"E{index}",
        verification_status="unverified",
        origin_artifact_id="finding:q1",
        origin_start_index=10,
        origin_end_index=68,
        corroboration_count=2,
    )


def _context_records(value: str) -> list[dict[str, object]]:
    body = value.partition("provide citeable [E<number>] markers.\n")[2]
    body = body.partition("\n\nQUALITY FEEDBACK\n")[0]
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def test_openai_report_agent_runs_structured_writer_critic_and_reviser() -> None:
    client = Mock()
    client.responses.create.side_effect = [
        _response('{"report":"A synthesized report [E1]."}'),
        _response(
            "{"
            '"coverage_gaps":[],"unsupported_claims":[],"contradictions":[],'
            '"clarity_issues":[],"revision_instructions":[],'
            '"needs_more_research":false}'
        ),
        _response('{"report":"A revised report [E1]."}'),
    ]
    source = _source()
    request = ResearchRequest(topic="Reliable agents", min_sources=1)
    finding = ResearchFinding(
        question_id="q1",
        question="How are agents evaluated?",
        answer="Agents are evaluated with benchmarks.",
        sources=(source,),
    )
    evidence = _evidence(source)
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

    assert draft.report == "A synthesized report [E1]."
    assert critique == ReportCritique(usage=critique.usage)
    assert revised.report == "A revised report [E1]."
    assert all(
        call.kwargs["text"]["format"]["type"] == "json_schema"
        for call in client.responses.create.call_args_list
    )
    writer_input = client.responses.create.call_args_list[0].kwargs["input"]
    writer_records = _context_records(writer_input)
    evidence_record = next(item for item in writer_records if item["kind"] == "evidence")
    assert evidence_record == {
        "kind": "evidence",
        "marker": "[E1]",
        "evidence_id": "E1",
        "claim": "Benchmarks measure reliability.",
        "excerpt": "The benchmark reports a measurable reliability score.",
        "question_id": "q1",
        "confidence": "high",
        "verification_status": "unverified",
        "corroboration_count": 2,
        "source": {
            "source_id": "S1",
            "title": "Evaluation",
            "url": "https://example.com/evaluation-1",
        },
        "origin": {
            "artifact_id": "finding:q1",
            "start_index": 10,
            "end_index": 68,
        },
    }
    assert "[S1]" not in writer_input
    assert "[E<number>]" in client.responses.create.call_args_list[2].kwargs["instructions"]


def test_openai_report_agent_verifies_claims_with_web_search() -> None:
    client = Mock()
    client.responses.create.return_value = _response(
        (
            '{"checks":[{"claim_id":"C1","evidence_id":"E1",'
            '"status":"supported",'
            '"reason":"The source reports the measured gain."}]}'
        ),
        output=[
            SimpleNamespace(
                type="web_search_call",
                action=SimpleNamespace(type="open_page", url="https://example.com/evaluation"),
            )
        ],
    )
    source = _source()
    agent = OpenAIReportAgent(client=client, model="report-model")

    result = agent.verify(
        request=ResearchRequest(topic="Reliable agents", min_sources=1),
        claims=(
            CitationClaim(
                claim="Evaluation improves reliability.",
                source=source,
                marker="[E1]",
                claim_id="C1",
                evidence_id="E1",
            ),
        ),
        max_tool_calls=2,
    )

    assert result.checks[0].status == "supported"
    assert result.checks[0].source == source
    assert result.checks[0].evidence_id == "E1"
    assert result.trace[0].action == "open_page"
    call = client.responses.create.call_args.kwargs
    assert call["tools"] == [{"type": "web_search", "search_context_size": "medium"}]
    assert call["tool_choice"] == "required"
    assert call["max_tool_calls"] == 2
    verifier_input = json.loads(call["input"])
    assert verifier_input["claims"][0]["evidence_id"] == "E1"
    assert verifier_input["claims"][0]["source_id"] == "S1"
    assert verifier_input["claims"][0]["source_url"] == (
        "https://example.com/evaluation-1"
    )
    schema_item = call["text"]["format"]["schema"]["properties"]["checks"]["items"]
    assert "evidence_id" in schema_item["required"]


def test_verifier_marks_a_model_omission_uncertain_and_preserves_evidence_id() -> None:
    client = Mock()
    client.responses.create.return_value = _response(
        '{"checks":[{"claim_id":"C1","evidence_id":"E1",'
        '"status":"supported","reason":"Direct support."}]}'
    )
    first_source = _source(1)
    second_source = _source(2)
    claims = (
        CitationClaim(
            claim="First claim.",
            source=first_source,
            marker="[E1]",
            claim_id="C1",
            evidence_id="E1",
        ),
        CitationClaim(
            claim="Second claim.",
            source=second_source,
            marker="[E2]",
            claim_id="C2",
            evidence_id="E2",
        ),
    )

    result = OpenAIReportAgent(client=client, model="report-model").verify(
        request=ResearchRequest(topic="Reliable agents", min_sources=1),
        claims=claims,
        max_tool_calls=2,
    )

    assert [item.claim_id for item in result.checks] == ["C1", "C2"]
    assert result.checks[0].status == "supported"
    assert result.checks[1].status == "uncertain"
    assert result.checks[1].evidence_id == "E2"
    assert "omitted this claim" in result.checks[1].reason


def test_verifier_preserves_evidence_id_when_tool_budget_is_exhausted() -> None:
    client = Mock()
    claim = CitationClaim(
        claim="Evaluation improves reliability.",
        source=_source(),
        marker="[E1]",
        claim_id="C1",
        evidence_id="E1",
    )

    result = OpenAIReportAgent(client=client, model="report-model").verify(
        request=ResearchRequest(topic="Reliable agents", min_sources=1),
        claims=(claim,),
        max_tool_calls=0,
    )

    assert result.checks[0].status == "uncertain"
    assert result.checks[0].evidence_id == "E1"
    client.responses.create.assert_not_called()


def test_verifier_does_not_attach_a_check_to_the_wrong_evidence() -> None:
    client = Mock()
    client.responses.create.return_value = _response(
        '{"checks":[{"claim_id":"C1","evidence_id":"E999",'
        '"status":"supported","reason":"Direct support."}]}'
    )
    claim = CitationClaim(
        claim="Evaluation improves reliability.",
        source=_source(),
        marker="[E1]",
        claim_id="C1",
        evidence_id="E1",
    )

    result = OpenAIReportAgent(client=client, model="report-model").verify(
        request=ResearchRequest(topic="Reliable agents", min_sources=1),
        claims=(claim,),
        max_tool_calls=2,
    )

    assert result.checks[0].status == "uncertain"
    assert result.checks[0].evidence_id == "E1"
    assert "mismatched evidence_id" in result.checks[0].reason


def test_verifier_treats_duplicate_checks_as_uncertain() -> None:
    client = Mock()
    client.responses.create.return_value = _response(
        '{"checks":['
        '{"claim_id":"C1","evidence_id":"E1","status":"supported",'
        '"reason":"Direct support."},'
        '{"claim_id":"C1","evidence_id":"E1","status":"unsupported",'
        '"reason":"Contradictory result."}'
        "]}"
    )
    claim = CitationClaim(
        claim="Evaluation improves reliability.",
        source=_source(),
        marker="[E1]",
        claim_id="C1",
        evidence_id="E1",
    )

    result = OpenAIReportAgent(client=client, model="report-model").verify(
        request=ResearchRequest(topic="Reliable agents", min_sources=1),
        claims=(claim,),
        max_tool_calls=2,
    )

    assert result.checks[0].status == "uncertain"
    assert "duplicate checks" in result.checks[0].reason


def test_verifier_treats_a_missing_reason_as_uncertain() -> None:
    client = Mock()
    client.responses.create.return_value = _response(
        '{"checks":[{"claim_id":"C1","evidence_id":"E1",'
        '"status":"supported","reason":""}]}'
    )
    claim = CitationClaim(
        claim="Evaluation improves reliability.",
        source=_source(),
        marker="[E1]",
        claim_id="C1",
        evidence_id="E1",
    )

    result = OpenAIReportAgent(client=client, model="report-model").verify(
        request=ResearchRequest(topic="Reliable agents", min_sources=1),
        claims=(claim,),
        max_tool_calls=2,
    )

    assert result.checks[0].status == "uncertain"
    assert "no reason" in result.checks[0].reason


def test_report_input_packs_only_complete_json_records_within_budget() -> None:
    source = _source()
    evidence = _evidence(source)
    request = ResearchRequest(
        topic="Reliable agents",
        min_sources=1,
        budget=ResearchBudget(max_context_chars=800),
    )
    oversized_finding = ResearchFinding(
        question_id="q1",
        question="How are agents evaluated?",
        answer="A" * 2_000,
        sources=(source,),
    )

    value = _report_input(
        request=request,
        report="R" * 2_000,
        findings=(oversized_finding,),
        evidence=(evidence,),
        conflicts=(),
        sources=(source,),
    )
    body = value.partition("provide citeable [E<number>] markers.\n")[2]
    records = [json.loads(line) for line in body.splitlines() if line.strip()]

    assert len(body) <= request.budget.max_context_chars
    assert [item["kind"] for item in records] == ["evidence"]
    assert records[0]["evidence_id"] == "E1"
    assert "R" * 100 not in body
    assert "A" * 100 not in body

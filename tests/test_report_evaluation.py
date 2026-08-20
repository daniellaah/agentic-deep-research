import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agentic_deep_research.models import CitationCheck, Evidence, Source
from agentic_deep_research.report_evaluation import (
    DeterministicReportJudge,
    OpenAIReportJudge,
    ReportJudgeError,
    ReportJudgment,
    RubricCriterion,
    RubricScore,
)


def _artifacts(
    *,
    citation_status: str = "supported",
    quality: str = "primary",
) -> tuple[tuple[Evidence, ...], tuple[CitationCheck, ...]]:
    source = Source(
        title="Official report",
        url="https://example.com/report",
        quality=quality,
        id="src_1",
        canonical_url="https://example.com/report",
    )
    evidence = Evidence(
        id="ev_1",
        claim="The measured result was 42 percent.",
        source=source,
        verification_status=citation_status,
    )
    check = CitationCheck(
        claim=evidence.claim,
        source=source,
        status=citation_status,
        reason="The source directly supports the measurement.",
        claim_id="claim_1",
        evidence_id=evidence.id,
    )
    return (evidence,), (check,)


def test_rubric_value_objects_reject_invalid_data() -> None:
    with pytest.raises(ValueError, match="criterion id"):
        RubricCriterion(" ", "Useful description", 1.0)
    with pytest.raises(ValueError, match="positive finite"):
        RubricCriterion("groundedness", "Useful description", float("nan"))
    with pytest.raises(ValueError, match="between 0 and 1"):
        RubricScore("groundedness", 1.01, "Reason")
    with pytest.raises(ValueError, match="reason"):
        RubricScore("groundedness", 0.5, " ")
    with pytest.raises(ValueError, match="duplicate"):
        ReportJudgment(
            overall_score=0.5,
            passed=False,
            scores=(
                RubricScore("groundedness", 0.5, "First"),
                RubricScore("groundedness", 0.5, "Second"),
            ),
            grader="test",
        )


def test_deterministic_report_judge_passes_supported_report() -> None:
    evidence, checks = _artifacts()

    judgment = DeterministicReportJudge().judge(
        report="A complete report with a supported measurement.",
        evidence=evidence,
        citation_checks=checks,
    )

    assert judgment.passed is True
    assert judgment.overall_score == 1.0
    assert [score.criterion_id for score in judgment.scores] == [
        "report_present",
        "evidence_present",
        "citation_integrity",
        "source_quality",
    ]
    assert judgment.grader == DeterministicReportJudge().grader


def test_deterministic_report_judge_fails_when_a_citation_is_uncertain() -> None:
    evidence, checks = _artifacts(citation_status="uncertain")

    judgment = DeterministicReportJudge().judge(
        report="A report whose citation could not be verified.",
        evidence=evidence,
        citation_checks=checks,
    )

    assert judgment.passed is False
    assert judgment.overall_score == pytest.approx(0.6)
    citation_score = next(
        score for score in judgment.scores if score.criterion_id == "citation_integrity"
    )
    assert citation_score.score == 0.0


def test_deterministic_report_judge_fails_an_unbound_supported_check() -> None:
    evidence, checks = _artifacts()
    unbound = CitationCheck(
        claim=checks[0].claim,
        source=checks[0].source,
        status="supported",
        reason="The source supports the claim.",
        claim_id="claim_1",
        evidence_id="ev_missing",
    )

    judgment = DeterministicReportJudge().judge(
        report="A report with a citation bound to the wrong evidence record.",
        evidence=evidence,
        citation_checks=(unbound,),
    )

    assert judgment.passed is False
    citation_score = next(
        score for score in judgment.scores if score.criterion_id == "citation_integrity"
    )
    assert citation_score.score == 0.0


def test_openai_report_judge_uses_structured_outputs_without_tools() -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text=json.dumps(
            {
                "scores": [
                    {
                        "criterion_id": "groundedness",
                        "score": 0.5,
                        "reason": "Most claims match the evidence.",
                    },
                    {
                        "criterion_id": "clarity",
                        "score": 1.0,
                        "reason": "The report is easy to follow.",
                    },
                ]
            }
        )
    )
    criteria = (
        RubricCriterion("groundedness", "Claims use evidence.", 1.0),
        RubricCriterion("clarity", "The report is clear.", 3.0),
    )
    evidence, checks = _artifacts()
    judge = OpenAIReportJudge(
        client=client,
        model="judge-model",
        criteria=criteria,
        pass_threshold=0.8,
    )

    judgment = judge.judge(
        report="A long-form research report.",
        evidence=evidence,
        citation_checks=checks,
    )

    assert judgment.overall_score == pytest.approx(0.875)
    assert judgment.passed is True
    assert judgment.grader == judge.grader
    assert [score.criterion_id for score in judgment.scores] == [
        "groundedness",
        "clarity",
    ]
    call = client.responses.create.call_args.kwargs
    assert call["model"] == "judge-model"
    assert "tools" not in call
    assert call["text"]["format"]["type"] == "json_schema"
    assert call["text"]["format"]["strict"] is True
    score_schema = call["text"]["format"]["schema"]["properties"]["scores"]
    assert score_schema["minItems"] == 2
    assert score_schema["maxItems"] == 2
    assert score_schema["items"]["properties"]["criterion_id"]["enum"] == [
        "groundedness",
        "clarity",
    ]
    payload = json.loads(call["input"])
    assert payload["report"] == "A long-form research report."
    assert payload["evidence"][0]["evidence_id"] == "ev_1"
    assert payload["citation_checks"][0]["status"] == "supported"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "scores": [
                {"criterion_id": "one", "score": 0.8, "reason": "First"},
                {"criterion_id": "one", "score": 0.7, "reason": "Duplicate"},
            ]
        },
        {
            "scores": [
                {"criterion_id": "one", "score": 0.8, "reason": "Missing two"},
            ]
        },
        {
            "scores": [
                {"criterion_id": "one", "score": 0.8, "reason": "First"},
                {"criterion_id": "unknown", "score": 0.7, "reason": "Unknown"},
            ]
        },
        {
            "scores": [
                {"criterion_id": "one", "score": 1.1, "reason": "Out of range"},
                {"criterion_id": "two", "score": 0.7, "reason": "Second"},
            ]
        },
        {
            "scores": [
                {"criterion_id": "one", "score": 0.8, "reason": " "},
                {"criterion_id": "two", "score": 0.7, "reason": "Second"},
            ]
        },
        {
            "scores": [
                {"criterion_id": "one", "score": True, "reason": "Boolean"},
                {"criterion_id": "two", "score": 0.7, "reason": "Second"},
            ]
        },
    ],
    ids=[
        "duplicate-id",
        "missing-id",
        "unknown-id",
        "score-out-of-range",
        "empty-reason",
        "boolean-score",
    ],
)
def test_openai_report_judge_fails_closed_on_invalid_output(
    payload: dict[str, object],
) -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(output_text=json.dumps(payload))
    criteria = (
        RubricCriterion("one", "First criterion", 1.0),
        RubricCriterion("two", "Second criterion", 1.0),
    )
    judge = OpenAIReportJudge(client=client, model="judge-model", criteria=criteria)

    with pytest.raises(ReportJudgeError, match="report judge failed"):
        judge.judge(report="Report", evidence=(), citation_checks=())


def test_openai_report_judge_rejects_duplicate_configured_criteria() -> None:
    criteria = (
        RubricCriterion("same", "First", 1.0),
        RubricCriterion("same", "Second", 1.0),
    )

    with pytest.raises(ValueError, match="unique"):
        OpenAIReportJudge(client=Mock(), model="judge-model", criteria=criteria)


def test_openai_report_judge_rejects_oversized_input_before_the_api_call() -> None:
    client = Mock()
    judge = OpenAIReportJudge(
        client=client,
        model="judge-model",
        max_input_chars=10,
    )

    with pytest.raises(ReportJudgeError, match="report judge failed"):
        judge.judge(report="A long report", evidence=(), citation_checks=())

    client.responses.create.assert_not_called()


def test_report_grader_identity_changes_with_scoring_configuration() -> None:
    assert (
        DeterministicReportJudge(pass_threshold=0.6).grader
        != DeterministicReportJudge(pass_threshold=0.8).grader
    )
    assert (
        OpenAIReportJudge(client=Mock(), model="judge-model").grader
        != OpenAIReportJudge(
            client=Mock(),
            model="judge-model",
            max_input_chars=50_000,
        ).grader
    )


def test_openai_report_judge_rejects_incomplete_response() -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text=json.dumps(
            {
                "scores": [
                    {
                        "criterion_id": criterion.id,
                        "score": 1.0,
                        "reason": "Looks complete.",
                    }
                    for criterion in (
                        RubricCriterion("groundedness", "Grounded.", 1.0),
                    )
                ]
            }
        ),
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
    )
    judge = OpenAIReportJudge(
        client=client,
        model="judge-model",
        criteria=(RubricCriterion("groundedness", "Grounded.", 1.0),),
    )

    with pytest.raises(ReportJudgeError, match="report judge failed"):
        judge.judge(report="Report", evidence=(), citation_checks=())

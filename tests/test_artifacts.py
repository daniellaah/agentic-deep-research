import hashlib
import json
from dataclasses import replace

import pytest

from agentic_deep_research.artifacts import (
    ExportIntegrityError,
    export_json,
    export_markdown,
    export_result,
)
from agentic_deep_research.checkpoint import EffectRecord, RunState
from agentic_deep_research.models import (
    Citation,
    CitationCheck,
    Evidence,
    EvidenceConflict,
    ResearchArtifact,
    ResearchFinding,
    ResearchPlan,
    ResearchQuestion,
    ResearchRequest,
    ResearchResult,
    ResearchStep,
    Source,
    TokenUsage,
)


def _completed_state(
    *,
    research_status: str = "completed",
    source_url: str = "https://example.com/research?a=1",
) -> RunState:
    source = Source(
        id="src_example",
        title="Example research",
        url=source_url,
        canonical_url=source_url,
        quality="primary",
    )
    citation_text = "An emoji 🧪 supports the finding [E1]."
    citation_start = citation_text.index("[E1]")
    evidence = Evidence(
        id="ev_example",
        claim="An emoji 🧪 supports the finding",
        source=source,
        question_id="q1",
        excerpt="Public supporting excerpt.",
        verification_status="supported",
        origin_artifact_id="finding:q1",
        origin_start_index=0,
        origin_end_index=32,
    )
    citation = Citation(
        source=source,
        start_index=citation_start,
        end_index=citation_start + len("[E1]"),
        evidence_id=evidence.id,
    )
    result = ResearchResult(
        topic="Reliable research",
        report="Rendered report is deliberately not trusted.",
        raw_report=citation_text,
        sources=(source,),
        citations=(citation,),
        evidence=(evidence,),
        trace=(
            ResearchStep(action="search", detail="PRIVATE TRACE QUERY"),
        ),
        status=research_status,
        stop_reason=(
            "completed" if research_status == "completed" else "insufficient_sources"
        ),
        usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        plan=ResearchPlan(
            objective="Assess reliability",
            questions=(
                ResearchQuestion(
                    id="q1",
                    question="What evidence supports reliability?",
                    rationale="Find measurable support.",
                    priority=1,
                ),
            ),
        ),
        findings=(
            ResearchFinding(
                question_id="q1",
                question="What evidence supports reliability?",
                answer="PRIVATE INTERMEDIATE FINDING",
            ),
        ),
        conflicts=(
            EvidenceConflict(
                question_id="q1",
                description="Sources disagree about one boundary.",
                sources=(source,),
            ),
        ),
        artifacts=(
            ResearchArtifact(
                id="report:draft",
                kind="report_draft",
                content="PRIVATE DRAFT CONTENT",
            ),
        ),
        citation_checks=(
            CitationCheck(
                claim=evidence.claim,
                source=source,
                status="supported",
                reason="The source directly supports the claim.",
                claim_id="C1",
                evidence_id=evidence.id,
            ),
        ),
        revision_count=1,
    )
    state = RunState.create(
        ResearchRequest(topic="Reliable research", language="English"),
        run_id="run-export",
    )
    effect = EffectRecord(
        effect_id="runner.run:private",
        kind="runner.run",
        input_hash="PRIVATE INPUT HASH",
        status="completed",
        attempts=1,
        result_type="AgentRun",
        result={"report": "PRIVATE EFFECT RESULT"},
    )
    return replace(
        state,
        engine_fingerprint="PRIVATE ENGINE FINGERPRINT",
        status="completed",
        current_step="completed",
        effects=(effect,),
        result=result,
        error_type="PrivateError",
        error_message="PRIVATE RAW ERROR",
    )


def test_json_export_is_deterministic_and_explicitly_public() -> None:
    state = _completed_state()

    first = export_json(state)
    second = export_json(state)

    assert first == second
    assert first.media_type == "application/json; charset=utf-8"
    assert first.filename == "run-export.json"
    assert first.etag == f'"sha256-{hashlib.sha256(first.body).hexdigest()}"'

    payload = json.loads(first.body)
    research = payload["research"]
    citation = research["citations"][0]
    semantics = research["citation_index_semantics"]
    assert payload["schema_version"] == 1
    assert payload["schema"] == "agentic-deep-research/research-artifact@1"
    assert payload["run"]["run_id"] == "run-export"
    assert payload["run"]["runtime_status"] == "completed"
    assert payload["request"]["budget"]["max_tool_calls"] == 8
    assert payload["request"]["min_sources"] == 2
    assert research["citation_text"] == state.result.raw_report
    assert citation["marker"] == "[E1]"
    assert state.result.raw_report[citation["start_index"] : citation["end_index"]] == "[E1]"
    assert semantics == {
        "end": "exclusive",
        "start": "inclusive",
        "text_field": "citation_text",
        "unit": "unicode_code_point",
    }
    assert research["quality"]["warning"] is None
    assert research["evidence"][0]["source_id"] == "src_example"

    serialized = first.body.decode()
    for private_value in (
        "PRIVATE TRACE QUERY",
        "PRIVATE INTERMEDIATE FINDING",
        "PRIVATE DRAFT CONTENT",
        "PRIVATE INPUT HASH",
        "PRIVATE EFFECT RESULT",
        "PRIVATE ENGINE FINGERPRINT",
        "PRIVATE RAW ERROR",
    ):
        assert private_value not in serialized


def test_markdown_marks_non_completed_research_and_omits_unsafe_links() -> None:
    state = _completed_state(
        research_status="needs_review",
        source_url="javascript:alert(1)",
    )
    raw_report = "Model link [click](javascript:alert(2)) and claim [E1]."
    citation_start = raw_report.index("[E1]")
    citation = replace(
        state.result.citations[0],
        start_index=citation_start,
        end_index=citation_start + 4,
    )
    state = replace(
        state,
        result=replace(
            state.result,
            raw_report=raw_report,
            citations=(citation,),
        ),
    )

    artifact = export_markdown(state)
    markdown = artifact.body.decode()

    assert artifact.media_type == "text/markdown; charset=utf-8"
    assert artifact.filename == "run-export.md"
    assert "> [!WARNING]" in markdown
    assert "Research status: `needs_review`" in markdown
    assert "link omitted: unsupported URL" in markdown
    assert "](javascript:" not in markdown
    assert "\\[click\\]\\(javascript:alert\\(2\\)\\)" in markdown


def test_markdown_creates_only_valid_http_source_links() -> None:
    artifact = export_markdown(_completed_state())
    markdown = artifact.body.decode()

    assert "[Example research](https://example.com/research?a=1)" in markdown
    assert "> [!WARNING]" not in markdown
    assert artifact.etag == f'"sha256-{hashlib.sha256(artifact.body).hexdigest()}"'


@pytest.mark.parametrize("status", ["created", "running", "failed", "cancelled"])
def test_export_rejects_a_run_that_has_not_completed(status: str) -> None:
    state = replace(_completed_state(), status=status)

    with pytest.raises(ExportIntegrityError, match="only completed runs"):
        export_json(state)


def test_export_rejects_invalid_citation_offsets() -> None:
    state = _completed_state()
    broken = replace(
        state.result.citations[0],
        end_index=len(state.result.raw_report) + 1,
    )
    state = replace(state, result=replace(state.result, citations=(broken,)))

    with pytest.raises(ExportIntegrityError, match="invalid citation_text offsets"):
        export_markdown(state)


def test_export_rejects_a_citation_with_an_unknown_source() -> None:
    state = _completed_state()
    unknown = replace(
        state.result.citations[0],
        source=replace(state.result.sources[0], id="src_unknown"),
    )
    state = replace(state, result=replace(state.result, citations=(unknown,)))

    with pytest.raises(ExportIntegrityError, match="unknown source"):
        export_json(state)


def test_export_rejects_a_dangling_evidence_reference() -> None:
    state = _completed_state()
    dangling = replace(state.result.citations[0], evidence_id="ev_unknown")
    state = replace(state, result=replace(state.result, citations=(dangling,)))

    with pytest.raises(ExportIntegrityError, match="unknown evidence"):
        export_json(state)


def test_export_rejects_a_source_mismatch_between_citation_and_evidence() -> None:
    state = _completed_state()
    second_source = Source(
        id="src_second",
        title="Second source",
        url="https://example.org/second",
        canonical_url="https://example.org/second",
    )
    citation = replace(state.result.citations[0], source=second_source)
    state = replace(
        state,
        result=replace(
            state.result,
            sources=(*state.result.sources, second_source),
            citations=(citation,),
        ),
    )

    with pytest.raises(ExportIntegrityError, match="does not match its evidence"):
        export_markdown(state)


def test_export_result_dispatches_formats_and_rejects_unknown_values() -> None:
    state = _completed_state()

    assert export_result(state, format="json") == export_json(state)
    assert export_result(state, format="markdown") == export_markdown(state)
    with pytest.raises(ValueError, match="unsupported export format"):
        export_result(state, format="pdf")  # type: ignore[arg-type]

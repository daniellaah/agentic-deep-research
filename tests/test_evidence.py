from agentic_deep_research.evidence import EvidenceStore, canonicalize_url
from agentic_deep_research.models import (
    AgentRun,
    Citation,
    CitationCheck,
    Evidence,
    Source,
)


def _cited_run(claim: str, url: str) -> AgentRun:
    source = Source(title="Study", url=url)
    report = f"{claim} [1]"
    start_index = report.index("[1]")
    return AgentRun(
        report=report,
        sources=(source,),
        citations=(Citation(source, start_index, start_index + 3),),
    )


def test_canonical_url_and_ids_are_stable_across_store_rebuilds() -> None:
    tracked_url = (
        "HTTPS://Example.COM:443/research?b=2&utm_source=newsletter&a=1#results"
    )
    canonical_url = "https://example.com/research?a=1&b=2"
    first = EvidenceStore()
    second = EvidenceStore()

    first.ingest_run("q1", _cited_run("Evaluation improves reliability.", tracked_url))
    second.ingest_run(
        "q1",
        _cited_run("Evaluation improves reliability.", canonical_url),
    )

    assert canonicalize_url(tracked_url) == canonical_url
    assert first.sources[0].canonical_url == canonical_url
    assert first.sources[0].id == second.sources[0].id
    assert first.evidence[0].id == second.evidence[0].id


def test_url_aliases_do_not_manufacture_corroboration() -> None:
    store = EvidenceStore()
    claim = "Evaluation improves reliability."

    store.ingest_run(
        "q1",
        _cited_run(claim, "https://example.com/study?utm_source=one#summary"),
    )
    store.ingest_run(
        "q2",
        _cited_run(claim, "https://EXAMPLE.com/study"),
    )

    assert len(store.sources) == 1
    assert len(store.evidence) == 2
    assert {item.question_id for item in store.evidence} == {"q1", "q2"}
    assert len({item.id for item in store.evidence}) == 2
    assert {item.corroboration_count for item in store.evidence} == {1}
    assert {item.confidence for item in store.evidence} == {"medium"}


def test_distinct_canonical_sources_increase_corroboration() -> None:
    store = EvidenceStore()
    claim = "Evaluation improves reliability."

    store.ingest_run("q1", _cited_run(claim, "https://one.example/study"))
    store.ingest_run("q2", _cited_run(claim, "https://two.example/study"))

    assert {item.corroboration_count for item in store.evidence} == {2}
    assert {item.confidence for item in store.evidence} == {"high"}


def test_source_quality_is_preserved_but_not_inferred_from_its_domain() -> None:
    store = EvidenceStore()
    store.ingest_run(
        "q1",
        _cited_run(
            "An agency page reports a result.",
            "https://example.gov/study",
        ),
    )
    explicit_primary = Source(
        "Original dataset",
        "https://data.example/dataset",
        quality="primary",
    )
    store.ingest_run(
        "q2",
        AgentRun(report="Dataset consulted.", sources=(explicit_primary,)),
    )

    assert store.sources[0].quality == "unknown"
    assert store.sources[1].quality == "primary"


def test_ingest_run_records_truthful_origin_without_fabricating_excerpt() -> None:
    source = Source("Evaluation", "https://example.com/evaluation")
    report = "Background.\n\nEvaluation improves reliability. [1]"
    citation_start = report.index("[1]")
    run = AgentRun(
        report=report,
        sources=(source,),
        citations=(Citation(source, citation_start, citation_start + 3),),
    )
    store = EvidenceStore()

    packet = store.ingest_run("q1", run, artifact_id="finding:q1:attempt-1")
    item = packet.evidence[0]

    assert packet.question_id == "q1"
    assert packet.answer == report
    assert packet.artifact_id == "finding:q1:attempt-1"
    assert item.claim == "Evaluation improves reliability."
    assert item.excerpt == ""
    assert item.verification_status == "unverified"
    assert item.origin_artifact_id == packet.artifact_id
    assert report[item.origin_start_index : item.origin_end_index] == item.claim
    assert store.snapshot().evidence == packet.evidence
    assert store.snapshot().entries == store.snapshot().evidence


def test_submitted_evidence_preserves_a_real_excerpt_and_is_deduplicated() -> None:
    source = Source("Evaluation", "https://example.com/evaluation")
    submitted = Evidence(
        claim="Evaluation improves reliability.",
        source=source,
        question_id="model-supplied-question",
        excerpt="The measured success rate increased by twelve percent.",
        verification_status="supported",
        origin_artifact_id="model-supplied-artifact",
    )
    report = "Evaluation improves reliability. [1]"
    citation_start = report.index("[1]")
    run = AgentRun(
        report=report,
        sources=(source,),
        citations=(Citation(source, citation_start, citation_start + 3),),
        evidence=(submitted,),
    )
    store = EvidenceStore()

    packet = store.ingest_run("q1", run)

    assert len(packet.evidence) == 1
    assert len(store.evidence) == 1
    assert packet.evidence[0].excerpt == submitted.excerpt
    assert packet.evidence[0].question_id == "q1"
    assert packet.evidence[0].verification_status == "unverified"
    assert packet.evidence[0].origin_artifact_id == "finding:q1"


def test_invalid_citations_are_not_added_to_the_ledger() -> None:
    source = Source("Evaluation", "https://example.com/evaluation")
    run = AgentRun(
        report="Short report.",
        citations=(Citation(source, 50, 53),),
    )
    store = EvidenceStore()

    packet = store.ingest_run("q1", run)

    assert packet.evidence == ()
    assert store.snapshot().evidence == ()


def test_ledger_with_checks_links_status_by_stable_evidence_id() -> None:
    store = EvidenceStore()
    store.ingest_run(
        "q1",
        _cited_run(
            "Evaluation improves reliability.",
            "https://example.com/evaluation",
        ),
    )
    ledger = store.snapshot()
    item = ledger.evidence[0]
    supported = CitationCheck(
        claim=item.claim,
        source=item.source,
        status="supported",
        reason="The source reports the measured result.",
        claim_id="C1",
        evidence_id=item.id,
    )
    unsupported = CitationCheck(
        claim=item.claim,
        source=item.source,
        status="unsupported",
        reason="A second use overstates the source.",
        claim_id="C2",
        evidence_id=item.id,
    )

    checked = ledger.with_checks((supported, unsupported))

    assert ledger.evidence[0].verification_status == "unverified"
    assert checked.evidence[0].verification_status == "unsupported"
    assert checked.checks == (supported, unsupported)

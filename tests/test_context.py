import json

import pytest

from agentic_deep_research.context import ContextBuilder
from agentic_deep_research.models import (
    BudgetSnapshot,
    Evidence,
    EvidenceConflict,
    EvidenceLedger,
    Source,
)


def _source(source_id: str) -> Source:
    return Source(
        id=source_id,
        title=f"Source {source_id}",
        url=f"https://{source_id}.example/study",
        canonical_url=f"https://{source_id}.example/study",
    )


def _evidence(
    evidence_id: str,
    *,
    question_id: str,
    source_id: str,
    status: str = "supported",
    corroboration_count: int = 1,
    claim: str | None = None,
) -> Evidence:
    return Evidence(
        id=evidence_id,
        question_id=question_id,
        claim=claim or f"Claim {evidence_id}.",
        source=_source(source_id),
        verification_status=status,  # type: ignore[arg-type]
        corroboration_count=corroboration_count,
    )


def _budget() -> BudgetSnapshot:
    return BudgetSnapshot(tool_calls_remaining=3, research_steps_remaining=2)


def _records(text: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in text.splitlines()]


def test_budget_snapshot_rejects_negative_remaining_capacity() -> None:
    with pytest.raises(ValueError, match="tool_calls_remaining"):
        BudgetSnapshot(tool_calls_remaining=-1, research_steps_remaining=0)

    with pytest.raises(ValueError, match="research_steps_remaining"):
        BudgetSnapshot(tool_calls_remaining=0, research_steps_remaining=-1)


def test_context_pack_uses_only_complete_jsonl_records() -> None:
    selected = _evidence(
        "ev_selected",
        question_id="q1",
        source_id="selected",
    )
    oversized = _evidence(
        "ev_oversized",
        question_id="q1",
        source_id="oversized",
        claim="X" * 1_000,
    )
    unsupported = _evidence(
        "ev_unsupported",
        question_id="q1",
        source_id="unsupported",
        status="unsupported",
    )

    pack = ContextBuilder(max_chars=500).build(
        purpose="worker",
        ledger=EvidenceLedger(evidence=(oversized, unsupported, selected)),
        budget=_budget(),
        active_question_id="q1",
    )

    records = _records(pack.text)
    assert all(isinstance(record, dict) for record in records)
    assert pack.used_chars == len(pack.text)
    assert pack.used_chars <= 500
    assert pack.selected_evidence_ids == ("ev_selected",)
    assert pack.omitted_evidence_count == 2
    assert not any(
        record.get("verification_status") == "unsupported" for record in records
    )
    assert "ev_oversized" not in pack.text
    assert "[context truncated]" not in pack.text


def test_active_question_precedes_stronger_non_active_evidence() -> None:
    active = _evidence(
        "ev_active",
        question_id="q-active",
        source_id="active",
        status="uncertain",
        corroboration_count=1,
    )
    non_active = _evidence(
        "ev_other",
        question_id="q-other",
        source_id="other",
        status="supported",
        corroboration_count=5,
    )
    ledger = EvidenceLedger(evidence=(non_active, active))
    wide = ContextBuilder(max_chars=10_000).build(
        purpose="worker",
        ledger=ledger,
        budget=_budget(),
        active_question_id="q-active",
    )
    lines = wide.text.splitlines()
    active_line = next(line for line in lines if '"ev_active"' in line)
    metadata_line = next(line for line in lines if '"kind":"context"' in line)

    narrow = ContextBuilder(
        max_chars=len(metadata_line) + 1 + len(active_line)
    ).build(
        purpose="worker",
        ledger=ledger,
        budget=_budget(),
        active_question_id="q-active",
    )

    assert narrow.selected_evidence_ids == ("ev_active",)
    assert narrow.omitted_evidence_count == 1


def test_verification_corroboration_and_source_round_robin_are_stable() -> None:
    supported_high_a = _evidence(
        "ev_a1",
        question_id="q1",
        source_id="a",
        status="supported",
        corroboration_count=3,
    )
    supported_second_a = _evidence(
        "ev_a2",
        question_id="q1",
        source_id="a",
        status="supported",
        corroboration_count=2,
    )
    supported_b = _evidence(
        "ev_b1",
        question_id="q1",
        source_id="b",
        status="supported",
        corroboration_count=1,
    )
    unverified = _evidence(
        "ev_c1",
        question_id="q1",
        source_id="c",
        status="unverified",
        corroboration_count=9,
    )
    items = (unverified, supported_second_a, supported_b, supported_high_a)

    forward = ContextBuilder(max_chars=10_000).build(
        purpose="planner",
        ledger=EvidenceLedger(evidence=items),
        budget=_budget(),
        active_question_id="q1",
    )
    reversed_pack = ContextBuilder(max_chars=10_000).build(
        purpose="planner",
        ledger=EvidenceLedger(evidence=tuple(reversed(items))),
        budget=_budget(),
        active_question_id="q1",
    )

    expected = ("ev_a1", "ev_b1", "ev_a2", "ev_c1")
    assert forward.selected_evidence_ids == expected
    assert reversed_pack.selected_evidence_ids == expected
    assert forward.text == reversed_pack.text


def test_context_builder_preserves_a_complete_prioritized_conflict() -> None:
    active_conflict = EvidenceConflict(
        question_id="q-active",
        description="Sources disagree on the measured effect.",
        sources=(_source("a"), _source("b")),
    )
    other_conflict = EvidenceConflict(
        question_id="q-other",
        description="A secondary disagreement.",
        sources=(_source("c"),),
    )
    evidence = _evidence(
        "ev1",
        question_id="q-active",
        source_id="a",
    )
    ledger = EvidenceLedger(
        evidence=(evidence,),
        conflicts=(other_conflict, active_conflict),
    )
    wide = ContextBuilder(max_chars=10_000).build(
        purpose="worker",
        ledger=ledger,
        budget=_budget(),
        active_question_id="q-active",
    )
    active_line = next(
        line
        for line in wide.text.splitlines()
        if "Sources disagree on the measured effect." in line
    )

    pack = ContextBuilder(max_chars=len(active_line)).build(
        purpose="worker",
        ledger=ledger,
        budget=_budget(),
        active_question_id="q-active",
    )

    assert _records(pack.text) == [json.loads(active_line)]
    assert pack.included_conflict_count == 1
    assert pack.omitted_conflict_count == 1
    assert pack.selected_evidence_ids == ()
    assert pack.omitted_evidence_count == 1
    assert pack.used_chars == len(pack.text) == len(active_line)

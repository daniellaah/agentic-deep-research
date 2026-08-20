from dataclasses import replace

import pytest

from agentic_deep_research.models import (
    BudgetSnapshot,
    Evidence,
    EvidenceLedger,
    ResearchQuestion,
    Source,
    VerificationStatus,
)
from agentic_deep_research.policy import ResearchDecision, SufficiencyPolicy


def _item(
    question_id: str,
    *,
    url: str = "https://example.com/source",
    claim: str = "Evaluation improves reliability.",
    status: VerificationStatus = "unverified",
) -> Evidence:
    return Evidence(
        claim=claim,
        source=Source(title="Study", url=url),
        question_id=question_id,
        verification_status=status,
    )


def _decide(
    *,
    questions: tuple[ResearchQuestion, ...] | None = None,
    completed_question_ids: frozenset[str] | None = None,
    ledger: EvidenceLedger | None = None,
    round_start_ledger: EvidenceLedger | None = None,
    min_sources: int = 1,
    budget: BudgetSnapshot | None = None,
    revision: int = 0,
    proposed_questions: tuple[ResearchQuestion, ...] | None = None,
) -> ResearchDecision:
    return SufficiencyPolicy().decide(
        questions=(
            questions
            if questions is not None
            else (ResearchQuestion("q1", "Question"),)
        ),
        completed_question_ids=(
            completed_question_ids
            if completed_question_ids is not None
            else frozenset({"q1"})
        ),
        ledger=ledger if ledger is not None else EvidenceLedger(),
        round_start_ledger=(
            round_start_ledger
            if round_start_ledger is not None
            else EvidenceLedger()
        ),
        min_sources=min_sources,
        budget=budget or BudgetSnapshot(1, 1),
        revision=revision,
        proposed_questions=proposed_questions,
    )


def test_sufficient_evidence_wins_at_the_exact_budget_boundary() -> None:
    decision = _decide(
        ledger=EvidenceLedger(evidence=(_item("q1"),)),
        budget=BudgetSnapshot(
            tool_calls_remaining=0,
            research_steps_remaining=0,
        ),
    )

    assert decision.reason == "sufficient"
    assert decision.uncovered_question_ids == ()
    assert decision.additional_sources_needed == 0
    assert decision.new_evidence_count == 1


@pytest.mark.parametrize("status", ["unverified", "supported"])
def test_supported_and_unverified_evidence_are_usable(
    status: VerificationStatus,
) -> None:
    decision = _decide(
        ledger=EvidenceLedger(evidence=(_item("q1", status=status),)),
    )

    assert decision.reason == "sufficient"


@pytest.mark.parametrize("status", ["unsupported", "uncertain"])
def test_unusable_verification_statuses_do_not_cover_or_count(
    status: VerificationStatus,
) -> None:
    decision = _decide(
        ledger=EvidenceLedger(evidence=(_item("q1", status=status),)),
    )

    assert decision.reason == "continue"
    assert decision.uncovered_question_ids == ("q1",)
    assert decision.additional_sources_needed == 1
    assert decision.new_evidence_count == 0


def test_evidence_from_an_incomplete_question_is_not_usable() -> None:
    decision = _decide(
        completed_question_ids=frozenset(),
        ledger=EvidenceLedger(evidence=(_item("q1"),)),
    )

    assert decision.uncovered_question_ids == ("q1",)
    assert decision.additional_sources_needed == 1
    assert decision.new_evidence_count == 0


def test_only_evidence_bound_canonical_sources_satisfy_minimum() -> None:
    first = _item(
        "q1",
        url="HTTPS://Example.COM:443/study?utm_source=newsletter#results",
    )
    alias = _item("q2", url="https://example.com/study")
    unused = Source("Consulted only", "https://other.example/source")
    decision = _decide(
        questions=(
            ResearchQuestion("q1", "First question"),
            ResearchQuestion("q2", "Second question"),
        ),
        completed_question_ids=frozenset({"q1", "q2"}),
        ledger=EvidenceLedger(
            sources=(first.source, alias.source, unused),
            evidence=(first, alias),
        ),
        min_sources=2,
    )

    assert decision.reason == "continue"
    assert decision.uncovered_question_ids == ()
    assert decision.additional_sources_needed == 1
    assert decision.new_evidence_count == 1


def test_novelty_ignores_question_and_evidence_ids_and_normalizes_aliases() -> None:
    prior = replace(
        _item(
            "old",
            claim="Evaluation   improves reliability.",
            url="https://example.com/study?utm_campaign=launch",
        ),
        id="ev_old",
    )
    repeated = replace(
        _item(
            "new",
            claim="evaluation improves reliability.",
            url="HTTPS://EXAMPLE.COM:443/study#summary",
        ),
        id="ev_new",
    )
    decision = _decide(
        questions=(ResearchQuestion("new", "Find corroboration"),),
        completed_question_ids=frozenset({"old", "new"}),
        round_start_ledger=EvidenceLedger(evidence=(prior,)),
        ledger=EvidenceLedger(evidence=(prior, repeated)),
        revision=1,
    )

    assert decision.reason == "sufficient"
    assert decision.new_evidence_count == 0


def test_a_second_canonical_source_is_new_evidence() -> None:
    prior = _item("old", url="https://one.example/study")
    corroboration = _item("new", url="https://two.example/study")
    decision = _decide(
        questions=(ResearchQuestion("new", "Find corroboration"),),
        completed_question_ids=frozenset({"old", "new"}),
        round_start_ledger=EvidenceLedger(evidence=(prior,)),
        ledger=EvidenceLedger(evidence=(prior, corroboration)),
        min_sources=2,
        revision=1,
    )

    assert decision.reason == "sufficient"
    assert decision.new_evidence_count == 1


@pytest.mark.parametrize(
    ("budget", "expected_reason"),
    [
        (BudgetSnapshot(1, 0), "max_research_steps"),
        (BudgetSnapshot(0, 1), "max_tool_calls"),
        (BudgetSnapshot(0, 0), "max_research_steps"),
    ],
)
def test_budget_stops_have_a_fixed_priority(
    budget: BudgetSnapshot,
    expected_reason: str,
) -> None:
    decision = _decide(budget=budget, revision=1)

    assert decision.reason == expected_reason


def test_zero_yield_allows_an_initial_replan_but_stops_a_revision() -> None:
    initial = _decide(revision=0)
    revised = _decide(revision=1)

    assert initial.reason == "continue"
    assert revised.reason == "no_new_evidence"


def test_no_novel_proposed_questions_is_a_distinct_stop() -> None:
    stopped = _decide(revision=0, proposed_questions=())
    continuing = _decide(
        revision=0,
        proposed_questions=(ResearchQuestion("q2", "New question"),),
    )

    assert stopped.reason == "no_new_questions"
    assert continuing.reason == "continue"


def test_coverage_applies_only_to_the_active_round() -> None:
    prior = _item("prior", url="https://one.example/source")
    active = _item("active", url="https://two.example/source")
    decision = _decide(
        questions=(ResearchQuestion("active", "Replacement question"),),
        completed_question_ids=frozenset({"prior", "active"}),
        ledger=EvidenceLedger(evidence=(prior, active)),
        min_sources=2,
        revision=1,
    )

    assert decision.reason == "sufficient"
    assert decision.uncovered_question_ids == ()


def test_uncovered_questions_preserve_the_active_plan_order() -> None:
    decision = _decide(
        questions=(
            ResearchQuestion("q2", "Second"),
            ResearchQuestion("q1", "First"),
        ),
        completed_question_ids=frozenset(),
    )

    assert decision.uncovered_question_ids == ("q2", "q1")


@pytest.mark.parametrize(
    ("min_sources", "revision", "message"),
    [
        (0, 0, "min_sources must be at least 1"),
        (1, -1, "revision must not be negative"),
    ],
)
def test_invalid_policy_inputs_are_rejected(
    min_sources: int,
    revision: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _decide(min_sources=min_sources, revision=revision)

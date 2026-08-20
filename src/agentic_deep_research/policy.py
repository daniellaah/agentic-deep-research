"""Deterministic sufficiency and stopping decisions for research rounds."""

import re
from dataclasses import dataclass
from typing import Literal

from .evidence import canonicalize_url
from .models import BudgetSnapshot, Evidence, EvidenceLedger, ResearchQuestion

DecisionReason = Literal[
    "continue",
    "sufficient",
    "max_research_steps",
    "max_tool_calls",
    "no_new_evidence",
    "no_new_questions",
]

_USABLE_VERIFICATION_STATUSES = frozenset({"supported", "unverified"})


@dataclass(frozen=True)
class ResearchDecision:
    """One inspectable decision made after a bounded research round."""

    reason: DecisionReason
    uncovered_question_ids: tuple[str, ...]
    additional_sources_needed: int
    new_evidence_count: int


class SufficiencyPolicy:
    """Decide whether research is sufficient, bounded, or stalled."""

    def decide(
        self,
        *,
        questions: tuple[ResearchQuestion, ...],
        completed_question_ids: frozenset[str],
        ledger: EvidenceLedger,
        round_start_ledger: EvidenceLedger,
        min_sources: int,
        budget: BudgetSnapshot,
        revision: int,
        proposed_questions: tuple[ResearchQuestion, ...] | None = None,
    ) -> ResearchDecision:
        """Evaluate active-question coverage and evidence gained in one round."""
        if min_sources < 1:
            raise ValueError("min_sources must be at least 1")
        if revision < 0:
            raise ValueError("revision must not be negative")

        usable_evidence = _usable_evidence(ledger, completed_question_ids)
        covered_question_ids = {
            item.question_id for item in usable_evidence
        }
        uncovered_question_ids = tuple(
            question.id
            for question in questions
            if question.id not in covered_question_ids
        )

        current_fingerprints = _fingerprints(usable_evidence)
        round_start_fingerprints = _fingerprints(
            _usable_evidence(round_start_ledger, completed_question_ids)
        )
        new_evidence_count = len(
            current_fingerprints.difference(round_start_fingerprints)
        )
        distinct_source_count = len(
            {source_key for _, source_key in current_fingerprints}
        )
        additional_sources_needed = max(0, min_sources - distinct_source_count)

        if not uncovered_question_ids and additional_sources_needed == 0:
            reason: DecisionReason = "sufficient"
        elif budget.research_steps_remaining == 0:
            reason = "max_research_steps"
        elif budget.tool_calls_remaining == 0:
            reason = "max_tool_calls"
        elif revision > 0 and new_evidence_count == 0:
            reason = "no_new_evidence"
        elif proposed_questions is not None and not proposed_questions:
            reason = "no_new_questions"
        else:
            reason = "continue"

        return ResearchDecision(
            reason=reason,
            uncovered_question_ids=uncovered_question_ids,
            additional_sources_needed=additional_sources_needed,
            new_evidence_count=new_evidence_count,
        )


def _usable_evidence(
    ledger: EvidenceLedger,
    completed_question_ids: frozenset[str],
) -> tuple[Evidence, ...]:
    return tuple(
        item
        for item in ledger.evidence
        if item.question_id in completed_question_ids
        and item.verification_status in _USABLE_VERIFICATION_STATUSES
        and _fingerprint(item) is not None
    )


def _fingerprints(evidence: tuple[Evidence, ...]) -> frozenset[tuple[str, str]]:
    return frozenset(
        fingerprint
        for item in evidence
        if (fingerprint := _fingerprint(item)) is not None
    )


def _fingerprint(evidence: Evidence) -> tuple[str, str] | None:
    claim = re.sub(r"\s+", " ", evidence.claim).strip().casefold()
    source_url = canonicalize_url(
        evidence.source.canonical_url or evidence.source.url
    )
    if not claim or not source_url:
        return None
    return claim, source_url

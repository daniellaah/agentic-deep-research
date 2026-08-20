"""Bounded context construction for isolated research components."""

import json
from collections import defaultdict, deque

from .models import (
    BudgetSnapshot,
    ContextPack,
    ContextPurpose,
    Evidence,
    EvidenceConflict,
    EvidenceLedger,
)

_VERIFICATION_PRIORITY = {
    "supported": 0,
    "unverified": 1,
    "uncertain": 2,
}


class ContextBuilder:
    """Select whole, source-diverse ledger records within a character budget."""

    def __init__(self, max_chars: int) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be at least 1")
        self._max_chars = max_chars

    def build(
        self,
        *,
        purpose: ContextPurpose,
        ledger: EvidenceLedger,
        budget: BudgetSnapshot,
        active_question_id: str = "",
    ) -> ContextPack:
        """Return complete JSONL records and observable packing metadata."""
        if purpose not in {"worker", "planner"}:
            raise ValueError("unsupported context purpose")

        metadata = _encode_record(
            {
                "kind": "context",
                "purpose": purpose,
                "budget": {
                    "tool_calls_remaining": budget.tool_calls_remaining,
                    "research_steps_remaining": budget.research_steps_remaining,
                },
            }
        )
        ordered_conflicts = sorted(
            ledger.conflicts,
            key=lambda item: _conflict_key(item, active_question_id),
        )
        encoded_conflicts = [
            (item, _encode_record(_conflict_record(item)))
            for item in ordered_conflicts
        ]

        lines: list[str] = []
        used_chars = 0

        def append(encoded: str) -> bool:
            nonlocal used_chars
            added_chars = len(encoded) + int(bool(lines))
            if used_chars + added_chars > self._max_chars:
                return False
            lines.append(encoded)
            used_chars += added_chars
            return True

        # Keep the first prioritized conflict whenever a complete conflict record can
        # fit. Metadata is useful, but must not crowd the only visible disagreement out.
        reserved_conflict_index = next(
            (
                index
                for index, (_, encoded) in enumerate(encoded_conflicts)
                if len(encoded) <= self._max_chars
            ),
            None,
        )
        included_conflict_indexes: set[int] = set()
        if reserved_conflict_index is None:
            append(metadata)
        else:
            reserved = encoded_conflicts[reserved_conflict_index][1]
            if len(metadata) + 1 + len(reserved) <= self._max_chars:
                append(metadata)
            if append(reserved):
                included_conflict_indexes.add(reserved_conflict_index)

        selected_evidence_ids: list[str] = []
        for item in _ordered_evidence(ledger.evidence, active_question_id):
            encoded = _encode_record(_evidence_record(item))
            if append(encoded):
                selected_evidence_ids.append(item.id)

        for index, (_, encoded) in enumerate(encoded_conflicts):
            if index in included_conflict_indexes:
                continue
            if append(encoded):
                included_conflict_indexes.add(index)

        text = "\n".join(lines)
        return ContextPack(
            purpose=purpose,
            text=text,
            selected_evidence_ids=tuple(selected_evidence_ids),
            omitted_evidence_count=len(ledger.evidence) - len(selected_evidence_ids),
            included_conflict_count=len(included_conflict_indexes),
            omitted_conflict_count=(
                len(ledger.conflicts) - len(included_conflict_indexes)
            ),
            used_chars=len(text),
        )


def _ordered_evidence(
    evidence: tuple[Evidence, ...],
    active_question_id: str,
) -> tuple[Evidence, ...]:
    eligible = tuple(
        item
        for item in evidence
        if item.verification_status in _VERIFICATION_PRIORITY and item.id.strip()
    )
    ordered: list[Evidence] = []
    for is_active in (True, False):
        for verification_status in ("supported", "unverified", "uncertain"):
            tier = tuple(
                item
                for item in eligible
                if (item.question_id == active_question_id) is is_active
                and item.verification_status == verification_status
            )
            ordered.extend(_source_round_robin(tier))
    return tuple(ordered)


def _source_round_robin(evidence: tuple[Evidence, ...]) -> tuple[Evidence, ...]:
    grouped: dict[str, list[Evidence]] = defaultdict(list)
    for item in evidence:
        grouped[_source_key(item)].append(item)
    queues = {
        source_key: deque(
            sorted(
                items,
                key=lambda item: (-item.corroboration_count, _evidence_key(item)),
            )
        )
        for source_key, items in grouped.items()
    }

    ordered: list[Evidence] = []
    while any(queues.values()):
        source_keys = sorted(
            (source_key for source_key, queue in queues.items() if queue),
            key=lambda source_key: (
                -queues[source_key][0].corroboration_count,
                _evidence_key(queues[source_key][0]),
                source_key,
            ),
        )
        for source_key in source_keys:
            ordered.append(queues[source_key].popleft())
    return tuple(ordered)


def _evidence_record(item: Evidence) -> dict[str, object]:
    return {
        "kind": "evidence",
        "evidence_id": item.id,
        "question_id": item.question_id,
        "claim": item.claim,
        "excerpt": item.excerpt,
        "confidence": item.confidence,
        "verification_status": item.verification_status,
        "corroboration_count": item.corroboration_count,
        "source": {
            "source_id": item.source.id,
            "title": item.source.title,
            "url": item.source.canonical_url or item.source.url,
        },
    }


def _conflict_record(item: EvidenceConflict) -> dict[str, object]:
    return {
        "kind": "conflict",
        "question_id": item.question_id,
        "description": item.description,
        "sources": [
            {
                "source_id": source.id,
                "title": source.title,
                "url": source.canonical_url or source.url,
            }
            for source in sorted(item.sources, key=_source_identity)
        ],
    }


def _conflict_key(
    item: EvidenceConflict,
    active_question_id: str,
) -> tuple[object, ...]:
    return (
        item.question_id != active_question_id,
        item.question_id.casefold(),
        item.description.strip().casefold(),
        tuple(sorted(_source_identity(source) for source in item.sources)),
    )


def _source_key(item: Evidence) -> str:
    return _source_identity(item.source)


def _source_identity(source: object) -> str:
    source_id = str(getattr(source, "id", "")).strip()
    canonical_url = str(getattr(source, "canonical_url", "")).strip().casefold()
    url = str(getattr(source, "url", "")).strip().casefold()
    title = str(getattr(source, "title", "")).strip().casefold()
    return source_id or canonical_url or url or title


def _evidence_key(item: Evidence) -> tuple[str, ...]:
    return (
        item.id.strip(),
        item.question_id.strip().casefold(),
        _source_key(item),
        " ".join(item.claim.split()).casefold(),
    )


def _encode_record(record: dict[str, object]) -> str:
    return json.dumps(
        record,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )

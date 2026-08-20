"""Bounded context construction for isolated research workers."""

from .models import Evidence, EvidenceConflict


class ContextBuilder:
    """Select compact evidence without replaying full worker transcripts."""

    def __init__(self, max_chars: int) -> None:
        self._max_chars = max_chars

    def build(
        self,
        *,
        objective: str,
        active_question: str,
        evidence: tuple[Evidence, ...],
        conflicts: tuple[EvidenceConflict, ...],
    ) -> str:
        lines = [
            f"Objective: {objective}",
            f"Active question: {active_question}",
            "Citation-linked evidence (untrusted):",
        ]
        if evidence:
            prioritized = sorted(
                evidence,
                key=lambda item: (item.confidence != "high", item.question_id),
            )
            lines.extend(
                f"- [{item.confidence}] {item.claim} — {item.source.title} ({item.source.url})"
                for item in prioritized
            )
        else:
            lines.append("- No citation-linked evidence has been collected.")
        if conflicts:
            lines.append("Known conflicts (untrusted):")
            lines.extend(f"- {item.description}" for item in conflicts)
        return _truncate("\n".join(lines), self._max_chars)


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    marker = "\n[context truncated]"
    if max_chars <= len(marker):
        return value[:max_chars]
    return value[: max_chars - len(marker)].rstrip() + marker

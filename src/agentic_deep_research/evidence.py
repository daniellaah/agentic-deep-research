"""Normalized, deduplicated evidence accumulated across research workers."""

import re
from dataclasses import replace
from urllib.parse import urlparse

from .models import AgentRun, Citation, Evidence, EvidenceConflict, Source


class EvidenceStore:
    """Keep the durable evidence state used by the supervisor and final report."""

    def __init__(self) -> None:
        self._sources: dict[str, Source] = {}
        self._evidence: list[Evidence] = []
        self._evidence_keys: set[tuple[str, str]] = set()
        self._conflicts: list[EvidenceConflict] = []
        self._conflict_keys: set[tuple[str, str, tuple[str, ...]]] = set()

    @property
    def sources(self) -> tuple[Source, ...]:
        return tuple(self._sources.values())

    @property
    def evidence(self) -> tuple[Evidence, ...]:
        return tuple(self._evidence)

    @property
    def conflicts(self) -> tuple[EvidenceConflict, ...]:
        return tuple(self._conflicts)

    def add_run(self, question_id: str, run: AgentRun) -> None:
        """Merge one worker output and recalculate corroboration confidence."""
        for source in run.sources:
            self._add_source(source)
        for citation in run.citations:
            if not _valid_citation(run.report, citation):
                continue
            source = self._add_source(citation.source)
            claim = _claim_before(run.report, citation.start_index)
            key = (_normalize_claim(claim), source.url)
            if claim and key not in self._evidence_keys:
                self._evidence.append(
                    Evidence(
                        claim=claim,
                        source=source,
                        question_id=question_id,
                        excerpt=claim,
                    )
                )
                self._evidence_keys.add(key)
        for conflict in run.conflicts:
            sources = tuple(self._add_source(source) for source in conflict.sources)
            normalized = EvidenceConflict(
                question_id=conflict.question_id or question_id,
                description=conflict.description.strip(),
                sources=sources,
            )
            key = (
                normalized.question_id,
                normalized.description.casefold(),
                tuple(sorted(source.url for source in normalized.sources)),
            )
            if normalized.description and key not in self._conflict_keys:
                self._conflicts.append(normalized)
                self._conflict_keys.add(key)
        self._update_confidence()

    def for_question(self, question_id: str) -> tuple[Evidence, ...]:
        return tuple(item for item in self._evidence if item.question_id == question_id)

    def conflicts_for_question(self, question_id: str) -> tuple[EvidenceConflict, ...]:
        return tuple(item for item in self._conflicts if item.question_id == question_id)

    def covers(self, question_id: str) -> bool:
        return any(item.question_id == question_id for item in self._evidence)

    def _add_source(self, source: Source) -> Source:
        if not source.url:
            return source
        existing = self._sources.get(source.url)
        if existing is not None:
            return existing
        quality = source.quality
        if quality == "unknown" and _is_primary_domain(source.url):
            quality = "primary"
        normalized = replace(source, quality=quality)
        self._sources[source.url] = normalized
        return normalized

    def _update_confidence(self) -> None:
        sources_per_claim: dict[str, set[str]] = {}
        for item in self._evidence:
            sources_per_claim.setdefault(_normalize_claim(item.claim), set()).add(item.source.url)
        self._evidence = [
            replace(
                item,
                confidence=(
                    "high"
                    if len(sources_per_claim[_normalize_claim(item.claim)]) >= 2
                    else "medium"
                ),
            )
            for item in self._evidence
        ]


def _is_primary_domain(url: str) -> bool:
    hostname = (urlparse(url).hostname or "").casefold()
    return hostname.endswith((".gov", ".edu")) or hostname in {
        "arxiv.org",
        "www.arxiv.org",
    }


def _normalize_claim(claim: str) -> str:
    return re.sub(r"\s+", " ", claim).strip().casefold()


def _valid_citation(report: str, citation: Citation) -> bool:
    return 0 <= citation.start_index < citation.end_index <= len(report) and bool(
        citation.source.url
    )


def _claim_before(report: str, citation_start: int) -> str:
    prefix = report[:citation_start].rstrip()
    if not prefix:
        return ""
    search_end = len(prefix) - 1 if prefix[-1] in ".!?。！？" else len(prefix)
    boundary = max(
        prefix.rfind("\n", 0, search_end),
        prefix.rfind(".", 0, search_end),
        prefix.rfind("!", 0, search_end),
        prefix.rfind("?", 0, search_end),
        prefix.rfind("。", 0, search_end),
        prefix.rfind("！", 0, search_end),
        prefix.rfind("？", 0, search_end),
    )
    return prefix[boundary + 1 :].strip()

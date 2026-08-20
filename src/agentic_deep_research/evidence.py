"""Normalized, deduplicated evidence accumulated across research workers."""

import hashlib
import re
from dataclasses import replace
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .models import (
    AgentRun,
    Citation,
    Evidence,
    EvidenceConflict,
    EvidenceLedger,
    ResearchPacket,
    Source,
)

_TRACKING_QUERY_PARAMETERS = {
    "dclid",
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "msclkid",
}


class EvidenceStore:
    """Build a canonical evidence ledger from independently produced runs."""

    def __init__(self, initial_ledger: EvidenceLedger | None = None) -> None:
        self._sources: dict[str, Source] = {}
        self._evidence: list[Evidence] = []
        self._evidence_ids_by_key: dict[tuple[str, str, str], str] = {}
        self._conflicts: list[EvidenceConflict] = []
        self._conflicts_by_key: dict[
            tuple[str, str, tuple[str, ...]],
            EvidenceConflict,
        ] = {}
        if initial_ledger is not None:
            self._seed(initial_ledger)

    @property
    def sources(self) -> tuple[Source, ...]:
        return tuple(self._sources.values())

    @property
    def evidence(self) -> tuple[Evidence, ...]:
        return tuple(self._evidence)

    @property
    def conflicts(self) -> tuple[EvidenceConflict, ...]:
        return tuple(self._conflicts)

    def snapshot(self) -> EvidenceLedger:
        """Return the immutable state accumulated so far."""
        return EvidenceLedger(
            sources=self.sources,
            evidence=self.evidence,
            conflicts=self.conflicts,
        )

    def ingest_run(
        self,
        question_id: str,
        run: AgentRun,
        *,
        artifact_id: str = "",
    ) -> ResearchPacket:
        """Normalize one run and return its ledger-backed research packet."""
        artifact_id = artifact_id or f"finding:{question_id}"
        for source in run.sources:
            self._add_source(source)

        evidence_ids: list[str] = []
        for item in run.evidence:
            normalized = self._normalize_submitted_evidence(
                item,
                question_id=question_id,
                artifact_id=artifact_id,
            )
            if normalized is not None:
                _append_once(evidence_ids, self._add_evidence(normalized))

        for citation in run.citations:
            if not _valid_citation(run.report, citation):
                continue
            source = self._add_source(citation.source)
            claim, start_index, end_index = _claim_span_before(
                run.report,
                citation.start_index,
            )
            if not claim or not source.id:
                continue
            evidence = Evidence(
                claim=claim,
                source=source,
                question_id=question_id,
                excerpt="",
                id=_evidence_id(question_id, claim, source.id),
                verification_status="unverified",
                origin_artifact_id=artifact_id,
                origin_start_index=start_index,
                origin_end_index=end_index,
            )
            _append_once(evidence_ids, self._add_evidence(evidence))

        packet_conflicts: list[EvidenceConflict] = []
        for conflict in run.conflicts:
            normalized = self._add_conflict(question_id, conflict)
            if normalized is not None and normalized not in packet_conflicts:
                packet_conflicts.append(normalized)

        self._update_corroboration()
        packet_evidence = tuple(
            item
            for evidence_id in evidence_ids
            if (item := self.get(evidence_id)) is not None
        )
        return ResearchPacket(
            question_id=question_id,
            answer=run.report,
            evidence=packet_evidence,
            conflicts=tuple(packet_conflicts),
            artifact_id=artifact_id,
            status=run.status,
            stop_reason=run.stop_reason,
        )

    def add_run(self, question_id: str, run: AgentRun) -> None:
        """Compatibility wrapper for callers that do not need a packet."""
        self.ingest_run(question_id, run)

    def _seed(self, ledger: EvidenceLedger) -> None:
        """Restore normalized research material without resetting its provenance."""
        for source in ledger.sources:
            self._add_source(source)
        for item in ledger.evidence:
            source = self._add_source(item.source)
            claim = item.claim.strip()
            if not claim or not source.id:
                continue
            self._add_evidence(
                replace(
                    item,
                    claim=claim,
                    source=source,
                    id=item.id or _evidence_id(item.question_id, claim, source.id),
                )
            )
        for conflict in ledger.conflicts:
            self._add_conflict(conflict.question_id, conflict)
        self._update_corroboration()

    def get(self, evidence_id: str) -> Evidence | None:
        """Return one evidence record by its stable identifier."""
        return next((item for item in self._evidence if item.id == evidence_id), None)

    def for_question(self, question_id: str) -> tuple[Evidence, ...]:
        return tuple(item for item in self._evidence if item.question_id == question_id)

    def conflicts_for_question(self, question_id: str) -> tuple[EvidenceConflict, ...]:
        return tuple(item for item in self._conflicts if item.question_id == question_id)

    def has_candidate(self, question_id: str) -> bool:
        """Return whether any citation-linked candidate exists for a question."""
        return any(item.question_id == question_id for item in self._evidence)

    def covers(self, question_id: str) -> bool:
        """Compatibility alias for the former candidate-coverage check."""
        return self.has_candidate(question_id)

    def _normalize_submitted_evidence(
        self,
        item: Evidence,
        *,
        question_id: str,
        artifact_id: str,
    ) -> Evidence | None:
        claim = item.claim.strip()
        source = self._add_source(item.source)
        if not claim or not source.id:
            return None
        return replace(
            item,
            claim=claim,
            source=source,
            question_id=question_id,
            id=_evidence_id(question_id, claim, source.id),
            verification_status="unverified",
            origin_artifact_id=artifact_id,
        )

    def _add_evidence(self, evidence: Evidence) -> str:
        key = (
            evidence.question_id,
            _normalize_claim(evidence.claim),
            evidence.source.id,
        )
        existing_id = self._evidence_ids_by_key.get(key)
        if existing_id is not None:
            return existing_id
        self._evidence.append(evidence)
        self._evidence_ids_by_key[key] = evidence.id
        return evidence.id

    def _add_source(self, source: Source) -> Source:
        url = source.url.strip()
        if not url:
            return replace(source, id="", canonical_url="")
        canonical_url = canonicalize_url(url)
        existing = self._sources.get(canonical_url)
        if existing is not None:
            return existing
        normalized = replace(
            source,
            url=url,
            id=_source_id(canonical_url),
            canonical_url=canonical_url,
        )
        self._sources[canonical_url] = normalized
        return normalized

    def _add_conflict(
        self,
        question_id: str,
        conflict: EvidenceConflict,
    ) -> EvidenceConflict | None:
        sources = tuple(self._add_source(source) for source in conflict.sources)
        normalized = EvidenceConflict(
            question_id=conflict.question_id or question_id,
            description=conflict.description.strip(),
            sources=sources,
        )
        key = (
            normalized.question_id,
            normalized.description.casefold(),
            tuple(sorted(source.id or source.url for source in normalized.sources)),
        )
        existing = self._conflicts_by_key.get(key)
        if existing is not None:
            return existing
        if normalized.description:
            self._conflicts.append(normalized)
            self._conflicts_by_key[key] = normalized
            return normalized
        return None

    def _update_corroboration(self) -> None:
        sources_per_claim: dict[str, set[str]] = {}
        for item in self._evidence:
            sources_per_claim.setdefault(_normalize_claim(item.claim), set()).add(
                item.source.id
            )
        self._evidence = [
            replace(
                item,
                confidence=(
                    "high"
                    if len(sources_per_claim[_normalize_claim(item.claim)]) >= 2
                    else "medium"
                ),
                corroboration_count=len(
                    sources_per_claim[_normalize_claim(item.claim)]
                ),
            )
            for item in self._evidence
        ]


def canonicalize_url(url: str) -> str:
    """Return a conservative canonical form suitable for source identity."""
    value = url.strip()
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return value
    scheme = parsed.scheme.casefold()
    if scheme not in {"http", "https"} or not parsed.hostname:
        return value

    try:
        hostname = parsed.hostname.encode("idna").decode("ascii").casefold()
    except UnicodeError:
        hostname = parsed.hostname.casefold()
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    if port is not None and not (
        (scheme == "http" and port == 80)
        or (scheme == "https" and port == 443)
    ):
        hostname = f"{hostname}:{port}"

    query_items = [
        (name, value)
        for name, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not _is_tracking_parameter(name)
    ]
    query_items.sort(key=lambda item: (item[0], item[1]))
    return urlunsplit(
        (
            scheme,
            hostname,
            parsed.path or "/",
            urlencode(query_items, doseq=True),
            "",
        )
    )


def _is_tracking_parameter(name: str) -> bool:
    normalized = name.casefold()
    return normalized.startswith("utm_") or normalized in _TRACKING_QUERY_PARAMETERS


def _source_id(canonical_url: str) -> str:
    return _stable_id("src", canonical_url)


def _evidence_id(question_id: str, claim: str, source_id: str) -> str:
    return _stable_id("ev", question_id, _normalize_claim(claim), source_id)


def _stable_id(prefix: str, *parts: str) -> str:
    payload = "v1\n" + "\n".join(parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def _append_once(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def _normalize_claim(claim: str) -> str:
    return re.sub(r"\s+", " ", claim).strip().casefold()


def _valid_citation(report: str, citation: Citation) -> bool:
    return 0 <= citation.start_index < citation.end_index <= len(report) and bool(
        citation.source.url
    )


def _claim_before(report: str, citation_start: int) -> str:
    """Return the claim immediately preceding a citation marker."""
    claim, _, _ = _claim_span_before(report, citation_start)
    return claim


def _claim_span_before(report: str, citation_start: int) -> tuple[str, int, int]:
    prefix = report[:citation_start].rstrip()
    if not prefix:
        return "", -1, -1
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
    start_index = boundary + 1
    end_index = len(prefix)
    while start_index < end_index and report[start_index].isspace():
        start_index += 1
    while end_index > start_index and report[end_index - 1].isspace():
        end_index -= 1
    return report[start_index:end_index], start_index, end_index

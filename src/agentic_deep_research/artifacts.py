"""Deterministic, fail-closed exports of completed research results."""

import hashlib
import json
from dataclasses import dataclass
from typing import Literal
from urllib.parse import quote, urlsplit

from .checkpoint import RunState
from .models import Evidence, ResearchBrief, ResearchResult, Source

ExportFormat = Literal["json", "markdown"]


class ExportIntegrityError(ValueError):
    """The persisted result is not safe to expose as a public artifact."""


@dataclass(frozen=True)
class ExportedArtifact:
    """One deterministic HTTP-ready result artifact."""

    body: bytes
    media_type: str
    filename: str
    etag: str


def export_result(state: RunState, *, format: ExportFormat) -> ExportedArtifact:
    """Export a completed run in one of the supported public formats."""
    if format == "json":
        return export_json(state)
    if format == "markdown":
        return export_markdown(state)
    raise ValueError(f"unsupported export format: {format}")


def export_json(state: RunState) -> ExportedArtifact:
    """Return a deterministic JSON projection without checkpoint internals."""
    result = _validated_result(state)
    payload = _public_projection(state, result)
    body = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return _artifact(
        body=body,
        media_type="application/json; charset=utf-8",
        filename=f"{state.run_id}.json",
    )


def export_markdown(state: RunState) -> ExportedArtifact:
    """Return a deterministic Markdown report with safe citation links."""
    result = _validated_result(state)
    warning = _quality_warning(result)
    lines = [
        f"# {_escape_markdown_text(result.topic)}",
        "",
        f"Research status: `{_escape_code(result.status)}`",
        f"Stop reason: `{_escape_code(result.stop_reason)}`",
    ]
    if warning is not None:
        lines.extend(
            [
                "",
                "> [!WARNING]",
                f"> {_escape_markdown_text(warning)}",
            ]
        )
    lines.extend(
        [
            "",
            "## Report",
            "",
            _safe_citation_markdown(result).strip(),
            "",
            "## Sources",
            "",
        ]
    )
    if result.sources:
        lines.extend(
            f"{index}. {_markdown_source(source)}"
            for index, source in enumerate(result.sources, start=1)
        )
    else:
        lines.append("No sources were recorded.")
    body = ("\n".join(lines).rstrip() + "\n").replace("\r\n", "\n").replace(
        "\r", "\n"
    ).encode("utf-8")
    return _artifact(
        body=body,
        media_type="text/markdown; charset=utf-8",
        filename=f"{state.run_id}.md",
    )


def _validated_result(state: RunState) -> ResearchResult:
    if state.status != "completed" or state.result is None:
        raise ExportIntegrityError("only completed runs with a result can be exported")
    result = state.result
    sources = _source_registry(result.sources)
    evidence = _evidence_registry(result.evidence, sources)
    _validate_citations(result, sources, evidence)
    _validate_conflicts(result, sources)
    _validate_checks(result, sources, evidence)
    return result


def _source_registry(sources: tuple[Source, ...]) -> dict[str, Source]:
    registry: dict[str, Source] = {}
    for source in sources:
        if not source.id:
            raise ExportIntegrityError("every exported source must have a stable id")
        if not source.url.strip():
            raise ExportIntegrityError(f"source {source.id} has no URL")
        if source.id in registry:
            raise ExportIntegrityError(f"duplicate source id: {source.id}")
        registry[source.id] = source
    return registry


def _evidence_registry(
    evidence: tuple[Evidence, ...],
    sources: dict[str, Source],
) -> dict[str, Evidence]:
    registry: dict[str, Evidence] = {}
    for item in evidence:
        if not item.id:
            raise ExportIntegrityError("every exported evidence record must have a stable id")
        if item.id in registry:
            raise ExportIntegrityError(f"duplicate evidence id: {item.id}")
        _require_source(item.source, sources, owner=f"evidence {item.id}")
        if not item.claim.strip():
            raise ExportIntegrityError(f"evidence {item.id} has no claim")
        registry[item.id] = item
    return registry


def _validate_citations(
    result: ResearchResult,
    sources: dict[str, Source],
    evidence: dict[str, Evidence],
) -> None:
    ordered = sorted(
        result.citations,
        key=lambda item: (item.start_index, item.end_index),
    )
    previous_end = 0
    for index, citation in enumerate(ordered, start=1):
        if type(citation.start_index) is not int or type(citation.end_index) is not int:
            raise ExportIntegrityError(f"citation {index} offsets must be integers")
        if not (
            0
            <= citation.start_index
            < citation.end_index
            <= len(result.raw_report)
        ):
            raise ExportIntegrityError(
                f"citation {index} has invalid citation_text offsets"
            )
        if index > 1 and citation.start_index < previous_end:
            raise ExportIntegrityError("citation ranges must not overlap")
        previous_end = citation.end_index
        source = _require_source(citation.source, sources, owner=f"citation {index}")
        if citation.evidence_id:
            item = evidence.get(citation.evidence_id)
            if item is None:
                raise ExportIntegrityError(
                    f"citation {index} references unknown evidence: "
                    f"{citation.evidence_id}"
                )
            if item.source.id != source.id:
                raise ExportIntegrityError(
                    f"citation {index} source does not match its evidence"
                )


def _validate_conflicts(result: ResearchResult, sources: dict[str, Source]) -> None:
    for conflict_index, conflict in enumerate(result.conflicts, start=1):
        for source in conflict.sources:
            _require_source(
                source,
                sources,
                owner=f"conflict {conflict_index}",
            )


def _validate_checks(
    result: ResearchResult,
    sources: dict[str, Source],
    evidence: dict[str, Evidence],
) -> None:
    for index, check in enumerate(result.citation_checks, start=1):
        source = _require_source(check.source, sources, owner=f"citation check {index}")
        if not check.evidence_id:
            continue
        item = evidence.get(check.evidence_id)
        if item is None:
            raise ExportIntegrityError(
                f"citation check {index} references unknown evidence: "
                f"{check.evidence_id}"
            )
        if item.source.id != source.id:
            raise ExportIntegrityError(
                f"citation check {index} source does not match its evidence"
            )


def _require_source(
    candidate: Source,
    sources: dict[str, Source],
    *,
    owner: str,
) -> Source:
    source = sources.get(candidate.id)
    if source is None:
        raise ExportIntegrityError(f"{owner} references an unknown source")
    if _source_projection(candidate) != _source_projection(source):
        raise ExportIntegrityError(f"{owner} contains inconsistent source data")
    return source


def _source_projection(source: Source) -> tuple[str, str, str, str, str]:
    return (
        source.id,
        source.title,
        source.url,
        source.canonical_url,
        source.quality,
    )


def _public_projection(state: RunState, result: ResearchResult) -> dict[str, object]:
    warning = _quality_warning(result)
    return {
        "schema": "agentic-deep-research/research-artifact@1",
        "schema_version": 1,
        "run": {
            "created_at": state.created_at,
            "run_id": state.run_id,
            "runtime_status": state.status,
            "updated_at": state.updated_at,
        },
        "request": {
            "brief": _brief_projection(state.request.brief),
            "budget": {
                "max_context_chars": state.request.budget.max_context_chars,
                "max_output_tokens": state.request.budget.max_output_tokens,
                "max_parallel_workers": state.request.budget.max_parallel_workers,
                "max_research_steps": state.request.budget.max_research_steps,
                "max_revision_rounds": state.request.budget.max_revision_rounds,
                "max_tool_calls": state.request.budget.max_tool_calls,
                "max_verification_tool_calls": (
                    state.request.budget.max_verification_tool_calls
                ),
            },
            "language": state.request.language,
            "min_sources": state.request.min_sources,
            "require_citations": state.request.require_citations,
            "topic": state.request.topic,
        },
        "research": {
            "citation_checks": [
                {
                    "claim": item.claim,
                    "claim_id": item.claim_id or None,
                    "evidence_id": item.evidence_id or None,
                    "reason": item.reason,
                    "source_id": item.source.id,
                    "status": item.status,
                }
                for item in result.citation_checks
            ],
            "citation_index_semantics": {
                "end": "exclusive",
                "start": "inclusive",
                "text_field": "citation_text",
                "unit": "unicode_code_point",
            },
            "citation_text": result.raw_report,
            "citations": [
                {
                    "end_index": item.end_index,
                    "evidence_id": item.evidence_id or None,
                    "marker": result.raw_report[item.start_index : item.end_index],
                    "source_id": item.source.id,
                    "start_index": item.start_index,
                }
                for item in result.citations
            ],
            "conflicts": [
                {
                    "description": item.description,
                    "question_id": item.question_id,
                    "source_ids": [source.id for source in item.sources],
                }
                for item in result.conflicts
            ],
            "evidence": [
                {
                    "claim": item.claim,
                    "confidence": item.confidence,
                    "corroboration_count": item.corroboration_count,
                    "excerpt": item.excerpt,
                    "id": item.id,
                    "origin": {
                        "artifact_id": item.origin_artifact_id or None,
                        "end_index": (
                            item.origin_end_index if item.origin_end_index >= 0 else None
                        ),
                        "start_index": (
                            item.origin_start_index
                            if item.origin_start_index >= 0
                            else None
                        ),
                    },
                    "question_id": item.question_id,
                    "source_id": item.source.id,
                    "verification_status": item.verification_status,
                }
                for item in result.evidence
            ],
            "language": state.request.language,
            "plan": _plan_projection(result),
            "quality": {
                "status": result.status,
                "stop_reason": result.stop_reason,
                "warning": warning,
            },
            "report_markdown": _safe_citation_markdown(result),
            "revision_count": result.revision_count,
            "sources": [
                {
                    "canonical_url": item.canonical_url,
                    "id": item.id,
                    "quality": item.quality,
                    "title": item.title,
                    "url": item.url,
                }
                for item in result.sources
            ],
            "topic": result.topic,
            "usage": {
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "total_tokens": result.usage.total_tokens,
            },
        },
    }


def _brief_projection(brief: ResearchBrief | None) -> dict[str, object] | None:
    if brief is None:
        return None
    return {
        "constraints": list(brief.constraints),
        "deliverable": brief.deliverable,
        "objective": brief.objective,
        "research_question": brief.research_question,
        "scope_exclusions": list(brief.scope_exclusions),
        "scope_inclusions": list(brief.scope_inclusions),
        "success_criteria": list(brief.success_criteria),
    }


def _plan_projection(result: ResearchResult) -> dict[str, object] | None:
    if result.plan is None:
        return None
    return {
        "objective": result.plan.objective,
        "questions": [
            {
                "id": item.id,
                "priority": item.priority,
                "question": item.question,
                "rationale": item.rationale,
            }
            for item in result.plan.questions
        ],
        "revision": result.plan.revision,
    }


def _quality_warning(result: ResearchResult) -> str | None:
    if result.status == "completed":
        return None
    return (
        "This research result requires review: "
        f"status={result.status}, stop_reason={result.stop_reason}."
    )


def _safe_citation_markdown(result: ResearchResult) -> str:
    parts: list[str] = []
    cursor = 0
    for citation in sorted(result.citations, key=lambda item: item.start_index):
        parts.append(_escape_report_links(result.raw_report[cursor : citation.start_index]))
        parts.append(_markdown_source(citation.source))
        cursor = citation.end_index
    parts.append(_escape_report_links(result.raw_report[cursor:]))
    return "".join(parts)


def _markdown_source(source: Source) -> str:
    title = _escape_markdown_text(source.title.strip() or source.url)
    target = _safe_http_target(source.url)
    if target is None:
        return f"{title} (link omitted: unsupported URL)"
    return f"[{title}]({target})"


def _safe_http_target(url: str) -> str | None:
    value = url.strip()
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
    except ValueError:
        return None
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        return None
    return quote(
        value,
        safe=":/?#[]@!$&'*+,;=%-._~",
    )


def _escape_report_links(value: str) -> str:
    """Disable model-authored Markdown/HTML links while preserving basic layout."""
    return (
        value.replace("\\", "\\\\")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .replace("<", "\\<")
        .replace(">", "\\>")
    )


def _escape_markdown_text(value: str) -> str:
    escaped = value.replace("\\", "\\\\")
    for character in "`*_{}[]<>()#+-.!|":
        escaped = escaped.replace(character, f"\\{character}")
    return " ".join(escaped.splitlines())


def _escape_code(value: str) -> str:
    return value.replace("`", "'").replace("\r", " ").replace("\n", " ")


def _artifact(*, body: bytes, media_type: str, filename: str) -> ExportedArtifact:
    digest = hashlib.sha256(body).hexdigest()
    return ExportedArtifact(
        body=body,
        media_type=media_type,
        filename=filename,
        etag=f'"sha256-{digest}"',
    )

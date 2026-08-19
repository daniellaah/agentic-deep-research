"""Agent execution through the OpenAI Responses API."""

from collections.abc import Mapping
from typing import Any, Protocol

from openai import OpenAI

from .models import (
    AgentRun,
    Citation,
    ResearchBudget,
    ResearchStep,
    Source,
    TokenUsage,
)


class AgentRunner(Protocol):
    """Interface for executing agent tasks."""

    def run(
        self,
        *,
        instructions: str,
        task: str,
        budget: ResearchBudget,
    ) -> AgentRun:
        """Execute a bounded agent task and return its report and metadata."""
        ...


class OpenAIAgentRunner:
    """Execute agent tasks through the OpenAI Responses API."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    def run(
        self,
        *,
        instructions: str,
        task: str,
        budget: ResearchBudget,
    ) -> AgentRun:
        """Run agentic web research and normalize the Responses API output."""
        response = self._client.responses.create(
            model=self._model,
            instructions=instructions,
            input=task,
            tools=[{"type": "web_search", "search_context_size": "medium"}],
            tool_choice="required",
            include=["web_search_call.action.sources"],
            max_tool_calls=budget.max_tool_calls,
            max_output_tokens=budget.max_output_tokens,
        )

        sources: list[Source] = []
        sources_by_url: dict[str, Source] = {}
        citations: list[Citation] = []
        trace: list[ResearchStep] = []

        for item in _value(response, "output", ()) or ():
            item_type = _value(item, "type", "")
            if item_type == "web_search_call":
                action = _value(item, "action", {})
                action_type = _value(action, "type", "web_search")
                trace.append(
                    ResearchStep(
                        action=action_type,
                        detail=_action_detail(action),
                    )
                )
                for raw_source in _value(action, "sources", ()) or ():
                    _add_source(raw_source, sources, sources_by_url)

            if item_type == "message":
                for content in _value(item, "content", ()) or ():
                    if _value(content, "type", "") != "output_text":
                        continue
                    for annotation in _value(content, "annotations", ()) or ():
                        if _value(annotation, "type", "") != "url_citation":
                            continue
                        source = _add_source(annotation, sources, sources_by_url)
                        citations.append(
                            Citation(
                                source=source,
                                start_index=_value(annotation, "start_index", 0),
                                end_index=_value(annotation, "end_index", 0),
                            )
                        )

        status = _value(response, "status", "completed") or "completed"
        incomplete_details = _value(response, "incomplete_details", None)
        stop_reason = _value(incomplete_details, "reason", None) or status
        raw_usage = _value(response, "usage", None)

        return AgentRun(
            report=response.output_text,
            sources=tuple(sources),
            citations=tuple(citations),
            trace=tuple(trace),
            status=status,
            stop_reason=stop_reason,
            usage=TokenUsage(
                input_tokens=_value(raw_usage, "input_tokens", 0),
                output_tokens=_value(raw_usage, "output_tokens", 0),
                total_tokens=_value(raw_usage, "total_tokens", 0),
            ),
        )


def _value(value: object, name: str, default: Any) -> Any:
    """Read a field from an SDK model, mapping, or test double."""
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _add_source(
    raw_source: object,
    sources: list[Source],
    sources_by_url: dict[str, Source],
) -> Source:
    """Add a source once while preserving first-seen order."""
    url = _value(raw_source, "url", "")
    existing = sources_by_url.get(url)
    if existing is not None:
        return existing

    source = Source(
        title=_value(raw_source, "title", "") or url,
        url=url,
    )
    sources.append(source)
    sources_by_url[url] = source
    return source


def _action_detail(action: object) -> str:
    """Return the useful human-readable part of a web-search action."""
    action_type = _value(action, "type", "")
    if action_type == "search":
        queries = _value(action, "queries", None)
        if queries:
            return " | ".join(queries)
        return _value(action, "query", "")
    if action_type == "open_page":
        return _value(action, "url", "")
    if action_type == "find_in_page":
        url = _value(action, "url", "")
        pattern = _value(action, "pattern", "")
        return f"{url} :: {pattern}".strip(" :")
    return action_type

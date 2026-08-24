"""Scope and run supervised research before refining a final report."""

import argparse
import ipaddress
import json
import os
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Literal
from urllib.parse import unquote, urlsplit

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from agent_instructions import (
    BRIEF_INSTRUCTIONS,
    BRIEF_REVISION_INSTRUCTIONS,
    CLARIFICATION_INSTRUCTIONS,
    CRITIC_INSTRUCTIONS,
    RESEARCH_BUDGET_EXHAUSTED_INPUT,
    RESEARCH_INSTRUCTIONS,
    RESEARCH_SUMMARY_INSTRUCTIONS,
    RESEARCH_SUMMARY_REQUEST_INPUT,
    REVISE_INSTRUCTIONS,
    RESUMED_RESEARCH_STATE_NOTICE,
    SUPERVISOR_INSTRUCTIONS,
    WRITE_INSTRUCTIONS,
)
from agent_tools import (
    MAX_SOURCE_CONTENT_CHARACTERS,
    MAX_SOURCE_READ_QUERY_CHARACTERS,
    RESEARCH_TOOLS,
    execute_research_tool,
)

MAX_WORKER_TURNS = 15
MAX_WORKER_TOOL_CALLS = 10
MAX_WORKER_SOURCE_READS = 4
MAX_RESEARCH_WORKERS = 4
MAX_SUPERVISOR_OUTPUT_TOKENS = 4000
MAX_CONTEXT_SESSIONS = 3
MAX_CONTEXT_SUMMARIES = 2
CONTEXT_TRIGGER_TOKENS = 12000
CONTEXT_HARD_LIMIT_TOKENS = 20000
MAX_SUMMARY_OUTPUT_TOKENS = 8000
FUNCTION_OUTPUT_PROJECTION_OVERHEAD_TOKENS = 256
MAX_USER_INPUT_CHARACTERS = 2000
MAX_AGENT_ERROR_CHARACTERS = 300
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
OUTPUT_WIDTH = 80
SEARCH_TOOL_NAMES = {"tavily_search_tool", "arxiv_search_tool"}
READ_SOURCE_TOOL_NAME = "read_source_tool"
LOCAL_HOST_NAMES = {"localhost", "local", "internal"}
LOCAL_HOST_SUFFIXES = (".localhost", ".local", ".internal")


ClarificationQuestion = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=300),
]
BriefText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=300),
]
BriefObjective = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=800),
]
EvidenceTarget = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
]
SummaryCompletedWork = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
]
SummarySourceId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=20),
]
SummaryKeyEvidence = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=600),
]
SummaryLimitation = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=300),
]
SummaryUnresolvedQuestion = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=400),
]
SummaryNextAction = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=300),
]


class AgentRunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentTerminationReason(StrEnum):
    COMPLETED = "completed"
    TOOL_LIMIT = "tool_limit"
    TURN_LIMIT = "turn_limit"
    CONTEXT_LIMIT = "context_limit"
    REFUSAL = "refusal"
    MODEL_ERROR = "model_error"
    TOOL_ERROR = "tool_error"
    CANCELLED = "cancelled"


class LLMProvider(StrEnum):
    OPENAI = "openai"
    DEEPSEEK = "deepseek"


class RunCancelled(Exception):
    """Stop a run after application cancellation or terminal input ends."""


class ClarificationAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    questions: Annotated[
        list[ClarificationQuestion],
        Field(min_length=0, max_length=3),
    ]


class ResearchBrief(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    objective: BriefObjective
    audience: BriefText
    scope: Annotated[list[BriefText], Field(min_length=1, max_length=6)]
    exclusions: Annotated[list[BriefText], Field(min_length=0, max_length=6)]
    time_horizon: BriefText
    source_preferences: Annotated[
        list[BriefText],
        Field(min_length=1, max_length=6),
    ]
    output_requirements: Annotated[
        list[BriefText],
        Field(min_length=1, max_length=6),
    ]
    success_criteria: Annotated[
        list[BriefText],
        Field(min_length=1, max_length=6),
    ]


class ResearchTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    title: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=100),
    ]
    research_question: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=600),
    ]
    evidence_targets: Annotated[
        list[EvidenceTarget],
        Field(min_length=1, max_length=3),
    ]


class SupervisorDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    next_tasks: Annotated[
        list[ResearchTask],
        Field(min_length=0, max_length=1),
    ]


class SummarySource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: SummarySourceId
    evidence_level: Literal["discovery_snippet", "selected_source"]
    key_evidence: SummaryKeyEvidence
    limitations: SummaryLimitation


class ResearchStateSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    completed_work: Annotated[
        list[SummaryCompletedWork],
        Field(min_length=1, max_length=8),
    ]
    visited_sources: Annotated[list[SummarySource], Field(max_length=8)]
    unresolved_questions: Annotated[
        list[SummaryUnresolvedQuestion],
        Field(max_length=6),
    ]
    next_actions: Annotated[
        list[SummaryNextAction],
        Field(min_length=1, max_length=4),
    ]


@dataclass(frozen=True)
class AgentRunRequest:
    worker_number: int
    approved_brief: str
    task: ResearchTask
    llm_provider: LLMProvider
    model_name: str


@dataclass
class AgentRunState:
    request: AgentRunRequest
    input_items: list
    model_turns_used: int = 0
    tool_calls_used: int = 0
    source_urls: dict[str, str] = field(default_factory=dict)
    source_reads_used: int = 0
    read_source_ids: set[str] = field(default_factory=set)
    context_session_number: int = 1
    context_summaries_used: int = 0
    latest_summary: ResearchStateSummary | None = None
    peak_input_tokens: int = 0
    peak_projected_input_tokens: int = 0
    status: AgentRunStatus = AgentRunStatus.RUNNING
    termination_reason: AgentTerminationReason | None = None
    notes: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class AgentRunResult:
    worker_number: int
    task: ResearchTask
    llm_provider: LLMProvider
    model_name: str
    status: AgentRunStatus
    termination_reason: AgentTerminationReason
    notes: str | None
    error_message: str | None
    model_turns_used: int
    model_turn_limit: int
    tool_calls_used: int
    tool_call_limit: int
    source_reads_used: int
    source_read_limit: int
    context_sessions_used: int
    context_session_limit: int
    context_summaries_used: int
    context_summary_limit: int
    peak_input_tokens: int
    peak_projected_input_tokens: int
    context_trigger_tokens: int
    context_hard_limit_tokens: int


def print_progress(label, subject, status, detail=None, indent=0):
    message = f"{'  ' * indent}[{label}] {subject} | {status.upper()}"
    if detail:
        message += f" | {detail}"
    print(message)


def print_block(title, content):
    separator = "=" * OUTPUT_WIDTH
    print(f"\n{separator}\n{title}\n{separator}")
    print(content.strip())
    print(f"{separator}\n")


def print_error(message):
    print(f"[ERROR] {message}", file=sys.stderr)


def read_terminal_input(prompt, *, choices=None, max_length=None):
    while True:
        try:
            value = input(prompt).strip()
        except EOFError as error:
            raise RunCancelled("Standard input ended before approval.") from error

        if not value:
            print("[INPUT] A non-empty response is required.")
            continue
        if max_length is not None and len(value) > max_length:
            print(f"[INPUT] Response must be at most {max_length} characters.")
            continue
        if choices is not None:
            normalized = value.lower()
            if normalized not in choices:
                print(f"[INPUT] Enter one of: {', '.join(choices)}.")
                continue
            return normalized
        return value


def format_worker_count(worker_count):
    noun = "worker" if worker_count == 1 else "workers"
    return f"{worker_count} {noun}"


def compact_agent_error(message):
    compacted = " ".join(str(message).split())
    if not compacted:
        compacted = "Research Worker failed."
    return compacted[:MAX_AGENT_ERROR_CHARACTERS]


def format_model_exception(error):
    details = [type(error).__name__]
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        details.append(f"status {status_code}")
    error_code = getattr(error, "code", None)
    if isinstance(error_code, str) and error_code.strip():
        details.append(f"code {error_code.strip()}")
    return compact_agent_error(
        f"Research model request failed ({', '.join(details)})."
    )


def format_response_failure(response):
    status = getattr(response, "status", None) or "unknown"
    details = [f"status {status}"]
    incomplete_details = getattr(response, "incomplete_details", None)
    incomplete_reason = getattr(incomplete_details, "reason", None)
    if isinstance(incomplete_reason, str) and incomplete_reason.strip():
        details.append(f"reason {incomplete_reason.strip()}")
    response_error = getattr(response, "error", None)
    error_code = getattr(response_error, "code", None)
    if isinstance(error_code, str) and error_code.strip():
        details.append(f"code {error_code.strip()}")
    return compact_agent_error(
        f"Research model response did not complete ({', '.join(details)})."
    )


def canonical_source_destination(source_url):
    if not isinstance(source_url, str) or not source_url.strip():
        raise ValueError("Source URL must be a non-empty string.")
    source_url = source_url.strip()
    if any(character.isspace() for character in source_url):
        raise ValueError("Source URL must not contain whitespace.")
    if any(ord(character) < 32 or ord(character) == 127 for character in source_url):
        raise ValueError("Source URL must not contain control characters.")

    parsed = urlsplit(source_url)
    scheme = parsed.scheme.casefold()
    if scheme not in {"http", "https"}:
        raise ValueError("Source URL must use HTTP or HTTPS.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Source URL must not contain credentials.")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Source URL must contain a hostname.")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("Source URL contains an invalid port.") from error

    normalized_host = hostname.rstrip(".").casefold()
    if not normalized_host:
        raise ValueError("Source URL must contain a hostname.")
    try:
        address = ipaddress.ip_address(normalized_host)
    except ValueError:
        try:
            normalized_host = normalized_host.encode("idna").decode("ascii")
        except UnicodeError as error:
            raise ValueError("Source URL contains an invalid hostname.") from error
        if normalized_host in LOCAL_HOST_NAMES or normalized_host.endswith(
            LOCAL_HOST_SUFFIXES
        ):
            raise ValueError("Source URL uses a local hostname.")
    else:
        if not address.is_global:
            raise ValueError("Source URL uses a non-global IP address.")

    if unquote(parsed.path).casefold().endswith(".pdf"):
        raise ValueError("PDF source reading is not supported.")

    if (scheme, port) in {("http", 80), ("https", 443)}:
        port = None
    return (
        scheme,
        normalized_host,
        port,
        parsed.path or "/",
        parsed.query,
    )


def get_source_hostname(source_url):
    return canonical_source_destination(source_url)[1]


def register_search_sources(state, tool_result):
    if not isinstance(tool_result, list):
        raise TypeError("Search tool result must be a list.")

    canonical_ids = {
        canonical_source_destination(source_url): source_id
        for source_id, source_url in state.source_urls.items()
    }
    new_source_count = 0
    for result in tool_result:
        if not isinstance(result, dict):
            raise TypeError("Search result must be an object.")
        result.pop("source_id", None)
        if "error" in result:
            continue
        source_url = result.get("url")
        try:
            destination = canonical_source_destination(source_url)
        except (TypeError, ValueError):
            continue

        source_id = canonical_ids.get(destination)
        if source_id is None:
            source_id = f"S{len(state.source_urls) + 1}"
            state.source_urls[source_id] = source_url.strip()
            canonical_ids[destination] = source_id
            new_source_count += 1
        result["source_id"] = source_id
    return new_source_count


def prepare_source_read(state, arguments):
    if set(arguments) != {"source_id", "query"}:
        raise ValueError(
            "read_source_tool accepts only source_id and query."
        )
    source_id = arguments.get("source_id")
    query = arguments.get("query")
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("source_id must be a non-empty string.")
    source_id = source_id.strip()
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if len(query.strip()) > MAX_SOURCE_READ_QUERY_CHARACTERS:
        raise ValueError(
            "query must be at most "
            f"{MAX_SOURCE_READ_QUERY_CHARACTERS} characters."
        )
    if source_id not in state.source_urls:
        raise ValueError("source_id was not returned by this Worker's searches.")
    if state.source_reads_used >= MAX_WORKER_SOURCE_READS:
        raise ValueError("Source read limit reached.")

    source_url = state.source_urls[source_id]
    canonical_source_destination(source_url)
    state.source_reads_used += 1
    return source_id, source_url


def normalize_source_read_result(state, source_id, source_url, tool_result):
    if not isinstance(tool_result, list) or len(tool_result) != 1:
        raise ValueError("Source read must return exactly one result.")
    result = tool_result[0]
    if not isinstance(result, dict):
        raise TypeError("Source read result must be an object.")
    if canonical_source_destination(result.get("url")) != (
        canonical_source_destination(source_url)
    ):
        raise ValueError("Source extraction returned a different destination.")

    content = result.get("content")
    content_characters = result.get("content_characters")
    truncated = result.get("truncated")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Source read returned no content.")
    if (
        isinstance(content_characters, bool)
        or not isinstance(content_characters, int)
        or content_characters != len(content)
    ):
        raise ValueError("Source read returned inconsistent content size.")
    if not isinstance(truncated, bool):
        raise TypeError("Source read returned an invalid truncation flag.")

    bounded_content = content[:MAX_SOURCE_CONTENT_CHARACTERS]
    bounded_truncated = truncated or len(content) > len(bounded_content)

    return [
        {
            "source_id": source_id,
            "url": source_url,
            "content": bounded_content,
            "content_characters": len(bounded_content),
            "truncated": bounded_truncated,
            "source_reads_remaining": (
                MAX_WORKER_SOURCE_READS - state.source_reads_used
            ),
        }
    ]


def get_refusal_text(response):
    output_items = getattr(response, "output", None)
    if not isinstance(output_items, list):
        return None
    for output in output_items:
        if getattr(output, "type", None) != "message":
            continue
        for content in getattr(output, "content", []):
            if getattr(content, "type", None) == "refusal":
                return getattr(content, "refusal", "")
    return None


def validate_response_usage(response):
    usage = getattr(response, "usage", None)
    values = {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Response usage {name} must be a non-negative integer.")
    if values["total_tokens"] != (
        values["input_tokens"] + values["output_tokens"]
    ):
        raise ValueError("Response usage token totals are inconsistent.")
    return values["input_tokens"], values["output_tokens"]


def project_next_research_input(
    input_tokens,
    output_tokens,
    function_call_outputs=None,
):
    for name, value in {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Projection {name} must be a non-negative integer.")

    projected_tokens = input_tokens + output_tokens
    if function_call_outputs is None:
        function_call_outputs = []
    if not isinstance(function_call_outputs, list):
        raise TypeError("Projection function_call_outputs must be a list.")
    for function_call_output in function_call_outputs:
        if not isinstance(function_call_output, dict):
            raise TypeError(
                "Each projected function_call_output must be an object."
            )
        compact_output = json.dumps(
            function_call_output,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        projected_tokens += len(compact_output.encode("utf-8"))
        projected_tokens += FUNCTION_OUTPUT_PROJECTION_OVERHEAD_TOKENS
    return projected_tokens


def llm_call(
    client,
    model_name,
    instructions,
    model_input,
    tools=None,
    text_format=None,
    tool_choice=None,
    parallel_tool_calls=None,
    max_output_tokens=None,
):
    if text_format is not None and (
        tools is not None
        or tool_choice is not None
        or parallel_tool_calls is not None
    ):
        raise ValueError(
            "Tool configuration and text_format cannot be used together."
        )
    if tool_choice is not None and tools is None:
        raise ValueError("tool_choice requires tools.")
    if parallel_tool_calls is not None and tools is None:
        raise ValueError("parallel_tool_calls requires tools.")

    request = {
        "model": model_name,
        "instructions": instructions,
        "input": model_input,
        "store": False,
    }
    if max_output_tokens is not None:
        request["max_output_tokens"] = max_output_tokens

    if text_format is not None:
        return client.responses.parse(
            text_format=text_format,
            **request,
        )

    if tools is None:
        return client.responses.create(**request)

    request["tools"] = tools
    request["tool_choice"] = tool_choice or "auto"
    if parallel_tool_calls is not None:
        request["parallel_tool_calls"] = parallel_tool_calls
    return client.responses.create(**request)


def require_output_text(response, stage):
    if response.status != "completed":
        incomplete_details = getattr(response, "incomplete_details", None)
        reason = getattr(incomplete_details, "reason", None)
        detail = f": {reason}" if reason else ""
        raise RuntimeError(f"{stage} response did not complete{detail}.")

    refusal = get_refusal_text(response)
    if refusal is not None:
        raise RuntimeError(f"{stage} was refused: {refusal[:300]}")

    if not response.output_text.strip():
        raise RuntimeError(f"{stage} returned no text.")
    return response.output_text


def require_structured_output(response, expected_type, stage):
    if response.status != "completed":
        incomplete_details = getattr(response, "incomplete_details", None)
        reason = getattr(incomplete_details, "reason", None)
        detail = f": {reason}" if reason else ""
        raise RuntimeError(f"{stage} response did not complete{detail}.")

    refusal = get_refusal_text(response)
    if refusal is not None:
        raise RuntimeError(f"{stage} was refused: {refusal[:300]}")

    parsed_value = response.output_parsed
    if not isinstance(parsed_value, expected_type):
        raise TypeError(f"{stage} returned no validated structured output.")
    return parsed_value


def assess_clarification(client, model_name, question):
    response = llm_call(
        client,
        model_name,
        CLARIFICATION_INSTRUCTIONS,
        question,
        text_format=ClarificationAssessment,
    )
    return require_structured_output(
        response,
        ClarificationAssessment,
        "Clarification assessment",
    )


def print_clarification_questions(assessment):
    content = "\n".join(
        f"{number}. {question}"
        for number, question in enumerate(assessment.questions, start=1)
    )
    print_block("CLARIFICATION QUESTIONS", content)


def collect_clarification_answers(assessment):
    answers = []
    question_count = len(assessment.questions)
    for number in range(1, question_count + 1):
        answers.append(
            read_terminal_input(
                f"Answer {number}/{question_count}: ",
                max_length=MAX_USER_INPUT_CHARACTERS,
            )
        )
    return answers


def build_clarification_input(question, assessment, answers):
    lines = ["## Original question", "", question]
    if not assessment.questions:
        lines.extend(["", "## Clarification", "", "No clarification was required."])
        return "\n".join(lines)

    lines.extend(["", "## Clarification questions and answers", ""])
    for number, (clarification_question, answer) in enumerate(
        zip(assessment.questions, answers, strict=True),
        start=1,
    ):
        lines.append(f"{number}. Question: {clarification_question}")
        lines.append(f"   Answer: {answer}")
    return "\n".join(lines)


def markdown_list(items):
    visible_items = items or ["None specified."]
    return "\n".join(f"- {item}" for item in visible_items)


def format_research_brief(brief):
    return f"""
## Objective

{brief.objective}

## Audience

{brief.audience}

## Scope

{markdown_list(brief.scope)}

## Exclusions

{markdown_list(brief.exclusions)}

## Time horizon

{brief.time_horizon}

## Source preferences

{markdown_list(brief.source_preferences)}

## Output requirements

{markdown_list(brief.output_requirements)}

## Success criteria

{markdown_list(brief.success_criteria)}
""".strip()


def request_research_brief(client, model_name, question, assessment, answers):
    response = llm_call(
        client,
        model_name,
        BRIEF_INSTRUCTIONS,
        build_clarification_input(question, assessment, answers),
        text_format=ResearchBrief,
    )
    return require_structured_output(response, ResearchBrief, "Research brief")


def request_revised_research_brief(
    client,
    model_name,
    question,
    assessment,
    answers,
    brief,
    revision_request,
):
    model_input = f"""
{build_clarification_input(question, assessment, answers)}

## Current research brief

{format_research_brief(brief)}

## Revision request

{revision_request}
""".strip()
    response = llm_call(
        client,
        model_name,
        BRIEF_REVISION_INSTRUCTIONS,
        model_input,
        text_format=ResearchBrief,
    )
    return require_structured_output(
        response,
        ResearchBrief,
        "Research brief revision",
    )


def run_scope_workflow(client, model_name, question):
    print_progress(
        "Scope",
        "Clarification assessment",
        "started",
        indent=1,
    )
    assessment = assess_clarification(client, model_name, question)
    question_count = len(assessment.questions)
    print_progress(
        "Scope",
        "Clarification assessment",
        "completed",
        f"{question_count} question{'s' if question_count != 1 else ''}",
        indent=1,
    )

    if assessment.questions:
        print_clarification_questions(assessment)
        answers = collect_clarification_answers(assessment)
    else:
        print_progress(
            "Scope",
            "Clarification",
            "not required",
            indent=1,
        )
        answers = []

    print_progress("Scope", "Research brief", "started", indent=1)
    brief = request_research_brief(
        client,
        model_name,
        question,
        assessment,
        answers,
    )
    print_progress("Scope", "Research brief", "completed", indent=1)
    print_block(
        "RESEARCH BRIEF | PENDING APPROVAL",
        format_research_brief(brief),
    )

    action = read_terminal_input(
        "Action [approve/revise/cancel]: ",
        choices=("approve", "revise", "cancel"),
    )
    if action == "cancel":
        raise RunCancelled("Research brief approval was cancelled.")
    if action == "approve":
        return brief

    revision_request = read_terminal_input(
        "Revision request: ",
        max_length=MAX_USER_INPUT_CHARACTERS,
    )
    print_progress("Scope", "Research brief revision", "started", indent=1)
    revised_brief = request_revised_research_brief(
        client,
        model_name,
        question,
        assessment,
        answers,
        brief,
        revision_request,
    )
    print_progress("Scope", "Research brief revision", "completed", indent=1)
    print_block(
        "RESEARCH BRIEF | PENDING FINAL APPROVAL",
        format_research_brief(revised_brief),
    )

    final_action = read_terminal_input(
        "Action [approve/cancel]: ",
        choices=("approve", "cancel"),
    )
    if final_action == "cancel":
        raise RunCancelled("Final research brief approval was cancelled.")
    return revised_brief


def format_research_task(task):
    targets = "\n".join(f"- {target}" for target in task.evidence_targets)
    return f"""
Title: {task.title}

Research question: {task.research_question}

Evidence targets:

{targets}
""".strip()


def build_worker_input(request):
    return f"""
## Approved research brief

{request.approved_brief}

## Worker {request.worker_number} task

{format_research_task(request.task)}
""".strip()


def validate_research_state_summary(summary, state):
    if not isinstance(summary, ResearchStateSummary):
        raise TypeError("Context summary returned no validated structured output.")

    source_ids = [source.source_id for source in summary.visited_sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Context summary contains duplicate source IDs.")
    for source in summary.visited_sources:
        if source.source_id not in state.source_urls:
            raise ValueError("Context summary contains an unknown source ID.")
        if (
            source.evidence_level == "selected_source"
            and source.source_id not in state.read_source_ids
        ):
            raise ValueError(
                "Context summary overstates selected-source evidence."
            )
    return summary


def render_research_state_summary(summary, state):
    validate_research_state_summary(summary, state)
    sections = [
        "## Completed work",
        "",
        *(f"- {item}" for item in summary.completed_work),
        "",
        "## Visited sources",
        "",
    ]
    if not summary.visited_sources:
        sections.append("None.")
    for source in summary.visited_sources:
        sections.extend(
            [
                f"### {source.source_id} — {source.evidence_level}",
                "",
                f"URL: {state.source_urls[source.source_id]}",
                "",
                f"Key evidence: {source.key_evidence}",
                "",
                f"Limitations: {source.limitations}",
                "",
            ]
        )
    sections.extend(["## Unresolved questions", ""])
    if summary.unresolved_questions:
        sections.extend(f"- {item}" for item in summary.unresolved_questions)
    else:
        sections.append("None.")
    sections.extend(
        [
            "",
            "## Next actions",
            "",
            *(f"- {item}" for item in summary.next_actions),
        ]
    )
    return "\n".join(sections).strip()


def build_resumed_worker_input(state, summary):
    rendered_summary = render_research_state_summary(summary, state)
    return f"""
{build_worker_input(state.request)}

## Context session

Session: {state.context_session_number}/{MAX_CONTEXT_SESSIONS}

## Remaining run budgets

Model turns: {MAX_WORKER_TURNS - state.model_turns_used}
Tool calls: {MAX_WORKER_TOOL_CALLS - state.tool_calls_used}
Selected-source reads: {MAX_WORKER_SOURCE_READS - state.source_reads_used}
Context sessions: {MAX_CONTEXT_SESSIONS - state.context_session_number}
Context summaries: {MAX_CONTEXT_SUMMARIES - state.context_summaries_used}

## Application-validated research state

{RESUMED_RESEARCH_STATE_NOTICE}

{rendered_summary}
""".strip()


def initialize_agent_run_state(request):
    if (
        isinstance(request.worker_number, bool)
        or not isinstance(request.worker_number, int)
        or request.worker_number < 1
    ):
        raise ValueError("Agent run worker_number must be a positive integer.")
    if not isinstance(request.approved_brief, str) or not request.approved_brief.strip():
        raise ValueError("Agent run approved_brief must be a non-empty string.")
    if not isinstance(request.task, ResearchTask):
        raise TypeError("Agent run task must be a validated ResearchTask.")
    if not isinstance(request.llm_provider, LLMProvider):
        raise TypeError("Agent run llm_provider must be a valid LLMProvider.")
    if not isinstance(request.model_name, str) or not request.model_name.strip():
        raise ValueError("Agent run model_name must be a non-empty string.")
    if request.model_name != request.model_name.strip():
        raise ValueError("Agent run model_name must not contain surrounding whitespace.")

    return AgentRunState(
        request=request,
        input_items=[
            {
                "role": "user",
                "content": build_worker_input(request),
            }
        ],
    )


def finalize_agent_run(
    state,
    status,
    termination_reason,
    *,
    notes=None,
    error_message=None,
):
    if state.status != AgentRunStatus.RUNNING:
        raise ValueError("Agent run state has already been finalized.")
    if (
        state.termination_reason is not None
        or state.notes is not None
        or state.error_message is not None
    ):
        raise ValueError("Running Agent state contains terminal values.")
    if not 0 <= state.model_turns_used <= MAX_WORKER_TURNS:
        raise ValueError("Agent run model-turn usage is outside its limit.")
    if not 0 <= state.tool_calls_used <= MAX_WORKER_TOOL_CALLS:
        raise ValueError("Agent run tool-call usage is outside its limit.")
    if not 0 <= state.source_reads_used <= MAX_WORKER_SOURCE_READS:
        raise ValueError("Agent run source-read usage is outside its limit.")
    if state.source_reads_used > state.tool_calls_used:
        raise ValueError("Agent run source-read usage exceeds tool-call usage.")
    if not 1 <= state.context_session_number <= MAX_CONTEXT_SESSIONS:
        raise ValueError("Agent run context-session usage is outside its limit.")
    if not 0 <= state.context_summaries_used <= MAX_CONTEXT_SUMMARIES:
        raise ValueError("Agent run context-summary usage is outside its limit.")
    if state.context_session_number != state.context_summaries_used + 1:
        raise ValueError("Agent run context session and summary counts disagree.")
    if not isinstance(state.read_source_ids, set):
        raise TypeError("Agent run read-source IDs must be a set.")
    if not state.read_source_ids.issubset(state.source_urls):
        raise ValueError("Agent run read-source IDs are outside its source registry.")
    if len(state.read_source_ids) > state.source_reads_used:
        raise ValueError("Agent run read-source IDs exceed successful read attempts.")
    for name, value in {
        "peak input tokens": state.peak_input_tokens,
        "peak projected input tokens": state.peak_projected_input_tokens,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Agent run {name} must be a non-negative integer.")
    if not isinstance(status, AgentRunStatus) or status == AgentRunStatus.RUNNING:
        raise ValueError("Agent run result requires a terminal status.")
    if not isinstance(termination_reason, AgentTerminationReason):
        raise TypeError("Agent run result requires a termination reason.")

    allowed_reasons = {
        AgentRunStatus.COMPLETED: {
            AgentTerminationReason.COMPLETED,
            AgentTerminationReason.TOOL_LIMIT,
        },
        AgentRunStatus.FAILED: {
            AgentTerminationReason.TURN_LIMIT,
            AgentTerminationReason.CONTEXT_LIMIT,
            AgentTerminationReason.REFUSAL,
            AgentTerminationReason.MODEL_ERROR,
            AgentTerminationReason.TOOL_ERROR,
        },
        AgentRunStatus.CANCELLED: {
            AgentTerminationReason.CANCELLED,
        },
    }
    if termination_reason not in allowed_reasons[status]:
        raise ValueError("Agent run status and termination reason are inconsistent.")
    if (
        state.peak_projected_input_tokens > CONTEXT_HARD_LIMIT_TOKENS
        and not (
            status == AgentRunStatus.FAILED
            and termination_reason == AgentTerminationReason.CONTEXT_LIMIT
        )
    ):
        raise ValueError("Agent run exceeded its context hard limit without stopping.")

    if status == AgentRunStatus.COMPLETED:
        if not isinstance(notes, str) or not notes.strip():
            raise ValueError("A completed Agent run requires non-empty notes.")
        if error_message is not None:
            raise ValueError("A completed Agent run cannot contain an error.")
        final_notes = notes.strip()
        final_error = None
    else:
        if notes is not None:
            raise ValueError("An unsuccessful Agent run cannot contain notes.")
        if not isinstance(error_message, str) or not error_message.strip():
            raise ValueError("An unsuccessful Agent run requires an error message.")
        final_notes = None
        final_error = compact_agent_error(error_message)

    if status != AgentRunStatus.CANCELLED and state.model_turns_used == 0:
        raise ValueError("A non-cancelled Agent run must attempt a model turn.")

    state.status = status
    state.termination_reason = termination_reason
    state.notes = final_notes
    state.error_message = final_error
    return AgentRunResult(
        worker_number=state.request.worker_number,
        task=state.request.task,
        llm_provider=state.request.llm_provider,
        model_name=state.request.model_name,
        status=state.status,
        termination_reason=state.termination_reason,
        notes=state.notes,
        error_message=state.error_message,
        model_turns_used=state.model_turns_used,
        model_turn_limit=MAX_WORKER_TURNS,
        tool_calls_used=state.tool_calls_used,
        tool_call_limit=MAX_WORKER_TOOL_CALLS,
        source_reads_used=state.source_reads_used,
        source_read_limit=MAX_WORKER_SOURCE_READS,
        context_sessions_used=state.context_session_number,
        context_session_limit=MAX_CONTEXT_SESSIONS,
        context_summaries_used=state.context_summaries_used,
        context_summary_limit=MAX_CONTEXT_SUMMARIES,
        peak_input_tokens=state.peak_input_tokens,
        peak_projected_input_tokens=state.peak_projected_input_tokens,
        context_trigger_tokens=CONTEXT_TRIGGER_TOKENS,
        context_hard_limit_tokens=CONTEXT_HARD_LIMIT_TOKENS,
    )


def print_agent_run_result(result):
    lines = [
        f"LLM provider: {result.llm_provider.value}",
        f"Model: {result.model_name}",
        f"Status: {result.status.value}",
        f"Termination reason: {result.termination_reason.value}",
        f"Model turns: {result.model_turns_used}/{result.model_turn_limit}",
        f"Tool calls: {result.tool_calls_used}/{result.tool_call_limit}",
        f"Source reads: {result.source_reads_used}/{result.source_read_limit}",
        (
            "Context sessions: "
            f"{result.context_sessions_used}/{result.context_session_limit}"
        ),
        (
            "Context summaries: "
            f"{result.context_summaries_used}/{result.context_summary_limit}"
        ),
        f"Peak observed input tokens: {result.peak_input_tokens}",
        (
            "Peak projected input tokens: "
            f"{result.peak_projected_input_tokens}/"
            f"{result.context_hard_limit_tokens}"
        ),
    ]
    if result.error_message is not None:
        lines.append(f"Error: {result.error_message}")
    print_block(
        (
            f"AGENT RUN RESULT | WORKER {result.worker_number} | "
            f"{result.task.title}"
        ),
        "\n".join(lines),
    )


def build_supervisor_input(research_state):
    worker_results = research_state["worker_results"]
    sections = [
        "## Approved research brief",
        "",
        research_state["approved_brief"],
        "",
        "## Worker budget",
        "",
        f"Completed: {len(worker_results)}/{MAX_RESEARCH_WORKERS}",
        f"Remaining: {MAX_RESEARCH_WORKERS - len(worker_results)}",
        "",
        "## Completed research",
    ]
    if not worker_results:
        sections.extend(["", "None yet."])
    for result in worker_results:
        sections.extend(
            [
                "",
                f"### Worker {result.worker_number}: {result.task.title}",
                "",
                format_research_task(result.task),
                "",
                "Notes:",
                "",
                result.notes,
            ]
        )
    return "\n".join(sections)


def request_supervisor_decision(client, model_name, research_state):
    response = llm_call(
        client,
        model_name,
        SUPERVISOR_INSTRUCTIONS,
        build_supervisor_input(research_state),
        text_format=SupervisorDecision,
        max_output_tokens=MAX_SUPERVISOR_OUTPUT_TOKENS,
    )
    return require_structured_output(
        response,
        SupervisorDecision,
        "Research Supervisor",
    )


def print_supervisor_decision(decision_number, decision):
    if not decision.next_tasks:
        content = "Decision: Finish"
    else:
        content = f"""Decision: Start one worker

{format_research_task(decision.next_tasks[0])}
""".strip()
    print_block(
        f"SUPERVISOR DECISION {decision_number}/{MAX_RESEARCH_WORKERS}",
        content,
    )


def execute_worker_tool_call(tool_request, state, tavily_api_key):
    tool_call_number = state.tool_calls_used
    print_progress(
        f"Tool {tool_call_number}/{MAX_WORKER_TOOL_CALLS}",
        tool_request.name,
        "started",
        indent=2,
    )
    read_number = None
    source_id = None
    source_url = None
    try:
        arguments = json.loads(tool_request.arguments)
        if not isinstance(arguments, dict):
            raise TypeError("Tool arguments must be a JSON object.")

        tool_arguments = {}
        if tool_request.name == READ_SOURCE_TOOL_NAME:
            source_id, source_url = prepare_source_read(state, arguments)
            read_number = state.source_reads_used
            print_progress(
                f"Read {read_number}/{MAX_WORKER_SOURCE_READS}",
                f"{source_id} ({get_source_hostname(source_url)})",
                "started",
                indent=3,
            )
            tool_arguments["source_url"] = source_url

        tool_result = execute_research_tool(
            tool_request.name,
            arguments,
            tavily_api_key,
            **tool_arguments,
        )
        if tool_request.name in SEARCH_TOOL_NAMES:
            new_source_count = register_search_sources(state, tool_result)
            print_progress(
                "Sources",
                "Worker registry",
                "updated",
                (
                    f"{new_source_count} new, "
                    f"{len(state.source_urls)} available"
                ),
                indent=3,
            )
        elif tool_request.name == READ_SOURCE_TOOL_NAME:
            tool_result = normalize_source_read_result(
                state,
                source_id,
                source_url,
                tool_result,
            )
            state.read_source_ids.add(source_id)
            content_characters = tool_result[0]["content_characters"]
            reads_remaining = MAX_WORKER_SOURCE_READS - state.source_reads_used
            read_noun = "read" if reads_remaining == 1 else "reads"
            print_progress(
                f"Read {read_number}/{MAX_WORKER_SOURCE_READS}",
                f"{source_id} ({get_source_hostname(source_url)})",
                "completed",
                (
                    f"{content_characters} characters, "
                    f"{reads_remaining} {read_noun} remaining"
                ),
                indent=3,
            )
    except Exception as error:
        if read_number is not None:
            reads_remaining = MAX_WORKER_SOURCE_READS - state.source_reads_used
            read_noun = "read" if reads_remaining == 1 else "reads"
            print_progress(
                f"Read {read_number}/{MAX_WORKER_SOURCE_READS}",
                f"{source_id} ({get_source_hostname(source_url)})",
                "failed",
                f"{reads_remaining} {read_noun} remaining",
                indent=3,
            )
            error_message = (
                "Selected-source read failed "
                f"({type(error).__name__})."
            )
        else:
            error_message = str(error)
        tool_result = [{"error": error_message[:300]}]

    tool_status = (
        "failed" if any("error" in item for item in tool_result) else "completed"
    )
    print_progress(
        f"Tool {tool_call_number}/{MAX_WORKER_TOOL_CALLS}",
        tool_request.name,
        tool_status,
        indent=2,
    )
    return {
        "type": "function_call_output",
        "call_id": tool_request.call_id,
        "output": json.dumps(tool_result, ensure_ascii=False),
    }


def finalize_context_summary_failure(
    state,
    summary_number,
    status,
    termination_reason,
    error_message,
):
    print_progress(
        f"Summary {summary_number}/{MAX_CONTEXT_SUMMARIES}",
        "Research state",
        "cancelled" if status == AgentRunStatus.CANCELLED else "failed",
        indent=2,
    )
    return finalize_agent_run(
        state,
        status,
        termination_reason,
        error_message=error_message,
    )


def run_context_boundary(client, state, projected_input_tokens):
    current_session = state.context_session_number
    print_progress(
        f"Context {current_session}/{MAX_CONTEXT_SESSIONS}",
        "Boundary",
        "required",
        (
            f"projected {projected_input_tokens:,} tokens; "
            f"trigger {CONTEXT_TRIGGER_TOKENS:,}"
        ),
        indent=2,
    )

    if projected_input_tokens > CONTEXT_HARD_LIMIT_TOKENS:
        return finalize_agent_run(
            state,
            AgentRunStatus.FAILED,
            AgentTerminationReason.CONTEXT_LIMIT,
            error_message=(
                "Projected next research input exceeds the pre-summary "
                f"session hard limit of {CONTEXT_HARD_LIMIT_TOKENS} tokens."
            ),
        )
    if current_session >= MAX_CONTEXT_SESSIONS:
        return finalize_agent_run(
            state,
            AgentRunStatus.FAILED,
            AgentTerminationReason.CONTEXT_LIMIT,
            error_message="Research Worker reached its context-session limit.",
        )
    if MAX_WORKER_TURNS - state.model_turns_used < 2:
        return finalize_agent_run(
            state,
            AgentRunStatus.FAILED,
            AgentTerminationReason.TURN_LIMIT,
            error_message=(
                "Research Worker lacks one summary turn and one resumed "
                "research turn."
            ),
        )

    summary_number = state.context_summaries_used + 1
    print_progress(
        f"Summary {summary_number}/{MAX_CONTEXT_SUMMARIES}",
        "Research state",
        "started",
        indent=2,
    )
    summary_input = [
        *state.input_items,
        {
            "role": "user",
            "content": RESEARCH_SUMMARY_REQUEST_INPUT,
        },
    ]
    state.model_turns_used += 1
    try:
        summary_response = llm_call(
            client,
            state.request.model_name,
            RESEARCH_SUMMARY_INSTRUCTIONS,
            summary_input,
            text_format=ResearchStateSummary,
            max_output_tokens=MAX_SUMMARY_OUTPUT_TOKENS,
        )
    except RunCancelled as error:
        return finalize_context_summary_failure(
            state,
            summary_number,
            AgentRunStatus.CANCELLED,
            AgentTerminationReason.CANCELLED,
            compact_agent_error(error),
        )
    except Exception as error:
        return finalize_context_summary_failure(
            state,
            summary_number,
            AgentRunStatus.FAILED,
            AgentTerminationReason.MODEL_ERROR,
            format_model_exception(error),
        )

    if getattr(summary_response, "status", None) == "cancelled":
        return finalize_context_summary_failure(
            state,
            summary_number,
            AgentRunStatus.CANCELLED,
            AgentTerminationReason.CANCELLED,
            "Context summary response was cancelled.",
        )
    refusal = get_refusal_text(summary_response)
    if refusal is not None:
        return finalize_context_summary_failure(
            state,
            summary_number,
            AgentRunStatus.FAILED,
            AgentTerminationReason.MODEL_ERROR,
            compact_agent_error(f"Context summary was refused: {refusal}"),
        )
    if getattr(summary_response, "status", None) != "completed":
        return finalize_context_summary_failure(
            state,
            summary_number,
            AgentRunStatus.FAILED,
            AgentTerminationReason.MODEL_ERROR,
            format_response_failure(summary_response),
        )

    try:
        summary_input_tokens, _ = validate_response_usage(summary_response)
        state.peak_input_tokens = max(
            state.peak_input_tokens,
            summary_input_tokens,
        )
        summary = validate_research_state_summary(
            getattr(summary_response, "output_parsed", None),
            state,
        )
        rendered_summary = render_research_state_summary(summary, state)
    except (TypeError, ValueError) as error:
        return finalize_context_summary_failure(
            state,
            summary_number,
            AgentRunStatus.FAILED,
            AgentTerminationReason.MODEL_ERROR,
            compact_agent_error(error),
        )

    print_block(
        (
            "RESEARCH STATE SUMMARY | "
            f"WORKER {state.request.worker_number} | "
            f"SESSION {current_session} -> {current_session + 1}"
        ),
        rendered_summary,
    )
    state.context_summaries_used += 1
    state.context_session_number += 1
    state.latest_summary = summary
    state.input_items = [
        {
            "role": "user",
            "content": build_resumed_worker_input(state, summary),
        }
    ]
    print_progress(
        f"Summary {summary_number}/{MAX_CONTEXT_SUMMARIES}",
        "Research state",
        "completed",
        "validated",
        indent=2,
    )
    print_progress(
        f"Context {state.context_session_number}/{MAX_CONTEXT_SESSIONS}",
        "Session",
        "started",
        "validated summary",
        indent=2,
    )
    return None


def run_research_worker_loop(
    client,
    request,
    tavily_api_key,
):
    state = initialize_agent_run_state(request)
    print_progress(
        f"Context {state.context_session_number}/{MAX_CONTEXT_SESSIONS}",
        "Session",
        "started",
        "fresh task history",
        indent=2,
    )

    while state.model_turns_used < MAX_WORKER_TURNS:
        tool_budget_exhausted = (
            state.tool_calls_used == MAX_WORKER_TOOL_CALLS
        )
        model_input = state.input_items
        if tool_budget_exhausted:
            model_input = [
                *state.input_items,
                {
                    "role": "user",
                    "content": RESEARCH_BUDGET_EXHAUSTED_INPUT,
                },
            ]

        state.model_turns_used += 1
        turn_number = state.model_turns_used
        tool_calls_remaining = MAX_WORKER_TOOL_CALLS - state.tool_calls_used
        source_reads_remaining = (
            MAX_WORKER_SOURCE_READS - state.source_reads_used
        )
        tool_noun = "tool" if tool_calls_remaining == 1 else "tools"
        source_read_noun = (
            "source read" if source_reads_remaining == 1 else "source reads"
        )
        print_progress(
            f"Turn {turn_number}/{MAX_WORKER_TURNS}",
            "Model",
            "started",
            (
                f"{tool_calls_remaining} {tool_noun}, "
                f"{source_reads_remaining} {source_read_noun} remaining"
            ),
            indent=2,
        )

        try:
            model_response = llm_call(
                client,
                request.model_name,
                RESEARCH_INSTRUCTIONS,
                model_input,
                RESEARCH_TOOLS,
                tool_choice="none" if tool_budget_exhausted else "auto",
                parallel_tool_calls=False,
            )
        except RunCancelled as error:
            return finalize_agent_run(
                state,
                AgentRunStatus.CANCELLED,
                AgentTerminationReason.CANCELLED,
                error_message=compact_agent_error(error),
            )
        except Exception as error:
            return finalize_agent_run(
                state,
                AgentRunStatus.FAILED,
                AgentTerminationReason.MODEL_ERROR,
                error_message=format_model_exception(error),
            )

        if getattr(model_response, "status", None) == "cancelled":
            return finalize_agent_run(
                state,
                AgentRunStatus.CANCELLED,
                AgentTerminationReason.CANCELLED,
                error_message="Research model response was cancelled.",
            )

        refusal = get_refusal_text(model_response)
        if refusal is not None:
            return finalize_agent_run(
                state,
                AgentRunStatus.FAILED,
                AgentTerminationReason.REFUSAL,
                error_message=compact_agent_error(
                    f"Research Worker was refused: {refusal}"
                ),
            )

        if getattr(model_response, "status", None) != "completed":
            return finalize_agent_run(
                state,
                AgentRunStatus.FAILED,
                AgentTerminationReason.MODEL_ERROR,
                error_message=format_response_failure(model_response),
            )

        output_items = getattr(model_response, "output", None)
        if not isinstance(output_items, list):
            return finalize_agent_run(
                state,
                AgentRunStatus.FAILED,
                AgentTerminationReason.MODEL_ERROR,
                error_message="Research model response returned invalid output items.",
            )

        try:
            input_tokens, output_tokens = validate_response_usage(model_response)
        except (TypeError, ValueError) as error:
            return finalize_agent_run(
                state,
                AgentRunStatus.FAILED,
                AgentTerminationReason.MODEL_ERROR,
                error_message=compact_agent_error(error),
            )
        state.peak_input_tokens = max(state.peak_input_tokens, input_tokens)
        state.input_items.extend(output_items)

        tool_requests = [
            item
            for item in output_items
            if getattr(item, "type", None) == "function_call"
        ]

        if tool_budget_exhausted and tool_requests:
            if state.model_turns_used == MAX_WORKER_TURNS:
                return finalize_agent_run(
                    state,
                    AgentRunStatus.FAILED,
                    AgentTerminationReason.TURN_LIMIT,
                    error_message=(
                        "Research Worker reached its model-turn limit without "
                        "final notes."
                    ),
                )
            return finalize_agent_run(
                state,
                AgentRunStatus.FAILED,
                AgentTerminationReason.MODEL_ERROR,
                error_message=(
                    "Research Worker requested tools after its tool budget "
                    "expired."
                ),
            )

        if not tool_requests:
            notes = getattr(model_response, "output_text", None)
            if not isinstance(notes, str) or not notes.strip():
                if state.model_turns_used == MAX_WORKER_TURNS:
                    return finalize_agent_run(
                        state,
                        AgentRunStatus.FAILED,
                        AgentTerminationReason.TURN_LIMIT,
                        error_message=(
                            "Research Worker reached its model-turn limit "
                            "without final notes."
                        ),
                    )
                return finalize_agent_run(
                    state,
                    AgentRunStatus.FAILED,
                    AgentTerminationReason.MODEL_ERROR,
                    error_message="Research model response returned no notes.",
                )
            print_progress(
                f"Turn {turn_number}/{MAX_WORKER_TURNS}",
                "Model",
                "completed",
                indent=2,
            )
            return finalize_agent_run(
                state,
                AgentRunStatus.COMPLETED,
                (
                    AgentTerminationReason.TOOL_LIMIT
                    if tool_budget_exhausted
                    else AgentTerminationReason.COMPLETED
                ),
                notes=notes,
            )

        tool_outputs = []
        for tool_request in tool_requests:
            if state.tool_calls_used == MAX_WORKER_TOOL_CALLS:
                print_progress(
                    "Tool limit",
                    tool_request.name,
                    "skipped",
                    indent=2,
                )
                tool_output = {
                    "type": "function_call_output",
                    "call_id": tool_request.call_id,
                    "output": json.dumps(
                        [{"error": "Tool call limit reached."}],
                        ensure_ascii=False,
                    ),
                }
            else:
                state.tool_calls_used += 1
                try:
                    tool_output = execute_worker_tool_call(
                        tool_request,
                        state,
                        tavily_api_key,
                    )
                except RunCancelled as error:
                    return finalize_agent_run(
                        state,
                        AgentRunStatus.CANCELLED,
                        AgentTerminationReason.CANCELLED,
                        error_message=compact_agent_error(error),
                    )
                except Exception as error:
                    return finalize_agent_run(
                        state,
                        AgentRunStatus.FAILED,
                        AgentTerminationReason.TOOL_ERROR,
                        error_message=compact_agent_error(
                            "Research tool boundary failed "
                            f"({type(error).__name__})."
                        ),
                    )
            state.input_items.append(tool_output)
            tool_outputs.append(tool_output)

        projected_input_tokens = project_next_research_input(
            input_tokens,
            output_tokens,
            tool_outputs,
        )
        state.peak_projected_input_tokens = max(
            state.peak_projected_input_tokens,
            projected_input_tokens,
        )

        if state.tool_calls_used == MAX_WORKER_TOOL_CALLS:
            print_progress(
                "Research",
                "Tool budget",
                "exhausted",
                "next turn will produce final notes",
                indent=2,
            )

        if projected_input_tokens >= CONTEXT_TRIGGER_TOKENS:
            boundary_result = run_context_boundary(
                client,
                state,
                projected_input_tokens,
            )
            if boundary_result is not None:
                return boundary_result

    return finalize_agent_run(
        state,
        AgentRunStatus.FAILED,
        AgentTerminationReason.TURN_LIMIT,
        error_message=(
            f"Research Worker did not finish within {MAX_WORKER_TURNS} "
            "model turns."
        ),
    )


def run_research_worker(
    client,
    llm_provider,
    model_name,
    research_brief,
    worker_number,
    task,
    tavily_api_key,
):
    request = AgentRunRequest(
        worker_number=worker_number,
        approved_brief=research_brief,
        task=task,
        llm_provider=llm_provider,
        model_name=model_name,
    )
    print_progress(
        f"Worker {worker_number}/{MAX_RESEARCH_WORKERS}",
        task.title,
        "started",
        indent=1,
    )
    result = run_research_worker_loop(
        client,
        request,
        tavily_api_key,
    )

    print_progress(
        f"Worker {worker_number}/{MAX_RESEARCH_WORKERS}",
        task.title,
        result.status.value,
        result.termination_reason.value,
        indent=1,
    )
    print_agent_run_result(result)
    if result.status == AgentRunStatus.COMPLETED:
        if result.notes is None:
            raise ValueError("Completed Agent run result contains no notes.")
        print_block(
            f"RESEARCH NOTES | WORKER {worker_number} | {task.title}",
            result.notes,
        )
    return result


def run_research_supervisor_loop(
    client,
    llm_provider,
    model_name,
    research_brief,
    tavily_api_key,
):
    research_state = {
        "approved_brief": research_brief,
        "worker_results": [],
        "stop_reason": None,
    }

    while len(research_state["worker_results"]) < MAX_RESEARCH_WORKERS:
        worker_number = len(research_state["worker_results"]) + 1
        print_progress(
            f"Supervisor {worker_number}/{MAX_RESEARCH_WORKERS}",
            "Decision",
            "started",
            indent=1,
        )
        decision = request_supervisor_decision(client, model_name, research_state)
        if not decision.next_tasks and not research_state["worker_results"]:
            raise RuntimeError(
                "Research Supervisor finished before any worker completed."
            )
        print_progress(
            f"Supervisor {worker_number}/{MAX_RESEARCH_WORKERS}",
            "Decision",
            "completed",
            "finish" if not decision.next_tasks else "next task selected",
            indent=1,
        )
        print_supervisor_decision(worker_number, decision)

        if not decision.next_tasks:
            research_state["stop_reason"] = "Research Supervisor selected no next task."
            break

        task = decision.next_tasks[0]
        worker_result = run_research_worker(
            client,
            llm_provider,
            model_name,
            research_brief,
            worker_number,
            task,
            tavily_api_key,
        )
        if worker_result.status != AgentRunStatus.COMPLETED:
            raise RuntimeError(
                f"Research Worker {worker_number} ({task.title}) "
                f"{worker_result.status.value}: {worker_result.error_message}"
            )
        research_state["worker_results"].append(worker_result)

    if research_state["stop_reason"] is None:
        research_state["stop_reason"] = "Research worker limit reached."

    print_progress(
        "Supervisor",
        "Research",
        "completed",
        research_state["stop_reason"],
        indent=1,
    )
    return research_state


def format_combined_research_notes(research_state):
    sections = []
    for result in research_state["worker_results"]:
        sections.append(
            f"""## Worker {result.worker_number}: {result.task.title}

{format_research_task(result.task)}

### Notes

{result.notes}"""
        )
    return "\n\n".join(sections)


def run_report_workflow(client, model_name, brief_text, combined_notes):
    print_progress("3/5", "Write", "started")
    write_input = f"""

## Approved research brief

{brief_text}

## Research notes

{combined_notes}
""".strip()
    draft = require_output_text(
        llm_call(client, model_name, WRITE_INSTRUCTIONS, write_input), "Write"
    )
    print_progress("3/5", "Write", "completed")
    print_block("DRAFT", draft)

    print_progress("4/5", "Critic", "started")
    critic_input = f"""{write_input}

## Draft

{draft}
"""
    critique = require_output_text(
        llm_call(client, model_name, CRITIC_INSTRUCTIONS, critic_input), "Critic"
    )
    print_progress("4/5", "Critic", "completed")
    print_block("CRITIQUE", critique)

    print_progress("5/5", "Revise", "started")
    revise_input = f"""{critic_input}

## Critique

{critique}
"""

    final_report = require_output_text(
        llm_call(client, model_name, REVISE_INSTRUCTIONS, revise_input), "Revise"
    )
    print_progress("5/5", "Revise", "completed")
    return final_report


def run_deep_research(
    client,
    llm_provider,
    model_name,
    question,
    tavily_api_key,
):
    print_progress(
        "LLM",
        "Provider and model",
        "configured",
        f"{llm_provider.value}, {model_name}",
    )
    print_progress("1/5", "Scope", "started")
    approved_brief = run_scope_workflow(client, model_name, question)
    brief_text = format_research_brief(approved_brief)
    print_progress("1/5", "Scope", "completed", "research brief approved")

    print_progress("2/5", "Research", "started")
    research_state = run_research_supervisor_loop(
        client,
        llm_provider,
        model_name,
        brief_text,
        tavily_api_key,
    )
    completed_worker_count = len(research_state["worker_results"])
    combined_notes = format_combined_research_notes(research_state)
    print_progress(
        "2/5",
        "Research",
        "completed",
        format_worker_count(completed_worker_count),
    )
    print_block("COMBINED RESEARCH NOTES", combined_notes)

    return run_report_workflow(
        client,
        model_name,
        brief_text,
        combined_notes,
    )


def parse_cli_question():
    parser = argparse.ArgumentParser(
        description="Scope and run supervised research, then refine its report.",
    )
    parser.add_argument("question", help="Research question to investigate")
    return parser.parse_args().question.strip()


def main():
    question = parse_cli_question()
    if not question:
        print_error("Question must not be empty.")
        return 2

    load_dotenv()
    llm_provider_value = os.getenv("LLM_PROVIDER", "").strip()
    tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()

    missing = [
        name
        for name, value in {
            "LLM_PROVIDER": llm_provider_value,
            "TAVILY_API_KEY": tavily_api_key,
        }.items()
        if not value
    ]
    if missing:
        print_error(
            f"Missing environment variable(s): {', '.join(missing)}."
        )
        return 2

    try:
        llm_provider = LLMProvider(llm_provider_value)
    except ValueError:
        print_error("LLM_PROVIDER must be one of: openai, deepseek.")
        return 2

    if llm_provider == LLMProvider.OPENAI:
        api_key_name = "OPENAI_API_KEY"
        model_name_name = "OPENAI_MODEL_NAME"
        base_url = None
    else:
        api_key_name = "DEEPSEEK_API_KEY"
        model_name_name = "DEEPSEEK_MODEL_NAME"
        base_url = DEEPSEEK_BASE_URL

    api_key = os.getenv(api_key_name, "").strip()
    model_name = os.getenv(model_name_name, "").strip()
    missing_provider = [
        name
        for name, value in {
            api_key_name: api_key,
            model_name_name: model_name,
        }.items()
        if not value
    ]
    if missing_provider:
        print_error(
            f"Missing environment variable(s): {', '.join(missing_provider)}."
        )
        return 2

    try:
        if base_url is None:
            client = OpenAI(api_key=api_key, max_retries=0)
        else:
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                max_retries=0,
            )
    except Exception as error:
        print_error(
            "Run failed: model client construction failed "
            f"({type(error).__name__})."
        )
        return 1

    try:
        final_report = run_deep_research(
            client,
            llm_provider,
            model_name,
            question,
            tavily_api_key,
        )
        print_block("FINAL REPORT", final_report)
        print_progress("Run", "Deep research", "completed")
        return 0
    except RunCancelled as error:
        print_progress("Run", "Deep research", "cancelled", str(error))
        return 1
    except Exception as error:
        print_error(f"Run failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

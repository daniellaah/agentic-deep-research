"""Scope and run supervised research before refining a final report."""

import argparse
import json
import os
import sys
from typing import Annotated

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
    REVISE_INSTRUCTIONS,
    SUPERVISOR_INSTRUCTIONS,
    WRITE_INSTRUCTIONS,
)
from agent_tools import RESEARCH_TOOLS, execute_research_tool

MAX_WORKER_TURNS = 6
MAX_WORKER_TOOL_CALLS = 5
MAX_RESEARCH_WORKERS = 4
MAX_SUPERVISOR_OUTPUT_TOKENS = 4000
MAX_LOCAL_INPUT_CHARACTERS = 2000
OUTPUT_WIDTH = 80


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


class RunCancelled(Exception):
    """Stop a run after local input is cancelled or ends."""


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


def read_local_input(prompt, *, choices=None, max_length=None):
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


def llm_call(
    client,
    model_name,
    instructions,
    model_input,
    tools=None,
    text_format=None,
    tool_choice=None,
    max_output_tokens=None,
):
    if text_format is not None and (tools is not None or tool_choice is not None):
        raise ValueError(
            "tools, tool_choice, and text_format cannot be used together."
        )
    if tool_choice is not None and tools is None:
        raise ValueError("tool_choice requires tools.")

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

    return client.responses.create(
        tools=tools,
        tool_choice=tool_choice or "auto",
        **request,
    )


def require_output_text(response, stage):
    if response.status != "completed":
        incomplete_details = getattr(response, "incomplete_details", None)
        reason = getattr(incomplete_details, "reason", None)
        detail = f": {reason}" if reason else ""
        raise RuntimeError(f"{stage} response did not complete{detail}.")

    for output in response.output:
        if output.type != "message":
            continue
        for content in output.content:
            if content.type == "refusal":
                raise RuntimeError(f"{stage} was refused: {content.refusal[:300]}")

    if not response.output_text.strip():
        raise RuntimeError(f"{stage} returned no text.")
    return response.output_text


def require_structured_output(response, expected_type, stage):
    if response.status != "completed":
        incomplete_details = getattr(response, "incomplete_details", None)
        reason = getattr(incomplete_details, "reason", None)
        detail = f": {reason}" if reason else ""
        raise RuntimeError(f"{stage} response did not complete{detail}.")

    for output in response.output:
        if output.type != "message":
            continue
        for content in output.content:
            if content.type == "refusal":
                raise RuntimeError(
                    f"{stage} was refused: {content.refusal[:300]}"
                )

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
            read_local_input(
                f"Answer {number}/{question_count}: ",
                max_length=MAX_LOCAL_INPUT_CHARACTERS,
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

    action = read_local_input(
        "Action [approve/revise/cancel]: ",
        choices=("approve", "revise", "cancel"),
    )
    if action == "cancel":
        raise RunCancelled("Research brief approval was cancelled.")
    if action == "approve":
        return brief

    revision_request = read_local_input(
        "Revision request: ",
        max_length=MAX_LOCAL_INPUT_CHARACTERS,
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

    final_action = read_local_input(
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
                f"### Worker {result['worker_number']}: {result['task'].title}",
                "",
                format_research_task(result["task"]),
                "",
                "Notes:",
                "",
                result["notes"],
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


def build_worker_input(research_brief, worker_number, task):
    return f"""
## Approved research brief

{research_brief}

## Worker {worker_number} task

{format_research_task(task)}
""".strip()


def execute_worker_tool_call(tool_request, tool_call_number, tavily_api_key):
    print_progress(
        f"Tool {tool_call_number}/{MAX_WORKER_TOOL_CALLS}",
        tool_request.name,
        "started",
        indent=2,
    )
    try:
        arguments = json.loads(tool_request.arguments)
        if not isinstance(arguments, dict):
            raise TypeError("Tool arguments must be a JSON object.")
        tool_result = execute_research_tool(
            tool_request.name,
            arguments,
            tavily_api_key,
        )
    except Exception as error:
        tool_result = [{"error": str(error)[:300]}]

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


def run_research_worker_loop(
    client,
    model_name,
    initial_input,
    tavily_api_key,
):
    worker_history = [{"role": "user", "content": initial_input}]
    tool_call_count = 0

    for iteration in range(1, MAX_WORKER_TURNS + 1):
        tool_budget_exhausted = tool_call_count == MAX_WORKER_TOOL_CALLS
        model_input = worker_history
        if tool_budget_exhausted:
            model_input = [
                *worker_history,
                {
                    "role": "user",
                    "content": RESEARCH_BUDGET_EXHAUSTED_INPUT,
                },
            ]

        print_progress(
            f"Turn {iteration}/{MAX_WORKER_TURNS}",
            "Model",
            "started",
            f"{MAX_WORKER_TOOL_CALLS - tool_call_count} tools remaining",
            indent=2,
        )

        model_response = llm_call(
            client,
            model_name,
            RESEARCH_INSTRUCTIONS,
            model_input,
            RESEARCH_TOOLS,
            tool_choice="none" if tool_budget_exhausted else "auto",
        )
        worker_history.extend(model_response.output)

        tool_requests = [
            item for item in model_response.output if item.type == "function_call"
        ]

        if tool_budget_exhausted and tool_requests:
            raise RuntimeError(
                "Research Worker requested tools after its tool budget expired."
            )

        if not tool_requests:
            print_progress(
                f"Turn {iteration}/{MAX_WORKER_TURNS}",
                "Model",
                "completed",
                indent=2,
            )
            return require_output_text(model_response, "Research")

        for tool_request in tool_requests:
            if tool_call_count == MAX_WORKER_TOOL_CALLS:
                print_progress(
                    "Tool limit",
                    tool_request.name,
                    "skipped",
                    indent=2,
                )
                worker_history.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_request.call_id,
                        "output": json.dumps(
                            [{"error": "Tool call limit reached."}],
                            ensure_ascii=False,
                        ),
                    }
                )
                continue

            tool_call_count += 1
            worker_history.append(
                execute_worker_tool_call(
                    tool_request,
                    tool_call_count,
                    tavily_api_key,
                )
            )

        if tool_call_count == MAX_WORKER_TOOL_CALLS:
            print_progress(
                "Research",
                "Tool budget",
                "exhausted",
                "next turn will produce final notes",
                indent=2,
            )

    raise RuntimeError(
        f"Research Worker did not finish within {MAX_WORKER_TURNS} model turns."
    )


def run_research_worker(
    client,
    model_name,
    research_brief,
    worker_number,
    task,
    tavily_api_key,
):
    print_progress(
        f"Worker {worker_number}/{MAX_RESEARCH_WORKERS}",
        task.title,
        "started",
        indent=1,
    )
    try:
        notes = run_research_worker_loop(
            client,
            model_name,
            build_worker_input(research_brief, worker_number, task),
            tavily_api_key,
        )
    except Exception as error:
        raise RuntimeError(
            f"Research Worker {worker_number} ({task.title}) failed: {error}"
        ) from error

    print_progress(
        f"Worker {worker_number}/{MAX_RESEARCH_WORKERS}",
        task.title,
        "completed",
        indent=1,
    )
    print_block(
        f"RESEARCH NOTES | WORKER {worker_number} | {task.title}",
        notes,
    )
    return {
        "worker_number": worker_number,
        "task": task,
        "notes": notes,
    }


def run_research_supervisor_loop(client, model_name, research_brief, tavily_api_key):
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
            model_name,
            research_brief,
            worker_number,
            task,
            tavily_api_key,
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
            f"""## Worker {result['worker_number']}: {result['task'].title}

{format_research_task(result['task'])}

### Notes

{result['notes']}"""
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


def run_deep_research(client, model_name, question, tavily_api_key):
    print_progress("1/5", "Scope", "started")
    approved_brief = run_scope_workflow(client, model_name, question)
    brief_text = format_research_brief(approved_brief)
    print_progress("1/5", "Scope", "completed", "research brief approved")

    print_progress("2/5", "Research", "started")
    research_state = run_research_supervisor_loop(
        client,
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
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model_name = os.getenv("MODEL_NAME", "").strip()
    tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()

    missing = [
        name
        for name, value in {
            "OPENAI_API_KEY": api_key,
            "MODEL_NAME": model_name,
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
        client = OpenAI(api_key=api_key)
        final_report = run_deep_research(
            client,
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

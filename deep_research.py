"""Plan and run tool-using research before refining a final report."""

import argparse
import json
import os
import sys
from typing import Annotated

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from agent_instructions import (
    CRITIC_INSTRUCTIONS,
    PLANNING_INSTRUCTIONS,
    RESEARCH_BUDGET_EXHAUSTED_INPUT,
    RESEARCH_INSTRUCTIONS,
    REVISE_INSTRUCTIONS,
    WRITE_INSTRUCTIONS,
)
from agent_tools import RESEARCH_TOOLS, execute_tool

MAX_AGENT_ITERATIONS = 10
MAX_TOOL_CALLS = 8
OUTPUT_WIDTH = 80


CompletionCriterion = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=240),
]


class ResearchTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=100),
    ]
    research_question: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=600),
    ]
    completion_criteria: Annotated[
        list[CompletionCriterion],
        Field(min_length=1, max_length=3),
    ]


class ResearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tasks: Annotated[
        list[ResearchTask],
        Field(min_length=1, max_length=4),
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


def task_count_text(task_count):
    noun = "task" if task_count == 1 else "tasks"
    return f"{task_count} {noun}"


def llm_call(
    client,
    model_name,
    instructions,
    model_input,
    tools=None,
    text_format=None,
    tool_choice=None,
):
    if text_format is not None and (tools is not None or tool_choice is not None):
        raise ValueError(
            "tools, tool_choice, and text_format cannot be used together."
        )
    if tool_choice is not None and tools is None:
        raise ValueError("tool_choice requires tools.")

    if text_format is not None:
        return client.responses.parse(
            model=model_name,
            instructions=instructions,
            input=model_input,
            text_format=text_format,
            store=False,
        )

    if tools is None:
        return client.responses.create(
            model=model_name,
            instructions=instructions,
            input=model_input,
            store=False,
        )

    return client.responses.create(
        model=model_name,
        instructions=instructions,
        input=model_input,
        tools=tools,
        tool_choice=tool_choice or "auto",
        store=False,
    )


def output_text(response, stage):
    if not response.output_text.strip():
        raise RuntimeError(f"{stage} returned no text.")
    return response.output_text


def parsed_research_plan(response):
    if response.status != "completed":
        incomplete_details = getattr(response, "incomplete_details", None)
        reason = getattr(incomplete_details, "reason", None)
        detail = f": {reason}" if reason else ""
        raise RuntimeError(f"Plan response did not complete{detail}.")

    for output in response.output:
        if output.type != "message":
            continue
        for content in output.content:
            if content.type == "refusal":
                raise RuntimeError(f"Plan was refused: {content.refusal[:300]}")

    plan = response.output_parsed
    if not isinstance(plan, ResearchPlan):
        raise TypeError("Plan returned no validated structured output.")
    return plan


def create_research_plan(client, model_name, question):
    response = llm_call(
        client,
        model_name,
        PLANNING_INSTRUCTIONS,
        question,
        text_format=ResearchPlan,
    )
    return parsed_research_plan(response)


def print_research_plan(plan):
    lines = []
    for task_number, task in enumerate(plan.tasks, start=1):
        if lines:
            lines.append("")
        lines.append(f"{task_number}. {task.title}")
        lines.append(f"   Research question: {task.research_question}")
        lines.append("   Completion criteria:")
        for criterion in task.completion_criteria:
            lines.append(f"     - {criterion}")
    print_block("RESEARCH PLAN", "\n".join(lines))


def task_input(question, task_number, task):
    criteria = "\n".join(f"- {criterion}" for criterion in task.completion_criteria)
    return f"""
## Original question

{question}

## Current task {task_number}: {task.title}

{task.research_question}

## Completion criteria

{criteria}
""".strip()


def agent_loop(
    client,
    model_name,
    research_input,
    tavily_api_key,
):
    history = [{"role": "user", "content": research_input}]
    tool_calls = 0

    for iteration in range(1, MAX_AGENT_ITERATIONS + 1):
        budget_exhausted = tool_calls == MAX_TOOL_CALLS
        request_input = history
        if budget_exhausted:
            request_input = [
                *history,
                {
                    "role": "user",
                    "content": RESEARCH_BUDGET_EXHAUSTED_INPUT,
                },
            ]

        print_progress(
            f"Turn {iteration}/{MAX_AGENT_ITERATIONS}",
            "Model",
            "started",
            f"{MAX_TOOL_CALLS - tool_calls} tools remaining",
            indent=2,
        )

        response = llm_call(
            client,
            model_name,
            RESEARCH_INSTRUCTIONS,
            request_input,
            RESEARCH_TOOLS,
            tool_choice="none" if budget_exhausted else "auto",
        )
        history.extend(response.output)

        function_calls = [
            item for item in response.output if item.type == "function_call"
        ]

        if budget_exhausted and function_calls:
            raise RuntimeError(
                "Research Agent requested tools after its tool budget expired."
            )

        if not function_calls:
            print_progress(
                f"Turn {iteration}/{MAX_AGENT_ITERATIONS}",
                "Model",
                "completed",
                indent=2,
            )
            return output_text(response, "Research")

        for function_call in function_calls:
            if tool_calls == MAX_TOOL_CALLS:
                result = [{"error": "Tool call limit reached."}]
                print_progress(
                    "Tool limit",
                    function_call.name,
                    "skipped",
                    indent=2,
                )
            else:
                tool_calls += 1
                print_progress(
                    f"Tool {tool_calls}/{MAX_TOOL_CALLS}",
                    function_call.name,
                    "started",
                    indent=2,
                )
                try:
                    arguments = json.loads(function_call.arguments)
                    if not isinstance(arguments, dict):
                        raise TypeError("Tool arguments must be a JSON object.")
                    result = execute_tool(
                        function_call.name,
                        arguments,
                        tavily_api_key,
                    )
                except Exception as error:
                    result = [{"error": str(error)[:300]}]

                status = (
                    "failed" if any("error" in item for item in result) else "completed"
                )
                print_progress(
                    f"Tool {tool_calls}/{MAX_TOOL_CALLS}",
                    function_call.name,
                    status,
                    indent=2,
                )

            history.append(
                {
                    "type": "function_call_output",
                    "call_id": function_call.call_id,
                    "output": json.dumps(result, ensure_ascii=False),
                }
            )

        if tool_calls == MAX_TOOL_CALLS:
            print_progress(
                "Research",
                "Tool budget",
                "exhausted",
                "next turn will produce final notes",
                indent=2,
            )

    raise RuntimeError(
        f"Research Agent did not finish within {MAX_AGENT_ITERATIONS} model iterations."
    )


def execute_research_plan(client, model_name, question, plan, tavily_api_key):
    task_notes = []
    task_count = len(plan.tasks)
    for task_number, task in enumerate(plan.tasks, start=1):
        print_progress(
            f"Task {task_number}/{task_count}",
            task.title,
            "started",
            indent=1,
        )
        try:
            notes = agent_loop(
                client,
                model_name,
                task_input(question, task_number, task),
                tavily_api_key,
            )
        except Exception as error:
            raise RuntimeError(
                f"Research task {task_number} ({task.title}) failed: {error}"
            ) from error
        task_notes.append(notes)
        print_progress(
            f"Task {task_number}/{task_count}",
            task.title,
            "completed",
            indent=1,
        )
        print_block(
            f"RESEARCH NOTES | TASK {task_number}/{task_count} | {task.title}",
            notes,
        )

    return "\n\n".join(task_notes)


def research_workflow(client, model_name, question, tavily_api_key):
    print_progress("1/5", "Plan", "started")
    plan = create_research_plan(client, model_name, question)
    task_count = len(plan.tasks)
    print_progress("1/5", "Plan", "completed", task_count_text(task_count))
    print_research_plan(plan)

    print_progress("2/5", "Research", "started", task_count_text(task_count))
    research_notes = execute_research_plan(
        client,
        model_name,
        question,
        plan,
        tavily_api_key,
    )
    print_progress("2/5", "Research", "completed", task_count_text(task_count))
    print_block("COMBINED RESEARCH NOTES", research_notes)

    print_progress("3/5", "Write", "started")
    context = f"""

## Original question

{question}

## Research notes

{research_notes}
""".strip()
    draft = output_text(
        llm_call(client, model_name, WRITE_INSTRUCTIONS, context), "Write"
    )
    print_progress("3/5", "Write", "completed")
    print_block("DRAFT", draft)

    print_progress("4/5", "Critic", "started")
    context += f"""

## Draft

{draft}
"""
    critique = output_text(
        llm_call(client, model_name, CRITIC_INSTRUCTIONS, context), "Critic"
    )
    print_progress("4/5", "Critic", "completed")
    print_block("CRITIQUE", critique)

    print_progress("5/5", "Revise", "started")
    context += f"""

## Critique

{critique}
"""

    final_report = output_text(
        llm_call(client, model_name, REVISE_INSTRUCTIONS, context), "Revise"
    )
    print_progress("5/5", "Revise", "completed")
    return final_report


def parse_question():
    parser = argparse.ArgumentParser(
        description="Plan and run tool-using research, then refine its report.",
    )
    parser.add_argument("question", help="Research question to investigate")
    return parser.parse_args().question.strip()


def main():
    question = parse_question()
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
        final_report = research_workflow(
            client,
            model_name,
            question,
            tavily_api_key,
        )
        print_block("FINAL REPORT", final_report)
        print_progress("Run", "Deep research", "completed")
        return 0
    except Exception as error:
        print_error(f"Run failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

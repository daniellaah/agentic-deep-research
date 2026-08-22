"""Run a fixed write-critic-revise workflow and save its artifacts."""

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import Response

PROJECT_ROOT = Path(__file__).resolve().parent

WRITE_INSTRUCTIONS = (
    "You are the writer in a fixed report-refinement workflow. "
    "Answer the user's question directly and completely. "
    "Follow the requested scope and output format. "
    "Output only the initial Markdown report without commentary, follow-up questions, "
    "or unrelated content."
)

CRITIC_INSTRUCTIONS = (
    "You are the critic in a fixed report-refinement workflow. "
    "Review the draft against the original question for structure, reasoning, completeness, "
    "clarity, and instruction following. Give concise, actionable revision guidance in "
    "Markdown. Do not rewrite the report. Because you have no research tools or external "
    "evidence, do not claim to verify factual accuracy, sources, or citations. "
    "Output only the critique."
)

REVISE_INSTRUCTIONS = (
    "You are the reviser in a fixed report-refinement workflow. "
    "Produce the final Markdown report that answers the original question. "
    "Preserve sound draft content and apply useful critique while following the requested "
    "scope and output format. Do not mention the draft, critique, or refinement workflow. "
    "Output only the final report."
)


def llm_call(
    client: OpenAI,
    model_name: str,
    instructions: str,
    model_input: str,
) -> Response:
    return client.responses.create(
        instructions=instructions,
        model=model_name,
        input=model_input,
        store=False,
    )


def require_output_text(openai_response: Response, stage_name: str) -> str:
    output_text = openai_response.output_text
    if not output_text.strip():
        raise RuntimeError(f"The {stage_name} stage did not contain output text.")
    return output_text


def create_run_directory() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{timestamp}-{uuid4().hex[:8]}"
    run_directory = PROJECT_ROOT / "runs" / run_id
    run_directory.mkdir(parents=True)
    return run_directory


def write_artifact(run_directory: Path, filename: str, content: str) -> Path:
    artifact_path = run_directory / filename
    artifact_path.write_text(content, encoding="utf-8")
    return artifact_path


def research_workflow(client: OpenAI, model_name: str, question: str) -> tuple[str, Path]:
    stage = "Write"
    print(f"[1/3] {stage} started: {model_name}")
    write_response = llm_call(client, model_name, WRITE_INSTRUCTIONS, question)
    draft = require_output_text(write_response, stage)
    run_directory = create_run_directory()
    draft_path = write_artifact(run_directory, "draft.md", draft)
    print(f"[1/3] {stage} completed: {draft_path}")

    stage = "Critic"
    critic_input = f"## Original question\n\n{question}\n\n## Draft to critique\n\n{draft}"
    print(f"[2/3] {stage} started: {model_name}")
    critic_response = llm_call(
        client,
        model_name,
        CRITIC_INSTRUCTIONS,
        critic_input,
    )
    critique = require_output_text(critic_response, stage)
    critique_path = write_artifact(run_directory, "critique.md", critique)
    print(f"[2/3] {stage} completed: {critique_path}")

    stage = "Revise"
    revise_input = (
        f"## Original question\n\n{question}\n\n## Draft\n\n{draft}\n\n## Critique\n\n{critique}"
    )
    print(f"[3/3] {stage} started: {model_name}")
    revise_response = llm_call(
        client,
        model_name,
        REVISE_INSTRUCTIONS,
        revise_input,
    )
    final_report = require_output_text(revise_response, stage)
    report_path = write_artifact(run_directory, "report.md", final_report)
    print(f"[3/3] {stage} completed: {report_path}")

    return final_report, report_path


def parse_question() -> str:
    parser = argparse.ArgumentParser(
        description="Run a fixed write-critic-revise workflow and save its Markdown artifacts.",
    )
    parser.add_argument("question", help="Question to send to the model")
    args = parser.parse_args()
    return str(args.question).strip()


def main() -> int:
    question = parse_question()
    if not question:
        print("Error: question must not be empty.", file=sys.stderr)
        return 2

    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    model_name = os.environ.get("MODEL_NAME", "").strip()
    missing_variables = [
        name
        for name, value in (
            ("OPENAI_API_KEY", api_key),
            ("MODEL_NAME", model_name),
        )
        if not value
    ]
    if missing_variables:
        missing = ", ".join(missing_variables)
        print(f"Error: missing required environment variable(s): {missing}.", file=sys.stderr)
        return 2

    try:
        client = OpenAI(api_key=api_key)
        final_report, report_path = research_workflow(client, model_name, question)

        print(f"\nFinal report:\n{final_report}\n")
        print(f"Report written: {report_path}")
        print("Run completed.")
        return 0
    except Exception as error:
        print(f"Run failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

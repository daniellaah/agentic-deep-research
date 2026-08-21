"""Make one OpenAI Responses API request and save its answer."""

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


def llm_call(client: OpenAI, model_name: str, model_input: str) -> Response:
    return client.responses.create(
        instructions=(
            "You are a research assistant. Answer the user's question directly. "
            "Follow the requested scope and output format. "
            "Output only the final report without follow-up questions or unrelated content."
        ),
        model=model_name,
        input=model_input,
        store=False,
    )


def write_report(answer: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{timestamp}-{uuid4().hex[:8]}"
    run_directory = PROJECT_ROOT / "runs" / run_id
    run_directory.mkdir(parents=True)

    report_path = run_directory / "report.md"
    report_path.write_text(answer, encoding="utf-8")
    return report_path


def parse_question() -> str:
    parser = argparse.ArgumentParser(
        description="Make one OpenAI Responses API call and save its answer as Markdown.",
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
        print(f"Model request started: {model_name}")

        client = OpenAI(api_key=api_key)
        openai_response = llm_call(client, model_name, question)
        answer = openai_response.output_text
        if not answer.strip():
            raise RuntimeError("The model response did not contain output text.")

        print(f"Model response received.\n{answer}\n")

        report_path = write_report(answer)

        print(f"Report written: {report_path}")
        print("Run completed.")
        return 0
    except Exception as error:
        print(f"Run failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

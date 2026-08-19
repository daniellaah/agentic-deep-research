"""Command-line entry point for a bounded web-research run."""

import argparse
import os
from collections.abc import Sequence

from dotenv import load_dotenv
from openai import OpenAI

from .models import ResearchBudget, ResearchRequest
from .runner import OpenAIAgentRunner
from .workflow import run_research


def main(argv: Sequence[str] | None = None) -> None:
    """Run one research topic and print its report and execution summary."""
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)
    model = args.model or os.getenv("MODEL_NAME")
    if not model:
        parser.error("set MODEL_NAME in .env or pass --model")

    request = ResearchRequest(
        topic=args.topic,
        language=args.language,
        budget=ResearchBudget(
            max_tool_calls=args.max_tool_calls,
            max_output_tokens=args.max_output_tokens,
        ),
    )
    runner = OpenAIAgentRunner(client=OpenAI(), model=model)
    result = run_research(request, runner=runner)

    print(result.report)
    if result.sources:
        print("\nSources")
        for index, source in enumerate(result.sources, start=1):
            print(f"{index}. {source.title}: {source.url}")

    if args.show_trace and result.trace:
        print("\nResearch trace")
        for index, step in enumerate(result.trace, start=1):
            print(f"{index}. {step.action}: {step.detail}")

    print(
        "\nRun summary: "
        f"status={result.status}, stop_reason={result.stop_reason}, "
        f"tool_calls={len(result.trace)}, total_tokens={result.usage.total_tokens}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deep-research",
        description="Run a bounded, citation-grounded web research task.",
    )
    parser.add_argument("topic", help="Question or topic to research")
    parser.add_argument("--model", help="OpenAI model; defaults to MODEL_NAME")
    parser.add_argument(
        "--language",
        default="the same language as the request",
        help="Language for the final report",
    )
    parser.add_argument("--max-tool-calls", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=4_000)
    parser.add_argument(
        "--show-trace",
        action="store_true",
        help="Print the observable search and page-reading actions",
    )
    return parser


if __name__ == "__main__":
    main()

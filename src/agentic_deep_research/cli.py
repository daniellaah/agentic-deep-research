"""Command-line entry point for a bounded web-research run."""

import argparse
import os
from collections.abc import Sequence

from dotenv import load_dotenv
from openai import OpenAI

from .models import ResearchBudget, ResearchRequest
from .planning import OpenAIAdaptivePlanner
from .reporting import OpenAIReportAgent
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
            max_research_steps=args.max_research_steps,
            max_parallel_workers=args.max_parallel_workers,
            max_context_chars=args.max_context_chars,
            max_verification_tool_calls=args.max_verification_tool_calls,
            max_revision_rounds=args.max_revision_rounds,
        ),
    )
    client = OpenAI()
    runner = OpenAIAgentRunner(client=client, model=model)
    planner = OpenAIAdaptivePlanner(client=client, model=model)
    report_agent = OpenAIReportAgent(client=client, model=model)
    result = run_research(
        request,
        runner=runner,
        planner=planner,
        report_agent=report_agent,
    )

    print(result.report)
    if result.sources:
        print("\nSources")
        for index, source in enumerate(result.sources, start=1):
            print(f"{index}. {source.title}: {source.url}")

    if args.show_trace and result.trace:
        if result.plan is not None:
            print("\nResearch plan")
            for question in result.plan.questions:
                print(f"{question.id}. {question.question}")
        print("\nResearch trace")
        for index, step in enumerate(result.trace, start=1):
            print(f"{index}. {step.action}: {step.detail}")

    print(
        "\nRun summary: "
        f"status={result.status}, stop_reason={result.stop_reason}, "
        f"tool_calls={sum(step.kind == 'tool' for step in result.trace)}, "
        f"research_steps={len(result.findings)}, revisions={result.revision_count}, "
        f"citation_checks={len(result.citation_checks)}, "
        f"total_tokens={result.usage.total_tokens}"
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
    parser.add_argument("--max-output-tokens", type=int, default=20_000)
    parser.add_argument("--max-research-steps", type=int, default=4)
    parser.add_argument("--max-parallel-workers", type=int, default=2)
    parser.add_argument("--max-context-chars", type=int, default=8_000)
    parser.add_argument("--max-verification-tool-calls", type=int, default=2)
    parser.add_argument("--max-revision-rounds", type=int, default=2)
    parser.add_argument(
        "--show-trace",
        action="store_true",
        help="Print the observable search and page-reading actions",
    )
    return parser


if __name__ == "__main__":
    main()

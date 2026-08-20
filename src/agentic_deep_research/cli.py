"""Command-line entry point for durable, bounded web research."""

import argparse
import os
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from openai import OpenAI

from .models import ResearchBudget, ResearchRequest, ResearchResult
from .planning import OpenAIAdaptivePlanner
from .reporting import OpenAIReportAgent
from .runner import OpenAIAgentRunner
from .runtime import ResearchRuntime, RuntimeOutcome, SQLiteCheckpointStore


def main(argv: Sequence[str] | None = None) -> None:
    """Start or control one durable research run."""
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)
    action = _selected_action(args)
    _validate_args(parser, args, action)
    store = SQLiteCheckpointStore(args.checkpoint_db)

    if action == "cancel":
        state = ResearchRuntime(store=store).request_cancel(args.cancel, reason=args.reason)
        _print_state(state.run_id, state.status, state.termination_reason)
        return
    if action == "reject":
        outcome = ResearchRuntime(store=store).reject(args.reject, reason=args.reason)
        _print_outcome(outcome, show_trace=args.show_trace)
        return

    controlled_run_id = args.resume or args.approve
    stored_state = store.load(controlled_run_id) if controlled_run_id is not None else None
    model = args.model or (stored_state.model_name if stored_state is not None else "")
    model = model or os.getenv("MODEL_NAME")
    if not model:
        parser.error("set MODEL_NAME in .env or pass --model")
    max_output_tokens = (
        stored_state.request.budget.max_output_tokens
        if stored_state is not None
        else args.max_output_tokens
    )
    runtime = _build_runtime(store, model, max_output_tokens)

    if action == "resume":
        outcome = runtime.resume(args.resume, retry_ambiguous=args.retry_ambiguous)
    elif action == "approve":
        outcome = runtime.approve(args.approve)
    else:
        new_run_id = args.run_id or f"run-{uuid4().hex}"
        print(f"Starting run_id={new_run_id}", flush=True)
        outcome = runtime.start(
            _build_request(args),
            run_id=new_run_id,
            require_approval=args.require_approval,
        )
    _print_outcome(outcome, show_trace=args.show_trace)
    if outcome.state.status == "failed":
        raise SystemExit(1)


def _build_runtime(
    store: SQLiteCheckpointStore,
    model: str,
    max_output_tokens: int,
) -> ResearchRuntime:
    # Runtime owns visible retries, so disable the SDK's hidden retry layer.
    client = OpenAI(max_retries=0)
    return ResearchRuntime(
        store=store,
        runner=OpenAIAgentRunner(client=client, model=model),
        planner=OpenAIAdaptivePlanner(
            client=client,
            model=model,
            max_output_tokens=max_output_tokens,
        ),
        report_agent=OpenAIReportAgent(
            client=client,
            model=model,
            max_output_tokens=max_output_tokens,
        ),
    )


def _build_request(args: argparse.Namespace) -> ResearchRequest:
    return ResearchRequest(
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


def _print_outcome(outcome: RuntimeOutcome, *, show_trace: bool) -> None:
    state = outcome.state
    if state.status == "waiting_for_human":
        if state.current_step == "ambiguous_effect":
            print(state.error_message or "A previous provider call has an unknown outcome.")
            print(
                "Retry after confirming the original process stopped: "
                f"uv run deep-research --resume {state.run_id} --retry-ambiguous"
            )
            return
        print(f"Research plan for run {state.run_id}")
        if state.plan is not None:
            for question in state.plan.questions:
                print(f"{question.id}. {question.question}")
        print(f"\nApprove: uv run deep-research --approve {state.run_id}")
        print(f"Reject:  uv run deep-research --reject {state.run_id}")
        return
    if outcome.result is None:
        _print_state(state.run_id, state.status, state.termination_reason or state.error_message)
        return
    _print_result(outcome.result, show_trace=show_trace)
    print(f"run_id={state.run_id}, checkpoint_version={state.state_version}")


def _print_result(result: ResearchResult, *, show_trace: bool) -> None:
    print(result.report)
    if result.sources:
        print("\nSources")
        for index, source in enumerate(result.sources, start=1):
            print(f"{index}. {source.title}: {source.url}")

    if show_trace and result.trace:
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


def _print_state(run_id: str, status: str, detail: str | None) -> None:
    suffix = f", detail={detail}" if detail else ""
    print(f"run_id={run_id}, status={status}{suffix}")


def _selected_action(args: argparse.Namespace) -> str:
    if args.resume:
        return "resume"
    if args.approve:
        return "approve"
    if args.reject:
        return "reject"
    if args.cancel:
        return "cancel"
    return "start"


def _validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    action: str,
) -> None:
    if action == "start" and not args.topic:
        parser.error("a topic is required when starting a run")
    if action != "start" and args.topic:
        parser.error("topic cannot be combined with a run control action")
    if action != "start" and args.run_id:
        parser.error("--run-id is only valid when starting a run")
    if action != "start" and args.require_approval:
        parser.error("--require-approval is only valid when starting a run")
    if action != "resume" and args.retry_ambiguous:
        parser.error("--retry-ambiguous is only valid with --resume")
    if args.reason and action not in {"reject", "cancel"}:
        parser.error("--reason is only valid with --reject or --cancel")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deep-research",
        description="Run or control a durable, citation-grounded research task.",
    )
    parser.add_argument("topic", nargs="?", help="Question or topic to research")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--resume", metavar="RUN_ID", help="Resume a failed run")
    actions.add_argument("--approve", metavar="RUN_ID", help="Approve a pending plan")
    actions.add_argument("--reject", metavar="RUN_ID", help="Reject a pending plan")
    actions.add_argument("--cancel", metavar="RUN_ID", help="Cancel a run cooperatively")
    parser.add_argument("--run-id", help="Stable run ID; generated automatically by default")
    parser.add_argument(
        "--checkpoint-db",
        type=Path,
        default=Path(".research-runs/checkpoints.sqlite3"),
        help="Checkpoint database (default: .research-runs/checkpoints.sqlite3)",
    )
    parser.add_argument(
        "--require-approval",
        action="store_true",
        help="Pause after planning and before the first research call",
    )
    parser.add_argument(
        "--retry-ambiguous",
        action="store_true",
        help="Retry an in-flight call only after confirming its old process stopped",
    )
    parser.add_argument("--reason", default="", help="Reason for rejecting or cancelling")
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

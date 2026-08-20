"""Local FastAPI product surface for durable deep-research runs."""

from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query, Response, status
from fastapi.sse import EventSourceResponse, ServerSentEvent
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .artifacts import ExportIntegrityError, export_result
from .checkpoint import RunState, SQLiteCheckpointStore
from .events import event_to_dict
from .models import ResearchBrief, ResearchBudget, ResearchRequest
from .planning import OpenAIAdaptivePlanner
from .reporting import OpenAIReportAgent
from .runner import OpenAIAgentRunner
from .runtime import ResearchRuntime
from .scoping import OpenAIResearchScoper
from .service import (
    ResearchService,
    RunAlreadyScheduledError,
    ServiceBusyError,
    ServiceLimits,
)

BriefItem = Annotated[str, Field(min_length=1, max_length=2_000)]


class _APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BudgetPayload(_APIModel):
    max_tool_calls: int = Field(default=8, ge=1)
    max_output_tokens: int = Field(default=20_000, ge=1)
    max_research_steps: int = Field(default=4, ge=1)
    max_parallel_workers: int = Field(default=2, ge=1)
    max_context_chars: int = Field(default=8_000, ge=1)
    max_verification_tool_calls: int = Field(default=2, ge=0)
    max_revision_rounds: int = Field(default=2, ge=0)

    def to_domain(self) -> ResearchBudget:
        return ResearchBudget(**self.model_dump())


class ResearchBriefPayload(_APIModel):
    research_question: str = Field(min_length=1, max_length=10_000)
    objective: str = Field(min_length=1, max_length=10_000)
    scope_inclusions: list[BriefItem] = Field(default_factory=list, max_length=32)
    scope_exclusions: list[BriefItem] = Field(default_factory=list, max_length=32)
    constraints: list[BriefItem] = Field(default_factory=list, max_length=32)
    deliverable: str = Field(
        default="A source-backed Markdown report.",
        min_length=1,
        max_length=2_000,
    )
    success_criteria: list[BriefItem] = Field(default_factory=list, max_length=32)

    def to_domain(self) -> ResearchBrief:
        return ResearchBrief(
            research_question=self.research_question.strip(),
            objective=self.objective.strip(),
            scope_inclusions=_clean_items(self.scope_inclusions),
            scope_exclusions=_clean_items(self.scope_exclusions),
            constraints=_clean_items(self.constraints),
            deliverable=self.deliverable.strip(),
            success_criteria=_clean_items(self.success_criteria),
        )


class CreateRunPayload(_APIModel):
    topic: str = Field(min_length=1, max_length=10_000)
    run_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$",
    )
    language: str = Field(
        default="the same language as the request",
        min_length=1,
        max_length=200,
    )
    min_sources: int = Field(default=2, ge=1)
    require_citations: bool = True
    require_plan_approval: bool = True
    budget: BudgetPayload = Field(default_factory=BudgetPayload)
    brief: ResearchBriefPayload | None = None

    def to_domain(self) -> ResearchRequest:
        return ResearchRequest(
            topic=self.topic.strip(),
            language=self.language.strip(),
            budget=self.budget.to_domain(),
            min_sources=self.min_sources,
            require_citations=self.require_citations,
            brief=None if self.brief is None else self.brief.to_domain(),
        )


class ClarificationAnswerPayload(_APIModel):
    answer: str = Field(min_length=1, max_length=10_000)
    expected_state_version: int = Field(ge=0)


class PlanQuestionPayload(_APIModel):
    question: str = Field(min_length=1, max_length=10_000)
    rationale: str = Field(default="", max_length=2_000)


class PlanEditPayload(_APIModel):
    expected_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    objective: str | None = Field(default=None, min_length=1, max_length=10_000)
    questions: list[PlanQuestionPayload] = Field(min_length=1, max_length=16)


class PlanDecisionPayload(_APIModel):
    expected_plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason: str = Field(default="", max_length=2_000)


class CancelPayload(_APIModel):
    reason: str = Field(default="", max_length=2_000)


class ResumePayload(_APIModel):
    retry_ambiguous: bool = False


def create_app(service: ResearchService) -> FastAPI:
    """Build the local single-process API around one application service."""

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        service.close(wait=True)

    app = FastAPI(
        title="Agentic Deep Research API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "testserver"],
        www_redirect=False,
    )
    app.state.research_service = service

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/research-runs", status_code=status.HTTP_202_ACCEPTED)
    def create_run(payload: CreateRunPayload) -> dict[str, object]:
        try:
            state = service.start(
                payload.to_domain(),
                run_id=payload.run_id,
                require_approval=payload.require_plan_approval,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except FileExistsError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ServiceBusyError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return _run_view(state)

    @app.get("/v1/research-runs/{run_id}", response_model=None)
    def get_run(
        run_id: str,
        response: Response,
        if_none_match: Annotated[str | None, Header()] = None,
    ) -> dict[str, object] | Response:
        state = _load(service, run_id)
        etag = _state_etag(state)
        if if_none_match == etag:
            return Response(status_code=304, headers={"ETag": etag})
        response.headers["ETag"] = etag
        response.headers["Cache-Control"] = "private, no-store"
        return _run_view(state)

    @app.post(
        "/v1/research-runs/{run_id}/clarification",
        status_code=status.HTTP_202_ACCEPTED,
    )
    def answer_clarification(
        run_id: str,
        payload: ClarificationAnswerPayload,
    ) -> dict[str, object]:
        try:
            state = service.answer_clarification(
                run_id,
                answer=payload.answer,
                expected_state_version=payload.expected_state_version,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            _raise_control_error(error)
        return _run_view(state)

    @app.put("/v1/research-runs/{run_id}/plan")
    def edit_plan(run_id: str, payload: PlanEditPayload) -> dict[str, object]:
        try:
            state = service.edit_plan(
                run_id,
                objective=payload.objective,
                questions=tuple(
                    (item.question, item.rationale) for item in payload.questions
                ),
                expected_plan_hash=payload.expected_plan_hash,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            _raise_control_error(error)
        return _run_view(state)

    @app.post(
        "/v1/research-runs/{run_id}/plan/approve",
        status_code=status.HTTP_202_ACCEPTED,
    )
    def approve_plan(
        run_id: str,
        payload: PlanDecisionPayload,
    ) -> dict[str, object]:
        try:
            state = service.approve_plan(
                run_id,
                expected_plan_hash=payload.expected_plan_hash,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            _raise_control_error(error)
        return _run_view(state)

    @app.post("/v1/research-runs/{run_id}/plan/reject")
    def reject_plan(
        run_id: str,
        payload: PlanDecisionPayload,
    ) -> dict[str, object]:
        try:
            state = service.reject_plan(
                run_id,
                expected_plan_hash=payload.expected_plan_hash,
                reason=payload.reason,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            _raise_control_error(error)
        return _run_view(state)

    @app.post(
        "/v1/research-runs/{run_id}/cancel",
        status_code=status.HTTP_202_ACCEPTED,
    )
    def cancel_run(run_id: str, payload: CancelPayload) -> dict[str, object]:
        try:
            state = service.cancel(run_id, reason=payload.reason)
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            _raise_control_error(error)
        return _run_view(state)

    @app.post(
        "/v1/research-runs/{run_id}/resume",
        status_code=status.HTTP_202_ACCEPTED,
    )
    def resume_run(
        run_id: str,
        payload: ResumePayload,
    ) -> dict[str, object]:
        try:
            state = service.resume(
                run_id,
                retry_ambiguous=payload.retry_ambiguous,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            _raise_control_error(error)
        return _run_view(state)

    @app.get("/v1/research-runs/{run_id}/events")
    def list_events(
        run_id: str,
        response: Response,
        after_sequence: int = Query(default=-1, ge=-1),
        limit: int = Query(default=100, ge=1, le=100),
    ) -> dict[str, object]:
        _load(service, run_id)
        response.headers["Cache-Control"] = "private, no-store"
        events = list(service.store.list_events(
            run_id,
            after_sequence=after_sequence,
            limit=limit,
        ))
        next_after = events[-1].sequence if events else after_sequence
        state = service.get(run_id)
        terminal_state = state.status in {"completed", "failed", "cancelled"}
        if terminal_state and len(events) < limit:
            events.extend(
                service.store.list_events(
                    run_id,
                    after_sequence=next_after,
                    limit=limit - len(events),
                )
            )
            next_after = events[-1].sequence if events else after_sequence
        has_more = bool(
            service.store.list_events(
                run_id,
                after_sequence=next_after,
                limit=1,
            )
        )
        return {
            "schema_version": 1,
            "run_id": run_id,
            "events": [event_to_dict(event) for event in events],
            "next_after": next_after,
            "terminal": terminal_state and not has_more,
        }

    @app.get(
        "/v1/research-runs/{run_id}/events/stream",
        response_class=EventSourceResponse,
    )
    async def stream_events(
        run_id: str,
        after_sequence: int = Query(default=-1, ge=-1),
        follow: bool = True,
        last_event_id: Annotated[str | None, Header()] = None,
    ) -> AsyncIterator[ServerSentEvent]:
        _load(service, run_id)
        cursor = _event_cursor(after_sequence, last_event_id)
        while True:
            events = await asyncio.to_thread(
                service.store.list_events,
                run_id,
                after_sequence=cursor,
                limit=100,
            )
            for event in events:
                cursor = event.sequence
                yield ServerSentEvent(
                    data=event_to_dict(event),
                    event=event.event_type,
                    id=str(event.sequence),
                    retry=1_000,
                )
            if not follow:
                return
            if events:
                continue
            state = await asyncio.to_thread(service.get, run_id)
            if state.status in {"completed", "failed", "cancelled"}:
                final_events = await asyncio.to_thread(
                    service.store.list_events,
                    run_id,
                    after_sequence=cursor,
                    limit=100,
                )
                if final_events:
                    for event in final_events:
                        cursor = event.sequence
                        yield ServerSentEvent(
                            data=event_to_dict(event),
                            event=event.event_type,
                            id=str(event.sequence),
                            retry=1_000,
                        )
                    continue
                return
            await asyncio.sleep(0.5)

    @app.get("/v1/research-runs/{run_id}/artifacts/{artifact_format}")
    def get_artifact(
        run_id: str,
        artifact_format: str,
        if_none_match: Annotated[str | None, Header()] = None,
    ) -> Response:
        if artifact_format not in {"research.json", "report.md"}:
            raise HTTPException(status_code=404, detail="artifact format not found")
        state = _load(service, run_id)
        try:
            artifact = export_result(
                state,
                format="json" if artifact_format.endswith(".json") else "markdown",
            )
        except ExportIntegrityError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        headers = {
            "ETag": artifact.etag,
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": f'attachment; filename="{artifact.filename}"',
        }
        if if_none_match == artifact.etag:
            return Response(status_code=304, headers=headers)
        return Response(
            content=artifact.body,
            media_type=artifact.media_type,
            headers=headers,
        )

    return app


def build_local_service(
    *,
    checkpoint_db: Path,
    model: str,
    max_concurrent_runs: int = 2,
    limits: ServiceLimits | None = None,
) -> ResearchService:
    """Compose the local API with one shared OpenAI client and SQLite store."""
    store = SQLiteCheckpointStore(checkpoint_db)
    client = OpenAI(max_retries=0)

    def runtime_factory(request: ResearchRequest) -> ResearchRuntime:
        max_output_tokens = request.budget.max_output_tokens
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
            scoper=OpenAIResearchScoper(
                client=client,
                model=model,
                max_output_tokens=min(5_000, max_output_tokens),
            ),
        )

    return ResearchService(
        store=store,
        runtime_factory=runtime_factory,
        max_concurrent_runs=max_concurrent_runs,
        limits=limits,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the explicitly local, single-process product API."""
    load_dotenv()
    parser = argparse.ArgumentParser(
        prog="deep-research-api",
        description="Serve the local durable deep-research API.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--checkpoint-db",
        type=Path,
        default=Path(".research-runs/checkpoints.sqlite3"),
    )
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", ""))
    parser.add_argument("--max-concurrent-runs", type=int, default=2)
    args = parser.parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost"}:
        parser.error("the unauthenticated learning service may only bind to loopback")
    if not args.model:
        parser.error("set MODEL_NAME in .env or pass --model")
    if not 1 <= args.port <= 65_535:
        parser.error("--port must be between 1 and 65535")
    service = build_local_service(
        checkpoint_db=args.checkpoint_db,
        model=args.model,
        max_concurrent_runs=args.max_concurrent_runs,
    )
    uvicorn.run(
        create_app(service),
        host=args.host,
        port=args.port,
        workers=1,
    )


def _clean_items(values: Sequence[str]) -> tuple[str, ...]:
    cleaned = tuple(value.strip() for value in values)
    if any(not value for value in cleaned):
        raise ValueError("brief list values must not be empty")
    return cleaned


def _load(service: ResearchService, run_id: str) -> RunState:
    try:
        return service.get(run_id)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail="research run not found") from error


def _raise_control_error(error: Exception) -> None:
    if isinstance(error, FileNotFoundError):
        raise HTTPException(status_code=404, detail="research run not found") from error
    if isinstance(error, ServiceBusyError):
        raise HTTPException(status_code=503, detail=str(error)) from error
    if isinstance(error, (RunAlreadyScheduledError, RuntimeError, ValueError)):
        raise HTTPException(status_code=409, detail=str(error)) from error
    raise error


def _state_etag(state: RunState) -> str:
    return f'W/"run-{state.run_id}-v{state.state_version}"'


def _event_cursor(after_sequence: int, last_event_id: str | None) -> int:
    if last_event_id is None:
        return after_sequence
    try:
        value = int(last_event_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail="invalid Last-Event-ID") from error
    if value < -1:
        raise HTTPException(status_code=400, detail="invalid Last-Event-ID")
    return value


def _run_view(state: RunState) -> dict[str, object]:
    result = state.result
    return {
        "run_id": state.run_id,
        "runtime_status": state.status,
        "current_step": state.current_step,
        "state_version": state.state_version,
        "created_at": state.created_at,
        "updated_at": state.updated_at,
        "action_required": _action_required(state),
        "approval_status": state.approval_status,
        "cancel_requested": state.cancel_requested,
        "error_type": state.error_type,
        "brief": _brief_view(state.brief),
        "clarification": _clarification_view(state),
        "plan": _plan_view(state),
        "plan_hash": state.plan_hash,
        "result_available": result is not None,
        "result_quality": (
            None
            if result is None
            else {"status": result.status, "stop_reason": result.stop_reason}
        ),
    }


def _action_required(state: RunState) -> str | None:
    if state.status != "waiting_for_human":
        return None
    if state.current_step == "clarification":
        return "clarification"
    if state.current_step == "ambiguous_effect":
        return "ambiguous_effect"
    if state.approval_status == "pending":
        return "plan_approval"
    return "human_action"


def _brief_view(brief: ResearchBrief | None) -> dict[str, object] | None:
    if brief is None:
        return None
    return {
        "research_question": brief.research_question,
        "objective": brief.objective,
        "scope_inclusions": list(brief.scope_inclusions),
        "scope_exclusions": list(brief.scope_exclusions),
        "constraints": list(brief.constraints),
        "deliverable": brief.deliverable,
        "success_criteria": list(brief.success_criteria),
    }


def _clarification_view(state: RunState) -> dict[str, object] | None:
    clarification = state.clarification
    if clarification is None or not clarification.needs_clarification:
        return None
    return {
        "question": clarification.question,
        "reason": clarification.reason,
        "state_version": state.state_version,
    }


def _plan_view(state: RunState) -> dict[str, object] | None:
    plan = state.plan
    if plan is None:
        return None
    return {
        "objective": plan.objective,
        "revision": plan.revision,
        "questions": [
            {
                "id": question.id,
                "question": question.question,
                "rationale": question.rationale,
                "priority": question.priority,
            }
            for question in plan.questions
        ],
    }


if __name__ == "__main__":
    main()

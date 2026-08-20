import json
import time

from fastapi.testclient import TestClient

from agentic_deep_research.api import create_app
from agentic_deep_research.checkpoint import (
    RunState,
    SQLiteCheckpointStore,
    _transition,
)
from agentic_deep_research.models import (
    AgentRun,
    Citation,
    ClarificationDecision,
    PlanningRun,
    ResearchBrief,
    ResearchBudget,
    ResearchPlan,
    ResearchQuestion,
    ResearchRequest,
    ScopingRun,
    Source,
)
from agentic_deep_research.runtime import ResearchRuntime
from agentic_deep_research.service import ResearchService


class ClarifyingScoper:
    def __init__(self) -> None:
        self.call_count = 0

    def scope(self, *, messages: tuple[object, ...]) -> ScopingRun:
        self.call_count += 1
        if len(messages) == 1:
            return ScopingRun(
                clarification=ClarificationDecision(
                    needs_clarification=True,
                    reason="The intended audience changes the scope.",
                    question="Who is the report for?",
                )
            )
        return ScopingRun(
            clarification=ClarificationDecision(
                needs_clarification=False,
                reason="The audience is now clear.",
            ),
            brief=ResearchBrief(
                research_question="How should research agents be made reliable?",
                objective="Give engineers an actionable reliability guide.",
                scope_inclusions=("Evidence quality",),
                success_criteria=("Use source-backed claims",),
            ),
        )


class APIPlanner:
    def plan(self, **_: object) -> PlanningRun:
        return PlanningRun(
            plan=ResearchPlan(
                objective="Generated objective",
                questions=(
                    ResearchQuestion(
                        id="q1",
                        question="Generated question",
                        priority=1,
                    ),
                ),
            )
        )


class APIRunner:
    def __init__(self) -> None:
        self.tasks: list[str] = []

    def run(
        self,
        *,
        instructions: str,
        task: str,
        budget: ResearchBudget,
    ) -> AgentRun:
        del instructions, budget
        self.tasks.append(task)
        source = Source("Reliability source", "https://example.com/reliability")
        report = "Evaluation improves reliability. [1]"
        start = report.index("[1]")
        return AgentRun(
            report=report,
            sources=(source,),
            citations=(Citation(source, start, start + 3),),
        )


def test_local_api_runs_clarification_plan_review_events_and_exports(tmp_path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.sqlite3")
    scoper = ClarifyingScoper()
    planner = APIPlanner()
    runner = APIRunner()

    def runtime_factory(_: ResearchRequest) -> ResearchRuntime:
        return ResearchRuntime(
            store=store,
            runner=runner,
            planner=planner,
            scoper=scoper,
        )

    service = ResearchService(
        store=store,
        runtime_factory=runtime_factory,
        max_concurrent_runs=1,
    )
    with TestClient(create_app(service)) as client:
        created = client.post(
            "/v1/research-runs",
            json={
                "topic": "A private research topic",
                "run_id": "api-product-run",
                "min_sources": 1,
                "budget": {
                    "max_tool_calls": 1,
                    "max_output_tokens": 2_000,
                    "max_research_steps": 1,
                    "max_parallel_workers": 1,
                    "max_context_chars": 2_000,
                    "max_verification_tool_calls": 0,
                    "max_revision_rounds": 0,
                },
            },
        )
        assert created.status_code == 202
        assert created.json()["runtime_status"] == "created"

        clarification = _wait_for_action(client, "api-product-run", "clarification")
        answered = client.post(
            "/v1/research-runs/api-product-run/clarification",
            json={
                "answer": "The report is for production ML engineers.",
                "expected_state_version": clarification["state_version"],
            },
        )
        assert answered.status_code == 202

        approval = _wait_for_action(client, "api-product-run", "plan_approval")
        original_hash = approval["plan_hash"]
        edited = client.put(
            "/v1/research-runs/api-product-run/plan",
            json={
                "expected_plan_hash": original_hash,
                "objective": "Edited production reliability objective",
                "questions": [
                    {
                        "question": "Edited production reliability question",
                        "rationale": "Focus the report on measurable controls.",
                    }
                ],
            },
        )
        assert edited.status_code == 200
        edited_hash = edited.json()["plan_hash"]
        assert edited_hash != original_hash

        approved = client.post(
            "/v1/research-runs/api-product-run/plan/approve",
            json={"expected_plan_hash": edited_hash},
        )
        assert approved.status_code == 202

        completed = _wait_for_status(client, "api-product-run", "completed")
        assert completed["result_quality"]["status"] == "completed"
        assert scoper.call_count == 2
        assert len(runner.tasks) == 1
        assert "Edited production reliability objective" in runner.tasks[0]
        assert "Edited production reliability question" in runner.tasks[0]

        status_response = client.get("/v1/research-runs/api-product-run")
        cached = client.get(
            "/v1/research-runs/api-product-run",
            headers={"If-None-Match": status_response.headers["etag"]},
        )
        assert cached.status_code == 304

        event_response = client.get("/v1/research-runs/api-product-run/events")
        assert event_response.status_code == 200
        events = event_response.json()["events"]
        event_types = [event["type"] for event in events]
        assert event_types[0] == "run.created"
        assert "scope.clarification_required" in event_types
        assert "scope.clarified" in event_types
        assert "approval.required" in event_types
        assert "plan.edited" in event_types
        assert "plan.approved" in event_types
        assert event_types[-1] == "run.completed"
        assert [event["sequence"] for event in events] == list(range(len(events)))
        assert "A private research topic" not in json.dumps(events)

        stream = client.get(
            "/v1/research-runs/api-product-run/events/stream",
            params={"follow": "false"},
            headers={"Last-Event-ID": "0"},
        )
        assert stream.status_code == 200
        assert stream.headers["content-type"].startswith("text/event-stream")
        assert "id: 0" not in stream.text
        assert "event: run.completed" in stream.text

        json_artifact = client.get(
            "/v1/research-runs/api-product-run/artifacts/research.json"
        )
        assert json_artifact.status_code == 200
        assert json_artifact.json()["research"]["quality"]["status"] == "completed"
        assert "attachment" in json_artifact.headers["content-disposition"]
        cached_artifact = client.get(
            "/v1/research-runs/api-product-run/artifacts/research.json",
            headers={"If-None-Match": json_artifact.headers["etag"]},
        )
        assert cached_artifact.status_code == 304

        markdown_artifact = client.get(
            "/v1/research-runs/api-product-run/artifacts/report.md"
        )
        assert markdown_artifact.status_code == 200
        assert "Evaluation improves reliability" in markdown_artifact.text

        simple_cross_origin_resume = client.post(
            "/v1/research-runs/api-product-run/resume",
            content="",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Origin": "https://attacker.example",
            },
        )
        assert simple_cross_origin_resume.status_code == 422


def test_local_api_rejects_unknown_runs_and_excessive_budgets(tmp_path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.sqlite3")
    runner = APIRunner()
    service = ResearchService(
        store=store,
        runtime_factory=lambda _: ResearchRuntime(store=store, runner=runner),
        max_concurrent_runs=1,
    )
    with TestClient(create_app(service)) as client:
        assert client.get("/health", headers={"Host": "attacker.example"}).status_code == 400
        assert client.get("/v1/research-runs/missing").status_code == 404
        response = client.post(
            "/v1/research-runs",
            json={
                "topic": "Too expensive",
                "budget": {"max_tool_calls": 33},
            },
        )
        assert response.status_code == 422
        assert "service limit" in response.json()["detail"]

        oversized_brief = client.post(
            "/v1/research-runs",
            json={
                "topic": "Bound prompt inputs",
                "brief": {
                    "research_question": "A bounded question",
                    "objective": "A bounded objective",
                    "scope_inclusions": ["x" * 2_001],
                },
            },
        )
        assert oversized_brief.status_code == 422


def test_terminal_event_is_refetched_after_an_empty_racing_read(
    tmp_path,
    monkeypatch,
) -> None:
    store = SQLiteCheckpointStore(tmp_path / "checkpoints.sqlite3")
    state = RunState.create(
        ResearchRequest(topic="Terminal event race"),
        run_id="terminal-race",
    )
    store.create(state)
    store.update(
        state.run_id,
        lambda current: _transition(
            current,
            status="completed",
            current_step="completed",
        ),
    )
    runner = APIRunner()
    service = ResearchService(
        store=store,
        runtime_factory=lambda _: ResearchRuntime(store=store, runner=runner),
    )
    original_list_events = store.list_events
    calls = 0

    def racing_list_events(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ()
        return original_list_events(*args, **kwargs)

    monkeypatch.setattr(store, "list_events", racing_list_events)
    with TestClient(create_app(service)) as client:
        page = client.get("/v1/research-runs/terminal-race/events")
        assert page.status_code == 200
        assert page.json()["terminal"] is True
        assert page.json()["events"][-1]["type"] == "run.completed"

        calls = 0
        stream = client.get("/v1/research-runs/terminal-race/events/stream")
        assert stream.status_code == 200
        assert "event: run.completed" in stream.text


def _wait_for_action(
    client: TestClient,
    run_id: str,
    action: str,
) -> dict[str, object]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/v1/research-runs/{run_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["action_required"] == action:
            return payload
        time.sleep(0.01)
    raise AssertionError(f"run {run_id} did not require {action}")


def _wait_for_status(
    client: TestClient,
    run_id: str,
    status: str,
) -> dict[str, object]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/v1/research-runs/{run_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["runtime_status"] == status:
            return payload
        time.sleep(0.01)
    raise AssertionError(f"run {run_id} did not reach {status}")

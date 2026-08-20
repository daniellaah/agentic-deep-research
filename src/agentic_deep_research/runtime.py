"""Public durable runtime for long-running research workflows."""

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import Event, Thread
from uuid import uuid4

from ._journal import (
    RetryPolicy,
    _component_model,
    _EffectJournal,
    _engine_fingerprint,
    _JournaledPlanner,
    _JournaledReportAgent,
    _JournaledRunner,
    _JournaledScoper,
    _RunCancelled,
    _WaitingForApproval,
)
from .checkpoint import (
    CheckpointStore,
    EffectRecord,
    JsonCheckpointStore,
    RunState,
    SQLiteCheckpointStore,
    _cancelled_state,
    _decode_result,
    _timestamp,
    _transition,
    plan_control_input_hash,
    research_plan_version_hash,
)
from .models import (
    ClarificationDecision,
    ConversationMessage,
    PlanControlKind,
    PlanControlRecord,
    ResearchBrief,
    ResearchPlan,
    ResearchRequest,
    ResearchResult,
    ScopingRun,
    TokenUsage,
)
from .planning import ResearchPlanner, TopicPlanner
from .reporting import ReportAgent
from .runner import AgentRunner
from .scoping import ResearchScoper
from .workflow import run_research

__all__ = [
    "CheckpointStore",
    "EffectRecord",
    "JsonCheckpointStore",
    "ResearchRuntime",
    "RetryPolicy",
    "RunState",
    "RuntimeOutcome",
    "SQLiteCheckpointStore",
]


@dataclass(frozen=True)
class RuntimeOutcome:
    """Latest durable state plus a result when the run completed."""

    state: RunState
    result: ResearchResult | None = None


class ResearchRuntime:
    """Run the existing workflow with durable provider-call checkpoints."""

    def __init__(
        self,
        *,
        store: CheckpointStore,
        runner: AgentRunner | None = None,
        planner: ResearchPlanner | None = None,
        report_agent: ReportAgent | None = None,
        scoper: ResearchScoper | None = None,
        retry_policy: RetryPolicy | None = None,
        lease_ttl_seconds: float = 30,
        sleeper: Callable[[float], None] = time.sleep,
        random_source: Callable[[], float] = random.random,
    ) -> None:
        self._store = store
        self._runner = runner
        self._planner = planner
        self._report_agent = report_agent
        self._scoper = scoper
        self._retry_policy = retry_policy or RetryPolicy()
        if lease_ttl_seconds <= 0:
            raise ValueError("lease_ttl_seconds must be positive")
        self._lease_ttl_seconds = lease_ttl_seconds
        self._sleeper = sleeper
        self._random_source = random_source

    @property
    def model_name(self) -> str:
        """Return the runner model included in durable configuration identity."""
        return _component_model(self._runner)

    @property
    def planner_model_name(self) -> str:
        """Return the planner model, when the planner is model-backed."""
        return _component_model(self._planner)

    @property
    def report_model_name(self) -> str:
        """Return the report-agent model, when reporting is model-backed."""
        return _component_model(self._report_agent)

    @property
    def scoper_model_name(self) -> str:
        """Return the scoping model, when conversational scoping is configured."""
        return _component_model(self._scoper)

    @property
    def corpus_sha256(self) -> str | None:
        """Return the runner's fixed-corpus identity, when one is configured."""
        value = getattr(self._runner, "corpus_sha256", None)
        return value if isinstance(value, str) else None

    @property
    def retry_policy(self) -> RetryPolicy:
        """Return the retry policy enforced and journaled by this runtime."""
        return self._retry_policy

    def start(
        self,
        request: ResearchRequest,
        *,
        run_id: str | None = None,
        require_approval: bool = False,
        conversation: tuple[ConversationMessage, ...] | None = None,
    ) -> RuntimeOutcome:
        """Create and execute a new durable research run."""
        state = self.create(
            request,
            run_id=run_id,
            require_approval=require_approval,
            conversation=conversation,
        )
        return self._execute(state)

    def create(
        self,
        request: ResearchRequest,
        *,
        run_id: str | None = None,
        require_approval: bool = False,
        conversation: tuple[ConversationMessage, ...] | None = None,
    ) -> RunState:
        """Persist a configured run without starting provider-facing work."""
        engine_fingerprint = self._current_engine_fingerprint()
        messages = (
            tuple(conversation)
            if conversation is not None
            else (ConversationMessage(role="user", content=request.topic),)
        )
        if not messages:
            raise ValueError("research conversation must not be empty")
        if any(not isinstance(message, ConversationMessage) for message in messages):
            raise TypeError("research conversation must contain ConversationMessage values")
        state = replace(
            RunState.create(request, run_id=run_id),
            requires_approval=require_approval,
            engine_fingerprint=engine_fingerprint,
            model_name=_component_model(self._runner),
            conversation=messages,
        )
        self._store.create(state)
        return state

    def resume(
        self,
        run_id: str,
        *,
        retry_ambiguous: bool = False,
    ) -> RuntimeOutcome:
        """Resume a run, replaying completed effects from its checkpoint."""
        state = self._store.load(run_id)
        if state.status == "completed":
            return RuntimeOutcome(state=state, result=state.result)
        if state.status == "cancelled":
            return RuntimeOutcome(state=state)
        return self._execute(
            state,
            resume_mode=True,
            retry_ambiguous=retry_ambiguous,
        )

    def get(self, run_id: str) -> RunState:
        """Return the latest persisted state without executing the workflow."""
        return self._store.load(run_id)

    def answer_clarification(
        self,
        run_id: str,
        *,
        answer: str,
        expected_state_version: int | None = None,
    ) -> RunState:
        """Persist one human answer without executing the next scoping call."""
        normalized_answer = answer.strip()
        if not normalized_answer:
            raise ValueError("clarification answer must not be empty")

        def mutation(state: RunState) -> RunState:
            self._validate_control_configuration(state)
            if (
                expected_state_version is not None
                and state.state_version != expected_state_version
            ):
                raise ValueError(
                    "run state changed; reload the clarification before answering"
                )
            clarification = state.clarification
            if (
                state.status != "waiting_for_human"
                or state.current_step != "clarification"
                or clarification is None
                or not clarification.needs_clarification
                or clarification.question is None
            ):
                raise ValueError("run is not waiting for clarification")
            return _transition(
                state,
                status="created",
                current_step="clarification_answered",
                conversation=(
                    *state.conversation,
                    ConversationMessage(
                        role="assistant",
                        content=clarification.question,
                    ),
                    ConversationMessage(role="user", content=normalized_answer),
                ),
                clarification=None,
                error_type=None,
                error_message=None,
            )

        return self._store.update(run_id, mutation)

    def edit_plan(
        self,
        run_id: str,
        plan: ResearchPlan,
        *,
        expected_plan_hash: str,
    ) -> RunState:
        """Append one validated human plan edit using optimistic concurrency."""

        def mutation(state: RunState) -> RunState:
            self._validate_control_configuration(state)
            if (
                state.status != "waiting_for_human"
                or state.current_step != "approval"
                or state.approval_status != "pending"
            ):
                raise ValueError("run is not waiting for plan approval")
            current_plan = state.plan
            if current_plan is None:
                raise ValueError("run has no plan to edit")
            _validate_edited_plan(
                plan,
                state.request.budget.max_research_steps,
            )
            record = _plan_control_record(
                state,
                kind="edit",
                base_plan_hash=expected_plan_hash,
                plan=plan,
            )
            if any(item.control_id == record.control_id for item in state.plan_controls):
                return state
            current_hash = state.plan_hash
            if current_hash != expected_plan_hash:
                raise ValueError("plan changed; reload it before editing")
            if plan == current_plan:
                raise ValueError("edited plan must differ from the current plan")
            return _transition(
                state,
                plan_controls=(*state.plan_controls, record),
                approval_reason="Review the edited plan before research begins.",
                current_step="approval",
            )

        return self._store.update(run_id, mutation)

    def approve_plan(self, run_id: str, *, expected_plan_hash: str) -> RunState:
        """Atomically approve the exact plan version inspected by the caller."""

        def mutation(state: RunState) -> RunState:
            self._validate_control_configuration(state)
            plan = state.plan
            if plan is None:
                raise ValueError("run has no plan to approve")
            _validate_edited_plan(plan, state.request.budget.max_research_steps)
            record = _plan_control_record(
                state,
                kind="approve",
                base_plan_hash=expected_plan_hash,
                plan=plan,
            )
            if any(item.control_id == record.control_id for item in state.plan_controls):
                return state
            if (
                state.status != "waiting_for_human"
                or state.current_step != "approval"
                or state.approval_status != "pending"
            ):
                raise ValueError("run is not waiting for plan approval")
            if state.plan_hash != expected_plan_hash:
                raise ValueError("plan changed; reload it before approving")
            return _transition(
                state,
                status="created",
                current_step="approved",
                plan_controls=(*state.plan_controls, record),
                approval_status="approved",
                approval_reason=None,
            )

        return self._store.update(run_id, mutation)

    def approve(self, run_id: str) -> RuntimeOutcome:
        """Approve a pending plan and continue the durable run."""
        state = self._store.load(run_id)
        if state.plan_hash is None:
            raise ValueError("run has no plan to approve")
        approved = self.approve_plan(
            run_id,
            expected_plan_hash=state.plan_hash,
        )
        return self._execute(approved, resume_mode=True)

    def reject(
        self,
        run_id: str,
        *,
        reason: str = "",
        expected_plan_hash: str | None = None,
    ) -> RuntimeOutcome:
        """Reject a pending plan without starting its research effects."""

        def mutation(state: RunState) -> RunState:
            self._validate_control_configuration(state, allow_store_only=True)
            if state.status != "waiting_for_human" or state.approval_status != "pending":
                raise ValueError("run is not waiting for approval")
            if expected_plan_hash is not None and state.plan_hash != expected_plan_hash:
                raise ValueError("plan changed; reload it before rejecting")
            return _transition(
                state,
                status="cancelled",
                approval_status="rejected",
                approval_reason=reason.strip() or None,
                termination_reason="approval_rejected",
            )

        return RuntimeOutcome(state=self._store.update(run_id, mutation))

    def request_cancel(self, run_id: str, *, reason: str = "") -> RunState:
        """Request cooperative cancellation, or cancel an idle run immediately."""

        def mutation(state: RunState) -> RunState:
            if state.status in {"completed", "cancelled"}:
                return state
            termination_reason = reason.strip() or "cancel_requested"
            if state.status == "running":
                return _transition(
                    state,
                    cancel_requested=True,
                    termination_reason=termination_reason,
                )
            return _transition(
                state,
                status="cancelled",
                cancel_requested=True,
                termination_reason=termination_reason,
            )

        return self._store.update(run_id, mutation)

    def _validate_control_configuration(
        self,
        state: RunState,
        *,
        allow_store_only: bool = False,
    ) -> None:
        """Reject controls from a runtime that cannot safely resume this checkpoint."""
        if self._runner is None:
            if allow_store_only:
                return
            raise RuntimeError(
                "runner configuration is required to mutate this research run"
            )
        if not state.engine_fingerprint:
            return
        if state.engine_fingerprint != self._current_engine_fingerprint():
            raise ValueError(
                "runtime configuration does not match the checkpoint; "
                "use the original model and agent configuration"
            )

    def _execute(
        self,
        state: RunState,
        *,
        resume_mode: bool = False,
        retry_ambiguous: bool = False,
    ) -> RuntimeOutcome:
        owner_id = f"executor-{uuid4().hex}"
        self._store.acquire_lease(state.run_id, owner_id, self._lease_ttl_seconds)
        heartbeat = _LeaseHeartbeat(
            self._store,
            state.run_id,
            owner_id,
            self._lease_ttl_seconds,
        )
        heartbeat.start()
        try:
            outcome = self._execute_with_lease(
                state,
                owner_id,
                resume_mode=resume_mode,
                retry_ambiguous=retry_ambiguous,
            )
            heartbeat.raise_if_failed()
            return outcome
        finally:
            heartbeat.stop()
            self._store.release_lease(state.run_id, owner_id)

    def _execute_with_lease(
        self,
        state: RunState,
        owner_id: str,
        *,
        resume_mode: bool,
        retry_ambiguous: bool,
    ) -> RuntimeOutcome:
        state = self._store.update_owned(
            state.run_id,
            owner_id,
            self._lease_ttl_seconds,
            lambda latest: latest,
        )
        if state.status == "completed":
            return RuntimeOutcome(state=state, result=state.result)
        if state.status == "cancelled":
            return RuntimeOutcome(state=state)
        if resume_mode:
            has_ambiguous_effect = any(
                effect.status == "started" for effect in state.effects
            )
            if has_ambiguous_effect and not retry_ambiguous:
                waiting = self._store.update_owned(
                    state.run_id,
                    owner_id,
                    self._lease_ttl_seconds,
                    _ambiguous_state,
                )
                return RuntimeOutcome(state=waiting)
            if (
                state.status == "waiting_for_human"
                and state.current_step != "ambiguous_effect"
            ):
                return RuntimeOutcome(state=state)
        engine_fingerprint = self._current_engine_fingerprint()
        if state.engine_fingerprint and state.engine_fingerprint != engine_fingerprint:
            raise ValueError(
                "runtime configuration does not match the checkpoint; "
                "use the original model and agent configuration"
            )
        state = self._store.update_owned(
            state.run_id,
            owner_id,
            self._lease_ttl_seconds,
            _starting_state,
        )
        if state.status in {"completed", "cancelled"}:
            return RuntimeOutcome(state=state, result=state.result)
        journal = _EffectJournal(
            self._store,
            state,
            retry_policy=self._retry_policy,
            lease_owner=owner_id,
            lease_ttl_seconds=self._lease_ttl_seconds,
            sleeper=self._sleeper,
            random_source=self._random_source,
        )
        planner = _JournaledPlanner(self._planner or TopicPlanner(), journal)
        report_agent = (
            _JournaledReportAgent(self._report_agent, journal)
            if self._report_agent is not None
            else None
        )
        scoper = (
            _JournaledScoper(self._scoper, journal)
            if self._scoper is not None
            else None
        )
        try:
            if scoper is not None and state.request.brief is None:
                scoping_run = scoper.scope(messages=state.conversation)
                if scoping_run.clarification.needs_clarification:
                    if _completed_clarification_count(journal.state.effects) > 1:
                        raise RuntimeError(
                            "scoper requested another clarification after the user answer"
                        )
                    waiting = self._store.update_owned(
                        state.run_id,
                        owner_id,
                        self._lease_ttl_seconds,
                        lambda latest: _clarification_state(
                            latest,
                            scoping_run.clarification,
                        ),
                    )
                    return RuntimeOutcome(state=waiting)
                if scoping_run.brief is None:
                    raise RuntimeError("scoping completed without a research brief")
                state = self._store.update_owned(
                    state.run_id,
                    owner_id,
                    self._lease_ttl_seconds,
                    lambda latest: _scoped_state(latest, scoping_run.brief),
                )
                if state.status == "cancelled":
                    return RuntimeOutcome(state=state)
            result = run_research(
                state.request,
                runner=_JournaledRunner(self._runner, journal),
                planner=planner,
                report_agent=report_agent,
            )
            result = replace(
                result,
                usage=_add_usage(
                    result.usage,
                    _completed_scoping_usage(journal.state.effects),
                ),
            )
        except _WaitingForApproval:
            return RuntimeOutcome(state=journal.state)
        except _RunCancelled:
            return RuntimeOutcome(state=journal.state)
        except Exception as error:  # noqa: BLE001 - persist arbitrary adapter failures.
            caught_error = error
            failed = self._store.update_owned(
                state.run_id,
                owner_id,
                self._lease_ttl_seconds,
                lambda latest: _failure_state(latest, caught_error),
            )
            return RuntimeOutcome(state=failed)

        completed = self._store.update_owned(
            state.run_id,
            owner_id,
            self._lease_ttl_seconds,
            lambda latest: _completion_state(latest, result),
        )
        return RuntimeOutcome(state=completed, result=completed.result)

    def _current_engine_fingerprint(self) -> str:
        if self._runner is None:
            raise RuntimeError("runner is required to start or resume research")
        return _engine_fingerprint(
            runner=self._runner,
            planner=self._planner or TopicPlanner(),
            report_agent=self._report_agent,
            scoper=self._scoper,
            retry_policy=self._retry_policy,
        )


class _LeaseHeartbeat:
    def __init__(
        self,
        store: CheckpointStore,
        run_id: str,
        owner_id: str,
        ttl_seconds: float,
    ) -> None:
        self._store = store
        self._run_id = run_id
        self._owner_id = owner_id
        self._ttl_seconds = ttl_seconds
        self._stopped = Event()
        self._error: Exception | None = None
        self._thread = Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        self._thread.join()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"execution lease heartbeat failed: {self._run_id}") from self._error

    def _run(self) -> None:
        interval = max(0.01, self._ttl_seconds / 3)
        while not self._stopped.wait(interval):
            try:
                self._store.renew_lease(
                    self._run_id,
                    self._owner_id,
                    self._ttl_seconds,
                )
            except Exception as error:  # noqa: BLE001 - report background lease loss.
                self._error = error
                return


def _ambiguous_state(state: RunState) -> RunState:
    if state.status in {"completed", "cancelled"}:
        return state
    return _transition(
        state,
        status="waiting_for_human",
        current_step="ambiguous_effect",
        error_type="AmbiguousEffect",
        error_message=(
            "A provider call may have completed before its result was saved. "
            "Retry only after confirming the original process stopped."
        ),
    )


def _starting_state(state: RunState) -> RunState:
    if state.status in {"completed", "cancelled"}:
        return state
    if state.cancel_requested:
        return _cancelled_state(state)
    return _transition(
        state,
        status="running",
        current_step="starting",
        error_type=None,
        error_message=None,
    )


def _failure_state(state: RunState, error: Exception) -> RunState:
    if state.cancel_requested or state.status == "cancelled":
        return _cancelled_state(state)
    return _transition(
        state,
        status="failed",
        error_type=type(error).__name__,
        error_message=str(error),
    )


def _completion_state(state: RunState, result: ResearchResult) -> RunState:
    if state.cancel_requested or state.status == "cancelled":
        return _cancelled_state(state)
    return _transition(
        state,
        status="completed",
        current_step="completed",
        result=result,
        error_type=None,
        error_message=None,
    )


def _clarification_state(
    state: RunState,
    clarification: ClarificationDecision,
) -> RunState:
    if state.cancel_requested or state.status == "cancelled":
        return _cancelled_state(state)
    return _transition(
        state,
        status="waiting_for_human",
        current_step="clarification",
        clarification=clarification,
        error_type=None,
        error_message=None,
    )


def _scoped_state(state: RunState, brief: ResearchBrief) -> RunState:
    if state.cancel_requested or state.status == "cancelled":
        return _cancelled_state(state)
    return _transition(
        state,
        request=replace(state.request, brief=brief),
        current_step="scoped",
        clarification=None,
        error_type=None,
        error_message=None,
    )


def _plan_control_record(
    state: RunState,
    *,
    kind: PlanControlKind,
    base_plan_hash: str,
    plan: ResearchPlan,
) -> PlanControlRecord:
    input_hash = plan_control_input_hash(
        run_id=state.run_id,
        kind=kind,
        base_plan_hash=base_plan_hash,
        plan=plan,
    )
    plan_hash = (
        research_plan_version_hash(plan, base_plan_hash)
        if kind == "edit"
        else base_plan_hash
    )
    return PlanControlRecord(
        control_id=f"plan.{kind}:{input_hash}",
        kind=kind,
        input_hash=input_hash,
        base_plan_hash=base_plan_hash,
        plan_hash=plan_hash,
        plan=plan,
        created_at=_timestamp(),
    )


def _validate_edited_plan(
    plan: ResearchPlan,
    max_questions: int,
) -> None:
    if not isinstance(plan.objective, str) or not plan.objective.strip():
        raise ValueError("edited plan objective must not be empty")
    if plan.revision != 0:
        raise ValueError("edited initial plan revision must remain 0")
    if not plan.questions:
        raise ValueError("edited plan must contain at least one question")
    if len(plan.questions) > max_questions:
        raise ValueError("edited plan exceeds the research-step budget")
    identifiers = [
        question.id.strip() if isinstance(question.id, str) else ""
        for question in plan.questions
    ]
    if any(not identifier for identifier in identifiers):
        raise ValueError("edited plan question IDs must not be empty")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("edited plan question IDs must be unique")
    normalized_questions = [
        " ".join(question.question.split()).casefold()
        if isinstance(question.question, str)
        else ""
        for question in plan.questions
    ]
    if any(not question for question in normalized_questions):
        raise ValueError("edited plan questions must not be empty")
    if len(set(normalized_questions)) != len(normalized_questions):
        raise ValueError("edited plan questions must be unique")
    if any(
        not isinstance(question.priority, int) or question.priority < 1
        for question in plan.questions
    ):
        raise ValueError("edited plan question priorities must be at least 1")


def _completed_scoping_usage(effects: tuple[EffectRecord, ...]) -> TokenUsage:
    seen_effect_ids: set[str] = set()
    usages: list[TokenUsage] = []
    for effect in effects:
        if (
            effect.effect_id in seen_effect_ids
            or effect.kind != "scope.resolve"
            or effect.status != "completed"
            or effect.result_type != "ScopingRun"
        ):
            continue
        seen_effect_ids.add(effect.effect_id)
        decoded = _decode_result(effect.result_type, effect.result)
        if isinstance(decoded, ScopingRun):
            usages.append(decoded.usage)
    return TokenUsage(
        input_tokens=sum(usage.input_tokens for usage in usages),
        output_tokens=sum(usage.output_tokens for usage in usages),
        total_tokens=sum(usage.total_tokens for usage in usages),
    )


def _completed_clarification_count(effects: tuple[EffectRecord, ...]) -> int:
    count = 0
    for effect in effects:
        if (
            effect.kind != "scope.resolve"
            or effect.status != "completed"
            or effect.result_type != "ScopingRun"
        ):
            continue
        decoded = _decode_result(effect.result_type, effect.result)
        if isinstance(decoded, ScopingRun) and decoded.clarification.needs_clarification:
            count += 1
    return count


def _add_usage(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    return TokenUsage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
    )

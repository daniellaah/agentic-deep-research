"""Structured data contracts shared by the research agents."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

AgentName = Literal["research_agent", "writer_agent", "editor_agent"]
ConversationRole = Literal["user", "assistant"]
WorkerAction = Literal["research", "write", "edit", "revise"]
SupervisorAction = Literal["research", "write", "edit", "revise", "finish"]
ArtifactStatus = Literal["completed", "failed"]
SupervisorStatus = Literal["running", "completed", "failed"]


class ConversationMessage(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    role: ConversationRole = Field(description="The author of the conversation message.")
    content: str = Field(
        min_length=1,
        description="The text content of the conversation message.",
    )


class ClarificationDecision(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    needs_clarification: bool = Field(
        description=("Whether missing information must be clarified before research begins.")
    )
    reason: str = Field(
        min_length=1,
        description=("A brief operational explanation for the clarification decision."),
    )
    question: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "A single clarification question for the user, or null when no "
            "clarification is required."
        ),
    )

    @model_validator(mode="after")
    def validate_question_consistency(self) -> Self:
        if self.needs_clarification and self.question is None:
            raise ValueError(
                "A clarification question is required when needs_clarification is true."
            )

        if not self.needs_clarification and self.question is not None:
            raise ValueError(
                "The clarification question must be null when needs_clarification is false."
            )

        return self


class ResearchBrief(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    research_question: str = Field(
        min_length=1,
        description=("A single normalized question that the research must answer."),
    )
    objective: str = Field(
        min_length=1,
        description=("The purpose of the research and the value its result should provide."),
    )
    scope_inclusions: list[str] = Field(
        min_length=1,
        description=("Topics, dimensions, and evidence that must be included."),
    )
    scope_exclusions: list[str] = Field(
        description=("Topics and boundaries that should be excluded from the research."),
    )
    constraints: list[str] = Field(
        description=(
            "Requirements such as time range, audience, geography, source types, length, or format."
        ),
    )
    deliverable: str = Field(
        min_length=1,
        description=("The expected final output and its presentation format."),
    )
    success_criteria: list[str] = Field(
        min_length=1,
        description=("Observable conditions that indicate the research is complete."),
    )


class ScopeOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clarification: ClarificationDecision = Field(
        description="The clarification decision produced for the conversation."
    )
    research_brief: ResearchBrief | None = Field(
        default=None,
        description=("The completed research brief, or null when clarification is required."),
    )

    @model_validator(mode="after")
    def validate_outcome_consistency(self) -> Self:
        if self.clarification.needs_clarification and self.research_brief is not None:
            raise ValueError("Research brief must be null when clarification is required.")

        if not self.clarification.needs_clarification and self.research_brief is None:
            raise ValueError("Research brief is required when clarification is not needed.")

        return self


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(description="A unique one-based identifier for this step.")
    agent: AgentName = Field(description="The specialist agent that must execute this step.")
    task: str = Field(
        description="A single atomic and executable instruction for the assigned agent."
    )


class TaskPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(description="The overall goal of the task.")
    steps: list[PlanStep] = Field(
        description="An ordered list of steps required to achieve the goal."
    )


class SupervisorDecision(BaseModel):
    """The next action selected by the adaptive supervisor."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    action: SupervisorAction = Field(
        description="The next workflow action, or finish when the report is complete."
    )
    task: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "A concrete instruction for the selected worker, or null when the action is finish."
        ),
    )
    reason: str = Field(
        min_length=1,
        description="A concise explanation grounded in the current workflow state.",
    )

    @model_validator(mode="after")
    def validate_task_consistency(self) -> Self:
        if self.action == "finish" and self.task is not None:
            raise ValueError("Task must be null when the supervisor action is finish.")

        if self.action != "finish" and self.task is None:
            raise ValueError("A task is required for every worker action.")

        return self


class AgentArtifact(BaseModel):
    """A durable result produced by one worker-agent invocation."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: int = Field(ge=1, description="A unique one-based artifact identifier.")
    action: WorkerAction = Field(description="The workflow action that produced the artifact.")
    agent: AgentName = Field(description="The worker agent that produced the artifact.")
    task: str = Field(min_length=1, description="The instruction executed by the worker agent.")
    status: ArtifactStatus = Field(description="Whether the worker invocation completed or failed.")
    content: str | None = Field(
        default=None,
        min_length=1,
        description="The worker output when the invocation completed, otherwise null.",
    )
    error: str | None = Field(
        default=None,
        min_length=1,
        description="The error message when the invocation failed, otherwise null.",
    )

    @model_validator(mode="after")
    def validate_result_consistency(self) -> Self:
        expected_agent = {
            "research": "research_agent",
            "write": "writer_agent",
            "edit": "editor_agent",
            "revise": "writer_agent",
        }[self.action]

        if self.agent != expected_agent:
            raise ValueError(f"Action {self.action!r} must be executed by {expected_agent!r}.")

        if self.status == "completed":
            if self.content is None:
                raise ValueError("Completed artifacts require content.")
            if self.error is not None:
                raise ValueError("Completed artifacts cannot contain an error.")

        if self.status == "failed":
            if self.content is not None:
                raise ValueError("Failed artifacts cannot contain content.")
            if self.error is None:
                raise ValueError("Failed artifacts require an error message.")

        return self


class ObservationAssessment(BaseModel):
    """A structured model assessment of a completed worker artifact."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    summary: str = Field(min_length=1, description="A concise summary of the artifact.")
    key_findings: list[str] = Field(description="Important findings or changes in the artifact.")
    evidence_gaps: list[str] = Field(description="Missing evidence that may require more research.")
    quality_issues: list[str] = Field(
        description="Quality problems that may require writing or editing work."
    )


class SupervisorObservation(BaseModel):
    """Compact feedback returned to the supervisor after a worker action."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    artifact_id: int = Field(ge=1, description="The artifact evaluated by this observation.")
    action: WorkerAction = Field(description="The action that produced the evaluated artifact.")
    agent: AgentName = Field(description="The agent that produced the evaluated artifact.")
    success: bool = Field(description="Whether the worker invocation completed successfully.")
    summary: str = Field(min_length=1, description="A concise result or failure summary.")
    key_findings: list[str] = Field(description="Important findings or changes observed.")
    evidence_gaps: list[str] = Field(description="Evidence gaps that remain after the action.")
    quality_issues: list[str] = Field(description="Quality issues that remain after the action.")
    source_urls: list[str] = Field(description="Source URLs extracted from the artifact.")
    error: str | None = Field(
        default=None,
        min_length=1,
        description="The worker error when success is false, otherwise null.",
    )

    @model_validator(mode="after")
    def validate_error_consistency(self) -> Self:
        expected_agent = {
            "research": "research_agent",
            "write": "writer_agent",
            "edit": "editor_agent",
            "revise": "writer_agent",
        }[self.action]

        if self.agent != expected_agent:
            raise ValueError(f"Action {self.action!r} must be associated with {expected_agent!r}.")

        if self.success and self.error is not None:
            raise ValueError("Successful observations cannot contain an error.")

        if not self.success and self.error is None:
            raise ValueError("Failed observations require an error message.")

        return self


class SupervisorState(BaseModel):
    """The observable state used by the adaptive supervisor loop."""

    model_config = ConfigDict(extra="forbid")

    research_brief: ResearchBrief = Field(
        description="The scoped research contract that guides every decision."
    )
    artifacts: list[AgentArtifact] = Field(
        default_factory=list,
        description="The ordered worker artifacts produced during this run.",
    )
    observations: list[SupervisorObservation] = Field(
        default_factory=list,
        description="Compact assessments of the worker artifacts.",
    )
    decision_errors: list[str] = Field(
        default_factory=list,
        description="Invalid supervisor decisions rejected by deterministic guards.",
    )
    iteration: int = Field(default=0, ge=0, description="The number of worker actions attempted.")
    status: SupervisorStatus = Field(default="running", description="The current run status.")
    final_report: str | None = Field(
        default=None,
        min_length=1,
        description="The validated final report after the supervisor finishes.",
    )

    @model_validator(mode="after")
    def validate_completion_consistency(self) -> Self:
        if self.status == "completed" and self.final_report is None:
            raise ValueError("Completed supervisor state requires a final report.")

        if self.status == "running" and self.final_report is not None:
            raise ValueError("Running supervisor state cannot contain a final report.")

        return self

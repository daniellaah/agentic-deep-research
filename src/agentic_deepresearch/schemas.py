"""Structured data contracts shared by the research agents."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

AgentName = Literal["research_agent", "writer_agent", "editor_agent"]
ConversationRole = Literal["user", "assistant"]


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

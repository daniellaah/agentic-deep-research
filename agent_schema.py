from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AgentName = Literal["research_agent", "writer_agent", "editor_agent"]


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(
        description="A unique one-based identifier for this step."
    )
    agent: AgentName = Field(
        description="The specialist agent that must execute this step."
    )
    task: str = Field(
        description="A single atomic and executable instruction for the assigned agent."
    )


class TaskPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(
        description="The overall goal of the task."
    )
    steps: list[PlanStep] = Field(
        description="An ordered list of steps required to achieve the goal."
    )

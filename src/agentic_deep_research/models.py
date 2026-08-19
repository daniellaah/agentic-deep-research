"""Domain models for a bounded web-research run."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ResearchBudget:
    """Hard limits passed to the model runtime."""

    max_tool_calls: int = 8
    max_output_tokens: int = 4_000

    def __post_init__(self) -> None:
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")


@dataclass(frozen=True)
class ResearchRequest:
    """A topic and the constraints for researching it."""

    topic: str
    language: str = "the same language as the request"
    budget: ResearchBudget = field(default_factory=ResearchBudget)

    def __post_init__(self) -> None:
        if not self.topic.strip():
            raise ValueError("research topic must not be empty")
        if not self.language.strip():
            raise ValueError("report language must not be empty")


@dataclass(frozen=True)
class Source:
    """A web source consulted during research."""

    title: str
    url: str


@dataclass(frozen=True)
class Citation:
    """A source annotation attached to a span of the generated report."""

    source: Source
    start_index: int
    end_index: int


@dataclass(frozen=True)
class ResearchStep:
    """An observable web action performed during research."""

    action: str
    detail: str


@dataclass(frozen=True)
class TokenUsage:
    """Token counts reported by the model provider."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class AgentRun:
    """Provider-neutral output returned by an agent runner."""

    report: str
    sources: tuple[Source, ...] = ()
    citations: tuple[Citation, ...] = ()
    trace: tuple[ResearchStep, ...] = ()
    status: str = "completed"
    stop_reason: str = "completed"
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass(frozen=True)
class ResearchResult:
    """Public result of a completed or bounded research workflow."""

    topic: str
    report: str
    sources: tuple[Source, ...]
    citations: tuple[Citation, ...]
    trace: tuple[ResearchStep, ...]
    status: str
    stop_reason: str
    usage: TokenUsage

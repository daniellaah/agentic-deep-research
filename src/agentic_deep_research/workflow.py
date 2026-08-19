"""Static deep research workflow."""


def run_research(topic: str) -> None:
    """Start a research run for a non-empty topic."""
    if not topic.strip():
        raise ValueError("research topic must not be empty")

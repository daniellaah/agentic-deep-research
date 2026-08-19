"""Public interface for the Agentic Deep Research package."""

from .models import ResearchResult
from .workflow import run_research

__all__ = ["ResearchResult", "run_research"]

import pytest

from agentic_deepresearch import run_research


def test_run_research_rejects_blank_topic() -> None:
    with pytest.raises(ValueError, match="research topic must not be empty"):
        run_research("   ")

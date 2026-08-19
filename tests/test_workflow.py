import pytest

from agentic_deep_research import run_research


class FakeAgentRunner:
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = iter(outputs)
        self.calls: list[tuple[str, str]] = []

    def run(self, *, instructions: str, task: str) -> str:
        self.calls.append((instructions, task))
        return next(self._outputs)


def test_run_research_rejects_blank_topic() -> None:
    with pytest.raises(ValueError, match="research topic must not be empty"):
        run_research("   ", runner=FakeAgentRunner([]))


def test_run_research_completes_static_report_workflow() -> None:
    runner = FakeAgentRunner(
        [
            "Plan: compare reliability methods.",
            "Notes: evaluation and observability improve reliability.",
            "Draft: reliable agents require evaluation and observability.",
            "Feedback: explain the evaluation strategy more clearly.",
            "Final: reliable agents require measurable evaluation and observability.",
        ]
    )

    result = run_research(
        "Reliable LLM agents",
        runner=runner,
    )

    assert len(runner.calls) == 5

    assert result.topic == "Reliable LLM agents"
    assert result.plan == "Plan: compare reliability methods."
    assert result.research_notes == (
        "Notes: evaluation and observability improve reliability."
    )
    assert result.draft_report == (
        "Draft: reliable agents require evaluation and observability."
    )
    assert result.editorial_feedback == (
        "Feedback: explain the evaluation strategy more clearly."
    )
    assert result.final_report == (
        "Final: reliable agents require measurable evaluation and observability."
    )

    planner_task = runner.calls[0][1]
    research_task = runner.calls[1][1]
    drafting_task = runner.calls[2][1]
    editing_task = runner.calls[3][1]
    revision_task = runner.calls[4][1]

    assert planner_task == result.topic
    assert result.plan in research_task
    assert result.plan in drafting_task
    assert result.research_notes in drafting_task
    assert result.draft_report in editing_task
    assert result.draft_report in revision_task
    assert result.editorial_feedback in revision_task

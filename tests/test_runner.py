from types import SimpleNamespace
from unittest.mock import Mock

from agentic_deep_research.runner import OpenAIAgentRunner


def test_run_returns_generated_text() -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text="Generated research plan."
    )
    runner = OpenAIAgentRunner(client=client, model="test-model")

    result = runner.run(
        instructions="You are a research planner.",
        task="Research AI agents.",
    )

    assert result == "Generated research plan."

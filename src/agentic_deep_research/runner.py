"""Agent execution through the OpenAI Responses API."""

from openai import OpenAI


class OpenAIAgentRunner:
    """Execute agent tasks through the OpenAI Responses API."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    def run(self, *, instructions: str, task: str) -> str:
        """Execute an agent task and return its generated text."""
        response = self._client.responses.create(
            model=self._model,
            instructions=instructions,
            input=task,
        )
        return response.output_text

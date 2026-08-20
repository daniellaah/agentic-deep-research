import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agentic_deep_research.benchmarks import (
    BenchmarkCase,
    ExactMatchJudge,
    Judgment,
    OpenAIAnswerJudge,
    load_benchmark_cases,
)


def test_load_benchmark_cases_maps_official_dataset_fields(tmp_path) -> None:
    frames_path = tmp_path / "frames.jsonl"
    frames_path.write_text(
        json.dumps(
            {
                "Unnamed: 0": 7,
                "Prompt": "A multi-hop question?",
                "Answer": "The answer",
                "reasoning_types": "Multiple constraints",
                "wiki_links": "['https://en.wikipedia.org/wiki/Answer']",
            }
        )
        + "\n"
    )
    browsecomp_path = tmp_path / "browsecomp-plus.jsonl"
    browsecomp_path.write_text(
        json.dumps(
            {
                "query_id": "bc-1",
                "query": "A hidden web fact?",
                "answer": "The fact",
            }
        )
        + "\n"
    )

    frames = load_benchmark_cases("frames", frames_path)
    browsecomp = load_benchmark_cases("browsecomp-plus", browsecomp_path)

    assert frames == (
        BenchmarkCase(
            id="7",
            question="A multi-hop question?",
            answer="The answer",
            metadata={
                "reasoning_types": "Multiple constraints",
                "wiki_links": "['https://en.wikipedia.org/wiki/Answer']",
            },
        ),
    )
    assert browsecomp == (
        BenchmarkCase(id="bc-1", question="A hidden web fact?", answer="The fact"),
    )


def test_exact_match_judge_returns_immutable_structured_judgment() -> None:
    judgment = ExactMatchJudge().judge(
        question="Who?",
        reference="The Jane Ballou!",
        prediction="jane ballou",
    )

    assert judgment == Judgment(
        correct=True,
        score=1.0,
        reason="Normalized prediction matches the reference answer.",
        grader="exact-match-v1",
    )
    with pytest.raises(AttributeError):
        judgment.score = 0.0  # type: ignore[misc]


def test_openai_answer_judge_returns_structured_judgment() -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text=(
            '{"correct": true, "score": 0.95, '
            '"reason": "The location matches the reference."}'
        )
    )
    judge = OpenAIAnswerJudge(client=client, model="judge-model")

    judgment = judge.judge(
        question="Where is the Eiffel Tower?",
        reference="Paris",
        prediction="The Eiffel Tower is in Paris, France.",
    )

    assert judgment == Judgment(
        correct=True,
        score=0.95,
        reason="The location matches the reference.",
        grader="openai:judge-model:answer-judge-v1",
    )
    call = client.responses.create.call_args.kwargs
    assert call["model"] == "judge-model"
    assert "Where is the Eiffel Tower?" in call["input"]
    assert "Paris" in call["input"]
    assert "The Eiffel Tower is in Paris, France." in call["input"]
    assert call["text"]["format"]["type"] == "json_schema"
    schema = call["text"]["format"]["schema"]
    assert schema["required"] == ["correct", "score", "reason"]
    assert schema["additionalProperties"] is False


def test_openai_answer_judge_rejects_incomplete_response() -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text='{"correct": true, "score": 1, "reason": "Looks complete."}',
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
    )

    with pytest.raises(RuntimeError, match="not completed: max_output_tokens"):
        OpenAIAnswerJudge(client=client, model="judge-model").judge(
            question="Question?",
            reference="Answer",
            prediction="Answer",
        )


@pytest.mark.parametrize(
    "output_text, error, message",
    [
        ('{"correct": 1, "score": 1, "reason": "ok"}', TypeError, "boolean"),
        (
            '{"correct": true, "score": 1.1, "reason": "ok"}',
            ValueError,
            "between 0 and 1",
        ),
        (
            '{"correct": true, "score": 1, "reason": "   "}',
            ValueError,
            "must not be empty",
        ),
        (
            '{"correct": true, "score": 1, "reason": "ok", "extra": 1}',
            ValueError,
            "only correct, score, and reason",
        ),
    ],
)
def test_openai_answer_judge_strictly_validates_output(
    output_text: str,
    error: type[Exception],
    message: str,
) -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(output_text=output_text)
    judge = OpenAIAnswerJudge(client=client, model="judge-model")

    with pytest.raises(error, match=message):
        judge.judge(question="Question?", reference="Answer", prediction="Prediction")

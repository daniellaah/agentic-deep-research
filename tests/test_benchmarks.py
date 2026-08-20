import json
from types import SimpleNamespace
from unittest.mock import Mock

from agentic_deep_research import CitationCheck, ResearchResult, Source, TokenUsage
from agentic_deep_research.benchmarks import (
    BenchmarkCase,
    ExactMatchJudge,
    OpenAIAnswerJudge,
    load_benchmark_cases,
    run_benchmark,
)


def _result(topic: str, report: str) -> ResearchResult:
    source = Source("Example", "https://example.com")
    return ResearchResult(
        topic=topic,
        report=report,
        raw_report=report,
        sources=(),
        citations=(),
        evidence=(),
        trace=(),
        status="completed",
        stop_reason="completed",
        usage=TokenUsage(total_tokens=10),
        citation_checks=(
            CitationCheck(
                claim="A supported claim.",
                source=source,
                status="supported",
                reason="The source supports it.",
                claim_id="C1",
            ),
        ),
        revision_count=1,
    )


def test_run_benchmark_scores_cases_and_resumes_from_jsonl(tmp_path) -> None:
    cases = (
        BenchmarkCase(id="one", question="First question?", answer="Jane Ballou"),
        BenchmarkCase(id="two", question="Second question?", answer="Paris"),
    )
    reports = iter(
        [
            _result("First question?", "Jane Ballou"),
            _result("Second question?", "London"),
        ]
    )
    calls: list[str] = []

    def research(question: str) -> ResearchResult:
        calls.append(question)
        return next(reports)

    output_path = tmp_path / "baseline.jsonl"
    summary = run_benchmark(
        benchmark="test",
        cases=cases,
        research=research,
        judge=ExactMatchJudge(),
        output_path=output_path,
    )

    assert summary.total == 2
    assert summary.correct == 1
    assert summary.accuracy == 0.5
    assert summary.completed == 2
    assert summary.average_latency_seconds is not None
    assert summary.average_latency_seconds >= 0
    assert calls == ["First question?", "Second question?"]
    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [record["case_id"] for record in records] == ["one", "two"]
    assert [record["correct"] for record in records] == [True, False]
    assert records[0]["revision_count"] == 1
    assert records[0]["citation_checks"][0]["status"] == "supported"

    calls.clear()
    resumed = run_benchmark(
        benchmark="test",
        cases=cases,
        research=research,
        judge=ExactMatchJudge(),
        output_path=output_path,
    )

    assert resumed == summary
    assert calls == []


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


def test_openai_answer_judge_returns_structured_correctness() -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(output_text='{"correct": true}')
    judge = OpenAIAnswerJudge(client=client, model="judge-model")

    correct = judge.judge(
        question="Where is the Eiffel Tower?",
        reference="Paris",
        prediction="The Eiffel Tower is in Paris, France.",
    )

    assert correct is True
    call = client.responses.create.call_args.kwargs
    assert call["model"] == "judge-model"
    assert "Where is the Eiffel Tower?" in call["input"]
    assert "Paris" in call["input"]
    assert "The Eiffel Tower is in Paris, France." in call["input"]
    assert call["text"]["format"]["type"] == "json_schema"


def test_run_benchmark_preserves_prediction_when_the_judge_fails(tmp_path) -> None:
    class FailingJudge:
        def judge(self, *, question: str, reference: str, prediction: str) -> bool:
            del question, reference, prediction
            raise ValueError("invalid judge output")

    output_path = tmp_path / "judge-error.jsonl"
    summary = run_benchmark(
        benchmark="test",
        cases=(BenchmarkCase(id="one", question="Question?", answer="Answer"),),
        research=lambda question: _result(question, "A useful prediction."),
        judge=FailingJudge(),
        output_path=output_path,
    )

    record = json.loads(output_path.read_text())
    assert record["prediction"] == "A useful prediction."
    assert record["status"] == "completed"
    assert record["judge_status"] == "error"
    assert record["judge_error"] == "ValueError"
    assert summary.completed == 1
    assert summary.correct == 0

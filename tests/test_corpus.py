import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agentic_deep_research import ResearchBudget
from agentic_deep_research.corpus import (
    CorpusDocument,
    InMemoryCorpusBackend,
    JsonlCorpusBackend,
    OpenAIFixedCorpusRunner,
)


def test_in_memory_corpus_has_unique_ids_and_order_independent_fingerprint() -> None:
    first = CorpusDocument(
        id="paper-a",
        title="Evaluation",
        text="Evaluation improves research-agent reliability.",
    )
    second = CorpusDocument(
        id="paper-b",
        title="Observability",
        text="Traces make agent failures observable.",
    )

    forward = InMemoryCorpusBackend([first, second])
    reverse = InMemoryCorpusBackend([second, first])

    assert forward.corpus_sha256 == reverse.corpus_sha256
    assert len(forward.corpus_sha256) == 64
    assert (
        InMemoryCorpusBackend(
            [
                first,
                CorpusDocument(
                    id="paper-b",
                    title="Observability",
                    text="Changed corpus content.",
                ),
            ]
        ).corpus_sha256
        != forward.corpus_sha256
    )

    with pytest.raises(ValueError, match="duplicate corpus document id: paper-a"):
        InMemoryCorpusBackend([first, first])


def test_search_is_deterministic_and_read_rejects_unknown_ids() -> None:
    backend = InMemoryCorpusBackend(
        [
            CorpusDocument(
                id="b",
                title="Agent reliability",
                text="Evaluation supports reliable systems.",
            ),
            CorpusDocument(
                id="a",
                title="Agent evaluation methods",
                text="Agent evaluation improves reliability and evaluation quality.",
            ),
            CorpusDocument(
                id="c",
                title="Unrelated document",
                text="This document discusses gardening.",
            ),
        ]
    )

    hits = backend.search("agent evaluation reliability", limit=5)

    assert [hit.document_id for hit in hits] == ["a", "b"]
    assert hits[0].score > hits[1].score
    assert "evaluation" in hits[0].snippet.casefold()
    assert backend.search("missing-token", limit=5) == ()
    assert backend.search("agent", limit=0) == ()
    assert backend.read("a").title == "Agent evaluation methods"
    with pytest.raises(KeyError, match="unknown corpus document id: missing"):
        backend.read("missing")


def test_jsonl_backend_loads_the_same_canonical_corpus(tmp_path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "doc-2",
                        "title": "Second",
                        "text": "Second document text.",
                    }
                ),
                json.dumps(
                    {
                        "id": "doc-1",
                        "title": "First",
                        "text": "First document text.",
                        "url": "https://example.test/first",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    loaded = JsonlCorpusBackend(corpus_path)
    expected = InMemoryCorpusBackend(
        [
            CorpusDocument(
                id="doc-1",
                title="First",
                text="First document text.",
                url="https://example.test/first",
            ),
            CorpusDocument(
                id="doc-2",
                title="Second",
                text="Second document text.",
            ),
        ]
    )

    assert loaded.corpus_sha256 == expected.corpus_sha256
    assert loaded.read("doc-1").url == "https://example.test/first"


def test_jsonl_backend_reports_the_invalid_line(tmp_path) -> None:
    corpus_path = tmp_path / "invalid.jsonl"
    corpus_path.write_text(
        '{"id":"valid","title":"Valid","text":"Text"}\n{"id": 2}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"invalid\.jsonl:2"):
        JsonlCorpusBackend(corpus_path)


def test_fixed_corpus_runner_uses_structured_outputs_without_web_search() -> None:
    client = Mock()
    report = (
        "Evaluation improves reliability [D1]. "
        "Traces expose failures [D2]. "
        "Unknown [D9], invalid [D0], and malformed [Dx] markers are ignored."
    )
    client.responses.create.return_value = SimpleNamespace(
        output_text=json.dumps({"report": report}),
        status="completed",
        incomplete_details=None,
        usage=SimpleNamespace(input_tokens=120, output_tokens=40, total_tokens=160),
    )
    backend = InMemoryCorpusBackend(
        [
            CorpusDocument(
                id="eval",
                title="Evaluation exposes reliable agent failures",
                text="Evaluation improves reliable research-agent behavior.",
                url="https://example.test/evaluation",
            ),
            CorpusDocument(
                id="trace",
                title="Reliable agent traces",
                text="Evaluation traces expose agent failures.",
            ),
        ]
    )
    runner = OpenAIFixedCorpusRunner(
        client=client,
        model="test-model",
        backend=backend,
    )
    budget = ResearchBudget(
        max_tool_calls=3,
        max_output_tokens=500,
        max_context_chars=1_000,
    )

    result = runner.run(
        instructions="Answer the assigned question.",
        task=(
            "Research topic:\nReliable agents\n\n"
            "Research question:\nHow does evaluation expose reliable agent failures?\n\n"
            "Return a concise answer."
        ),
        budget=budget,
    )

    call = client.responses.create.call_args.kwargs
    assert call["model"] == "test-model"
    assert call["max_output_tokens"] == 500
    assert call["text"]["format"]["type"] == "json_schema"
    assert "tools" not in call
    assert "tool_choice" not in call
    assert "web_search" not in json.dumps(call)
    input_data = json.loads(call["input"])
    assert input_data["research_question"] == (
        "How does evaluation expose reliable agent failures?"
    )
    assert input_data["fixed_corpus_sha256"] == backend.corpus_sha256
    assert [item["marker"] for item in input_data["documents"]] == ["[D1]", "[D2]"]

    assert result.report == report
    assert [source.id for source in result.sources] == ["eval", "trace"]
    assert result.sources[0].url == "https://example.test/evaluation"
    assert result.sources[1].url.startswith(f"corpus://{backend.corpus_sha256}/")
    assert [citation.source.id for citation in result.citations] == ["eval", "trace"]
    assert [report[item.start_index : item.end_index] for item in result.citations] == [
        "[D1]",
        "[D2]",
    ]
    assert [(step.action, step.detail) for step in result.trace] == [
        ("search", "How does evaluation expose reliable agent failures?"),
        ("read", "eval"),
        ("read", "trace"),
    ]
    assert result.usage.total_tokens == 160


def test_fixed_corpus_runner_respects_tool_and_context_budgets() -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text='{"report":"Bounded answer [D1]. Unknown [D2]."}',
        status="completed",
        incomplete_details=None,
        usage=None,
    )
    backend = InMemoryCorpusBackend(
        [
            CorpusDocument(
                id="a",
                title="Agents",
                text="agents " * 500,
            ),
            CorpusDocument(
                id="b",
                title="More agents",
                text="agents " * 500,
            ),
        ]
    )
    runner = OpenAIFixedCorpusRunner(client, "test-model", backend)
    budget = ResearchBudget(
        max_tool_calls=2,
        max_output_tokens=100,
        max_context_chars=180,
    )

    result = runner.run(
        instructions="Research.",
        task="Research question:\nHow do agents work?",
        budget=budget,
    )

    input_data = json.loads(client.responses.create.call_args.kwargs["input"])
    serialized_documents = json.dumps(
        input_data["documents"],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert len(serialized_documents) <= budget.max_context_chars
    assert len(input_data["documents"]) == 1
    assert len(result.trace) == budget.max_tool_calls
    assert [citation.source.id for citation in result.citations] == ["a"]


def test_fixed_corpus_runner_rejects_invalid_structured_output() -> None:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text="not JSON",
        status="completed",
        incomplete_details=None,
        usage=None,
    )
    runner = OpenAIFixedCorpusRunner(
        client,
        "test-model",
        InMemoryCorpusBackend(
            [CorpusDocument(id="doc", title="Document", text="research text")]
        ),
    )

    with pytest.raises(ValueError, match="invalid structured output"):
        runner.run(
            instructions="Research.",
            task="research",
            budget=ResearchBudget(max_tool_calls=2, max_output_tokens=100),
        )

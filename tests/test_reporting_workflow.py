from agentic_deep_research import (
    AgentRun,
    Citation,
    CitationCheck,
    CitationVerification,
    ReportCritique,
    ReportDraft,
    ResearchBudget,
    ResearchRequest,
    Source,
    TokenUsage,
    run_research,
)


class SingleRunRunner:
    def __init__(self, run: AgentRun) -> None:
        self._run = run

    def run(
        self,
        *,
        instructions: str,
        task: str,
        budget: ResearchBudget,
    ) -> AgentRun:
        del instructions, task, budget
        return self._run


class WriterOnlyAgent:
    def write(self, **_: object) -> ReportDraft:
        return ReportDraft(
            report="# Reliable agents\n\nEvaluation improves reliability [S1].",
            usage=TokenUsage(total_tokens=7),
        )

    def critique(self, **_: object) -> ReportCritique:
        return ReportCritique(usage=TokenUsage(total_tokens=3))

    def verify(self, **kwargs: object) -> CitationVerification:
        claims = kwargs["claims"]
        return CitationVerification(
            checks=tuple(
                CitationCheck(
                    claim=claim.claim,
                    source=claim.source,
                    status="supported",
                    reason="The source supports the claim.",
                )
                for claim in claims
            ),
            usage=TokenUsage(total_tokens=4),
        )


class GapCriticAgent(WriterOnlyAgent):
    def critique(self, **_: object) -> ReportCritique:
        return ReportCritique(
            coverage_gaps=("Explain how reliability is measured.",),
            unsupported_claims=("The method always works.",),
            clarity_issues=("Define the evaluation metric.",),
            revision_instructions=("Add measurable evaluation criteria.",),
            needs_more_research=True,
            usage=TokenUsage(total_tokens=5),
        )

    def revise(self, **_: object) -> ReportDraft:
        return ReportDraft(
            report=(
                "# Reliable agents\n\nEvaluation improves reliability [S1]. "
                "Measurable benchmarks confirm the result [S2]."
            ),
            usage=TokenUsage(total_tokens=8),
        )


class UnsupportedCitationAgent(WriterOnlyAgent):
    def verify(self, **kwargs: object) -> CitationVerification:
        claim = kwargs["claims"][0]
        return CitationVerification(
            checks=(
                CitationCheck(
                    claim=claim.claim,
                    source=claim.source,
                    status="unsupported",
                    reason="The source discusses evaluation but not reliability gains.",
                ),
            ),
            usage=TokenUsage(total_tokens=4),
        )


class GapRevisionAgent(WriterOnlyAgent):
    def __init__(self) -> None:
        self.critique_calls = 0

    def critique(self, **_: object) -> ReportCritique:
        self.critique_calls += 1
        if self.critique_calls == 1:
            return ReportCritique(
                coverage_gaps=("Find a measurable reliability benchmark.",),
                revision_instructions=("Add benchmark evidence.",),
                needs_more_research=True,
                usage=TokenUsage(total_tokens=5),
            )
        return ReportCritique(usage=TokenUsage(total_tokens=3))

    def revise(self, **_: object) -> ReportDraft:
        return ReportDraft(
            report=(
                "# Reliable agents\n\nEvaluation improves reliability [S1]. "
                "A benchmark measures the gain [S2]."
            ),
            usage=TokenUsage(total_tokens=8),
        )


class PersistentGapAgent(GapRevisionAgent):
    def critique(self, **_: object) -> ReportCritique:
        self.critique_calls += 1
        return ReportCritique(
            coverage_gaps=("A critical gap remains.",),
            needs_more_research=True,
            usage=TokenUsage(total_tokens=3),
        )


class MappingRunner:
    def __init__(self, runs: dict[str, AgentRun]) -> None:
        self._runs = runs

    def run(
        self,
        *,
        instructions: str,
        task: str,
        budget: ResearchBudget,
    ) -> AgentRun:
        del instructions, budget
        for question, run in self._runs.items():
            if f"Research question:\n{question}" in task:
                return run
        raise AssertionError(f"unexpected task: {task}")


def _cited_run() -> AgentRun:
    source = Source("Evaluation study", "https://example.com/evaluation")
    report = "Evaluation improves reliability. [1]"
    start = report.index("[1]")
    return AgentRun(
        report=report,
        sources=(source,),
        citations=(Citation(source, start, start + 3),),
        usage=TokenUsage(total_tokens=15),
    )


def _gap_run() -> AgentRun:
    source = Source("Reliability benchmark", "https://example.com/benchmark")
    report = "A benchmark measures reliability gains. [1]"
    start = report.index("[1]")
    return AgentRun(
        report=report,
        sources=(source,),
        citations=(Citation(source, start, start + 3),),
        usage=TokenUsage(total_tokens=15),
    )


def test_writer_synthesizes_findings_with_controlled_source_markers() -> None:
    result = run_research(
        ResearchRequest(topic="Reliable agents", min_sources=1),
        runner=SingleRunRunner(_cited_run()),
        report_agent=WriterOnlyAgent(),
    )

    assert result.raw_report == ("# Reliable agents\n\nEvaluation improves reliability [S1].")
    assert result.report == (
        "# Reliable agents\n\nEvaluation improves reliability "
        "[Evaluation study](https://example.com/evaluation)."
    )
    assert len(result.citations) == 1
    assert result.citations[0].source == result.sources[0]
    assert result.usage.total_tokens == 29
    assert result.artifacts[-1].kind == "report_draft"
    assert any(step.action == "report_written" for step in result.trace)


def test_critic_exposes_report_quality_gaps() -> None:
    result = run_research(
        ResearchRequest(
            topic="Reliable agents",
            min_sources=1,
            budget=ResearchBudget(max_revision_rounds=0),
        ),
        runner=SingleRunRunner(_cited_run()),
        report_agent=GapCriticAgent(),
    )

    assert result.critique is not None
    assert result.critique.coverage_gaps == ("Explain how reliability is measured.",)
    assert result.critique.unsupported_claims == ("The method always works.",)
    assert result.critique.needs_more_research is True
    assert result.status == "needs_review"
    assert result.stop_reason == "coverage_gaps"
    assert any(step.action == "report_critiqued" for step in result.trace)


def test_citation_verifier_rejects_an_unsupported_claim_source_pair() -> None:
    result = run_research(
        ResearchRequest(
            topic="Reliable agents",
            min_sources=1,
            budget=ResearchBudget(max_revision_rounds=0),
        ),
        runner=SingleRunRunner(_cited_run()),
        report_agent=UnsupportedCitationAgent(),
    )

    assert len(result.citation_checks) == 1
    assert result.citation_checks[0].status == "unsupported"
    assert result.citation_checks[0].source == result.sources[0]
    assert result.status == "needs_review"
    assert result.stop_reason == "unsupported_citations"
    assert any(step.action == "citations_verified" for step in result.trace)


def test_gap_search_adds_evidence_and_reviser_completes_the_report() -> None:
    agent = GapRevisionAgent()
    runner = MappingRunner(
        {
            "Reliable agents": _cited_run(),
            "Find a measurable reliability benchmark.": _gap_run(),
        }
    )

    result = run_research(
        ResearchRequest(
            topic="Reliable agents",
            min_sources=2,
            budget=ResearchBudget(
                max_tool_calls=4,
                max_research_steps=2,
                max_parallel_workers=1,
                max_revision_rounds=2,
            ),
        ),
        runner=runner,
        report_agent=agent,
    )

    assert result.revision_count == 1
    assert len(result.findings) == 2
    assert len(result.sources) == 2
    assert "https://example.com/benchmark" in result.report
    assert result.status == "completed"
    assert result.stop_reason == "completed"
    assert any(step.action == "gap_search_started" for step in result.trace)
    assert any(step.action == "report_revised" for step in result.trace)
    assert agent.critique_calls == 2


def test_revision_loop_stops_at_the_configured_maximum() -> None:
    agent = PersistentGapAgent()
    runner = MappingRunner(
        {
            "Reliable agents": _cited_run(),
            "A critical gap remains.": _gap_run(),
        }
    )

    result = run_research(
        ResearchRequest(
            topic="Reliable agents",
            min_sources=1,
            budget=ResearchBudget(
                max_tool_calls=4,
                max_research_steps=2,
                max_parallel_workers=1,
                max_revision_rounds=1,
            ),
        ),
        runner=runner,
        report_agent=agent,
    )

    assert result.revision_count == 1
    assert result.status == "needs_review"
    assert result.stop_reason == "max_revision_rounds"
    assert agent.critique_calls == 2

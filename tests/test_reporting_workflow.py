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


class EvidenceMarkerAgent(WriterOnlyAgent):
    def write(self, **_: object) -> ReportDraft:
        return ReportDraft(
            report="# Reliable agents\n\nEvaluation improves reliability [E1].",
            usage=TokenUsage(total_tokens=7),
        )


class MixedBoundAndForgedEvidenceMarkerAgent(WriterOnlyAgent):
    def write(self, **_: object) -> ReportDraft:
        return ReportDraft(
            report=(
                "# Reliable agents\n\nEvaluation improves reliability [E1]. "
                "A fabricated ledger entry makes another claim [E999]."
            ),
            usage=TokenUsage(total_tokens=7),
        )


class MisusedEvidenceMarkerAgent(WriterOnlyAgent):
    def write(self, **_: object) -> ReportDraft:
        return ReportDraft(
            report="# Reliable agents\n\nThe moon is made of cheese [E1].",
            usage=TokenUsage(total_tokens=7),
        )


class MalformedEvidenceMarkerAgent(WriterOnlyAgent):
    def write(self, **_: object) -> ReportDraft:
        return ReportDraft(
            report=(
                "# Reliable agents\n\nEvaluation improves reliability [E1]. "
                "Malformed markers [Ebogus] and [E01] are not catalog entries."
            ),
            usage=TokenUsage(total_tokens=7),
        )


class UnboundLegacySourceAgent(WriterOnlyAgent):
    def write(self, **_: object) -> ReportDraft:
        return ReportDraft(
            report="# Reliable agents\n\nAn uncited source makes this claim [S1].",
            usage=TokenUsage(total_tokens=7),
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


class WrongBindingCitationAgent(WriterOnlyAgent):
    def verify(self, **kwargs: object) -> CitationVerification:
        claim = kwargs["claims"][0]
        return CitationVerification(
            checks=(
                CitationCheck(
                    claim=claim.claim,
                    source=Source("Wrong source", "https://example.com/wrong"),
                    status="supported",
                    reason="This result must not be rebound by position.",
                    claim_id="C999",
                    evidence_id="ev_wrong",
                ),
            )
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
                "# Reliable agents\n\nEvaluation improves reliability [E1]. "
                "A benchmark measures reliability gains [E2]."
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


def _run_with_an_unbound_first_source() -> AgentRun:
    unbound_source = Source("Overview", "https://example.com/overview")
    cited_source = Source("Evaluation study", "https://example.com/evaluation")
    report = "Evaluation improves reliability. [1]"
    start = report.index("[1]")
    return AgentRun(
        report=report,
        sources=(unbound_source, cited_source),
        citations=(Citation(cited_source, start, start + 3),),
        usage=TokenUsage(total_tokens=15),
    )


def _run_with_a_canonical_source_alias() -> AgentRun:
    source = Source("Evaluation study", "https://example.com/evaluation")
    alias = Source(
        "Tracked evaluation link",
        "https://EXAMPLE.com/evaluation?utm_source=newsletter#result",
    )
    report = "Evaluation improves reliability. [1]"
    start = report.index("[1]")
    return AgentRun(
        report=report,
        sources=(source,),
        citations=(Citation(alias, start, start + 3),),
    )


def _run_with_a_forged_evidence_id() -> AgentRun:
    source = Source("Evaluation study", "https://example.com/evaluation")
    report = "Evaluation improves reliability. [1]"
    start = report.index("[1]")
    return AgentRun(
        report=report,
        sources=(source,),
        citations=(Citation(source, start, start + 3, evidence_id="ev_forged"),),
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
    assert result.ledger is not None
    assert result.citations[0].evidence_id == result.ledger.evidence[0].id
    assert result.citation_checks[0].evidence_id == result.citations[0].evidence_id
    assert result.citation_checks == result.ledger.checks
    assert result.evidence[0].verification_status == "supported"
    assert result.findings[0].evidence[0].verification_status == "supported"
    assert result.usage.total_tokens == 29
    assert result.artifacts[-1].kind == "report_draft"
    assert any(step.action == "report_written" for step in result.trace)


def test_writer_can_cite_evidence_ledger_markers() -> None:
    result = run_research(
        ResearchRequest(topic="Reliable agents", min_sources=1),
        runner=SingleRunRunner(_cited_run()),
        report_agent=EvidenceMarkerAgent(),
    )

    assert result.raw_report.endswith("[E1].")
    assert result.ledger is not None
    assert result.citations[0].evidence_id == result.ledger.evidence[0].id
    assert result.citations[0].source == result.ledger.evidence[0].source
    assert "[Evaluation study](https://example.com/evaluation)" in result.report


def test_unbound_legacy_source_marker_requires_review() -> None:
    result = run_research(
        ResearchRequest(topic="Reliable agents", min_sources=2),
        runner=SingleRunRunner(_run_with_an_unbound_first_source()),
        report_agent=UnboundLegacySourceAgent(),
    )

    assert len(result.citations) == 1
    assert result.citations[0].evidence_id == ""
    assert result.status == "needs_review"
    assert result.stop_reason == "unbound_citations"


def test_forged_evidence_marker_cannot_hide_beside_a_valid_marker() -> None:
    result = run_research(
        ResearchRequest(topic="Reliable agents", min_sources=1),
        runner=SingleRunRunner(_cited_run()),
        report_agent=MixedBoundAndForgedEvidenceMarkerAgent(),
    )

    assert len(result.citations) == 1
    assert result.citations[0].evidence_id
    assert result.status == "needs_review"
    assert result.stop_reason == "unbound_citations"


def test_evidence_marker_cannot_be_moved_to_an_unrelated_claim() -> None:
    result = run_research(
        ResearchRequest(topic="Reliable agents", min_sources=1),
        runner=SingleRunRunner(_cited_run()),
        report_agent=MisusedEvidenceMarkerAgent(),
    )

    assert len(result.citations) == 1
    assert result.citations[0].evidence_id == ""
    assert result.status == "needs_review"
    assert result.stop_reason == "unbound_citations"
    assert result.evidence[0].verification_status == "unverified"


def test_malformed_evidence_markers_fail_closed() -> None:
    result = run_research(
        ResearchRequest(topic="Reliable agents", min_sources=1),
        runner=SingleRunRunner(_cited_run()),
        report_agent=MalformedEvidenceMarkerAgent(),
    )

    assert len(result.citations) == 1
    assert result.status == "needs_review"
    assert result.stop_reason == "unbound_citations"


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


def test_citation_verifier_cannot_rebind_a_wrong_id_by_position() -> None:
    result = run_research(
        ResearchRequest(
            topic="Reliable agents",
            min_sources=1,
            budget=ResearchBudget(max_revision_rounds=0),
        ),
        runner=SingleRunRunner(_cited_run()),
        report_agent=WrongBindingCitationAgent(),
    )

    assert len(result.citation_checks) == 1
    assert result.citation_checks[0].status == "uncertain"
    assert result.citation_checks[0].claim_id == "C1"
    assert result.citation_checks[0].evidence_id == result.citations[0].evidence_id
    assert result.status == "needs_review"
    assert result.stop_reason == "unverified_citations"


def test_fallback_report_binds_canonical_source_aliases() -> None:
    result = run_research(
        ResearchRequest(topic="Reliable agents", min_sources=1),
        runner=SingleRunRunner(_run_with_a_canonical_source_alias()),
    )

    assert result.status == "completed"
    assert result.citations[0].evidence_id
    assert result.citations[0].source == result.sources[0]
    assert "utm_source" not in result.citations[0].source.url


def test_fallback_report_replaces_a_runner_supplied_forged_evidence_id() -> None:
    result = run_research(
        ResearchRequest(topic="Reliable agents", min_sources=1),
        runner=SingleRunRunner(_run_with_a_forged_evidence_id()),
    )

    assert result.status == "completed"
    assert result.ledger is not None
    assert result.citations[0].evidence_id == result.ledger.evidence[0].id
    assert result.citations[0].evidence_id != "ev_forged"


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
    assert result.ledger is not None
    assert len({item.id for item in result.ledger.evidence}) == 2
    assert [item.evidence_id for item in result.citations] == [
        item.id for item in result.ledger.evidence
    ]
    assert result.sources == result.ledger.sources
    assert result.evidence == result.ledger.evidence
    assert result.conflicts == result.ledger.conflicts
    assert result.citation_checks == result.ledger.checks
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

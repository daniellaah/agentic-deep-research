# Development Baselines

This document records reproducible development baselines without publishing benchmark questions, reference answers, or per-case predictions.

## Protocol

- Architecture: pre-adaptive single Responses API research call at commit `f452a68`
- Evaluation date: 2026-08-19
- Agent model: `gpt-5-nano`
- Judge model: `gpt-5-nano` with Structured Outputs
- Sample: deterministic first 10 cases from each benchmark
- Search protocol: OpenAI Responses API live `web_search`
- Maximum web-tool calls: 8 per case
- Maximum output tokens: 20,000 per case
- Quality gates: at least two consulted sources, a non-empty report, and valid citation-linked Evidence
- Execution: sequential, one run per case, no retry or best-of selection

## Results

| Benchmark | Agent completed | Judge completed | Correct | Accuracy | Avg. Evidence | Avg. web actions | Avg. sources | Avg. tokens | Approx. wall time/case |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FRAMES dev-10 | 10/10 | 10/10 | 9/10 | 90% | 12.8 | 7.0 | 90.5 | 40,070.7 | 68.1 s |
| BrowseComp-Plus dev-10 | 8/10 | 8/10 | 6/10 | 60% | 6.6 | 8.8 | 105.3 | 46,482.8 | 79.4 s |

The two BrowseComp-Plus cases that did not complete generated reports without citation-linked Evidence. The quality gate marked them `needs_review/missing_citations`, and the judge did not score them as successful answers.

## Interpretation

These are development baselines, not official full-benchmark scores:

- Ten deterministic cases are too few for a stable performance claim or resume bullet.
- FRAMES contains 824 cases. Its paper evaluates free-form answers with an LLM autorater; our judge follows that pattern but uses a different model.
- BrowseComp-Plus contains 830 cases and is designed around a fixed corpus of roughly 100,000 documents. This development run used live web search, so it is not comparable to its official leaderboard protocol.
- The agent and judge used the same model, which may introduce correlated judging bias.
- The aggregate wall times above come from whole CLI runs. Per-case elapsed time is recorded by the harness for future runs, but was added after these two baselines.

Official sources: [FRAMES dataset](https://huggingface.co/datasets/google/frames-benchmark), [FRAMES paper](https://arxiv.org/abs/2409.12941), and [BrowseComp-Plus repository](https://github.com/texttron/BrowseComp-Plus).

## Verified adaptive calibration

A one-case calibration was run after adding adaptive planning, evidence-aware workers,
report critique, semantic citation verification, gap search, and bounded revision. This is a
pipeline calibration, not an accuracy comparison with the 10-case baseline above.

- Evaluation date: 2026-08-19
- Agent and judge model: `gpt-5-nano`
- Sample: deterministic first case from each local development dataset
- Maximum research web-tool calls: 4 per case
- Maximum research steps: 2 per case
- Maximum output tokens: 10,000 per model response
- Maximum citation-verification calls: 1 per case
- Maximum report revisions: 1 per case
- Execution: sequential, one run per benchmark

| Benchmark | Completed | Correct | Trace tool actions | Sources | Tokens | Latency | Final stop reason |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FRAMES calibration-1 | 0/1 | not judged | 7 | 67 | 85,752 | 308.8 s | `research_budget_exhausted` |
| BrowseComp-Plus calibration-1 | 0/1 | not judged | 6 | 59 | 65,053 | 152.7 s | `max_research_steps` |

The calibration exposed three concrete bottlenecks:

1. A low structured-output limit can end a reasoning-model response before its JSON is
   complete. The planner now reports provider incompleteness before attempting JSON parsing,
   and CLI/benchmark output limits are passed consistently to planner and report stages.
2. Large search-result lists do not guarantee usable evidence. Both BrowseComp workers
   completed but produced no citation-linked Evidence, so replanning consumed the remaining
   research-step budget without resolving the evidence gap.
3. The quality gate correctly prevented judging weak outputs. The FRAMES report retained one
   unsupported and four uncertain citation checks; the BrowseComp report retained one
   unsupported check.

The `tool_calls` development metric currently counts observable search, open-page, and
find-in-page actions. It should be read as trace tool actions, not as a provider billing count.
The next optimization target is evidence yield per action, not a larger search budget.

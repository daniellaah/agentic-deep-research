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

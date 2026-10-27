# Memo benchmark: routed vs all-Ultra vs all-Nano

Memo pass rule (headline): mean of the five criteria >= 4.0 and accuracy >= 4.

Locked labeling thresholds: pass_threshold=4.0, min_accuracy=5 (locked 2026-10-21T02:54:17+00:00, source: calibration); unchanged, reported as a second pass rate below.

Frozen prompts: 40 in data/memo_bench/prompts.jsonl. Complete memo/judgment pairs: 119 of 120.
Excluded (no memo or no judgment after the resume pass, see logs): m019/all_nano.

Method:
- Prompts are frozen before any memo runs; every prompt is run under all three strategies.
- Evidence is shared: for each question, Nano plans the search queries once and the cached search results are frozen to data/memo_bench/evidence.jsonl. All three strategies get identical sources (checked before judging).
- `routed` uses the Nano router to pick the synthesis tier (extract is Nano); `all_ultra` and `all_nano` use that model for extract and synthesis. The shared plan step is Nano for every strategy, so `all_ultra` is all-Ultra except for planning.
- The judge is blind: it sees the question, the memo and numbered source snippets, never the strategy or model.
- Confidence intervals are 95% bootstrap CIs over complete memos.
- Memo cost sums every model call in the pipeline (route, plan, extract, synthesize); each memo includes the shared plan call's cost. The judge is evaluation overhead and is not included in the memo cost.

| Strategy | n | Cost / memo (USD) | 95% CI | Cost as % of all-Ultra | p50 latency (s) | p95 latency (s) | Mean score | 95% CI | Pass rate | 95% CI | Tier mix |
|---|---|---|---|---|---|---|---|---|---|---|---|
| routed | 40 | 0.00576 | [0.00489, 0.00669] | 28% | 37.64 | 65.86 | 4.13 | [3.96, 4.29] | 32.5% | [17.5, 47.5]% | T2:15, T3:25 |
| all_ultra | 40 | 0.02045 | [0.01947, 0.02147] | 100% | 27.17 | 35.98 | 4.41 | [4.28, 4.53] | 62.5% | [47.5, 77.5]% | T3:40 |
| all_nano | 39 | 0.00133 | [0.00122, 0.00145] | 7% | 43.91 | 72.86 | 3.85 | [3.65, 4.05] | 25.6% | [12.8, 38.5]% | T1:39 |

Pass rate above uses the memo rule (mean >= 4.0, accuracy >= 4). This is the labeling config's pre-calibration default; calibration tightened routing labels to accuracy >= 5, a bar a multi-claim memo rarely meets, so that stricter rule is reported separately below.

Pass rate under the locked labeling rule (mean >= 4.0, accuracy >= 5):

| Strategy | n | Pass rate | 95% CI |
|---|---|---|---|
| routed | 40 | 12.5% | [2.5%, 22.5%] |
| all_ultra | 40 | 15.0% | [5.0%, 27.5%] |
| all_nano | 39 | 10.3% | [2.6%, 20.5%] |

Paired view: routed vs all_ultra on the same questions, by the tier the router picked. On T3 rows both strategies synthesize with Ultra, so a gap there points to the extract step (Nano for routed, Ultra for all_ultra), plus sampling noise.

| Routed tier | n | Mean: routed | Mean: all_ultra | routed better / tie / worse | Accuracy: routed | Accuracy: all_ultra | Grounding: routed | Grounding: all_ultra |
|---|---|---|---|---|---|---|---|---|
| T2 | 15 | 4.27 | 4.55 | 6 / 1 / 8 | 3.60 | 3.93 | 3.60 | 3.80 |
| T3 | 25 | 4.06 | 4.33 | 6 / 3 / 16 | 3.04 | 3.60 | 2.68 | 3.24 |

![Cost vs quality](charts/cost_vs_quality_memos.png)

![Routed tier mix](charts/tier_mix_memos.png)

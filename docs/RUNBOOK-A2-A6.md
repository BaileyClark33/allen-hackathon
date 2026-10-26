# Runbook: A2–A6 (Person A, Oct 25–29)

Branch `a2-a6-results`. Paid runs are serial and each one is estimated before it runs.
Every command below runs from the repo root with `.venv/bin/python`.

**Budget at start:** lifetime spend $5.5486 of $100 (remaining $94.4514).
Chunk caps: A2 $35 total ($2.53 spent, $32.47 left; this run cap $30), A3 $3, A3b $3, A4 $12.
On any resume, the cap is the remaining chunk budget, never the original cap.

**Stop and ask on:** an over-budget estimate; a failed gate below; two consecutive test
failures; an overwrite of a frozen artifact (`data/splits/test.jsonl`,
`data/memo_bench/prompts.jsonl`); missing B2 at the A4 gate; any spend anomaly outside
`logs/calls.jsonl`.

## Status

| ID | Step | Command | Exit check | Cap | Artifact | Gate |
|---|---|---|---|---|---|---|
| 0.1 | Runbook | — | this file committed | $0 | `docs/RUNBOOK-A2-A6.md` | — |
| 0.2 | Branch + PR | `git fetch origin && git checkout a1-calibration-hardening && git merge origin/main && git checkout -b a2-a6-results` | branch pushed, draft PR open | $0 | branch `a2-a6-results` | tests pass |
| 0.3 | Verify lock | `.venv/bin/python -c "from nemoroute.config import load_config, active_thresholds; print(active_thresholds(load_config()))"` | prints `(4.0, 5)` | $0 | — | — |
| 0.4 | Tavily key path | `.venv/bin/python -c "from nemoroute.config import load_config, api_key; print(len(api_key(load_config(), 'TAVILY_API_KEY')))"` | prints 58 | $0 | — | — |
| A2.1 | Label full set | `.venv/bin/python scripts/label_queries.py --dataset full --full --yes --max-cost 30` | 600/600 labels, 1800 answers, 1800 judgments | $30 | `data/full/{responses,judgments,labels}.jsonl` | no tier >60%; unresolved reported |
| A2.2 | Relabel (no API) | `.venv/bin/python scripts/label_queries.py --dataset full --relabel` | 600 labels, distribution + confusion printed | $0 | `data/full/labels.jsonl` | — |
| A2.3 | Truncation audit | list `finish_reason=length` responses; inspect deciding passes | every truncated deciding pass is label-neutral | $0 | note in this file | stop if a truncated answer sets a label |
| A2.4 | Frozen split | `.venv/bin/python scripts/split_dataset.py` (never `--force`) | 600 labels, test 20%, disjoint, calibration excluded | $0 | `data/splits/{train,test,split_meta}.jsonl/json` | calibration-overlap check passes |
| A3.1 | Router evaluation | `.venv/bin/python scripts/evaluate.py --full --yes --max-cost 3` | results + charts written | $3 | `results/results.md`, `results/charts/` | under-routing reported (prompt edits listed, not applied) |
| A3b.1 | MMLU-Pro dry run | `.venv/bin/python scripts/run_public_bench.py --limit 20` (estimate only first) | estimate captured, unparseable rate | ≤$3 chunk | — | stop if unparseable >5% |
| A3b.2 | MMLU-Pro full | `.venv/bin/python scripts/run_public_bench.py --full --yes --max-cost <remaining A3b>` | 300 frozen items, results table | $3 chunk | `results/public_bench/results_zero_shot_nano.md`, `data/public/mmlu_pro/items.jsonl` | — |
| A4.0 | Memo gate (Oct 26 18:00) | check `src/nemoroute/agent/memo.py`, `scripts/run_memo.py`, Tavily key | run_memo exists AND key loads | $0 | — | stop and ask if either is missing |
| A4.1 | Memo bench code | build `build_memo_bench.py`, `memo_judge.txt`, `evaluate_memos.py`, tests | `pytest` passes with fakes | $0 | code + tests | no live calls in tests |
| A4.2 | Human prompt gate | show 20 draft prompts | user picks/edits 15 | $0 | — | do not freeze before confirmation |
| A4.3 | Freeze memo prompts | `.venv/bin/python scripts/build_memo_bench.py` | 40 prompts (25 generated + 15 hand-written), frozen | ≤$12 chunk | `data/memo_bench/prompts.jsonl` | refuse overwrite |
| A4.4 | Memo dry run | `.venv/bin/python scripts/evaluate_memos.py --limit 3` | costs shown | ≤$12 chunk | `data/memo_bench/` outputs | — |
| A4.5 | Memo full | `.venv/bin/python scripts/evaluate_memos.py --full --yes --max-cost <remaining A4>` | mean score, pass rate, cost, latency, CIs | $12 chunk | `results/memo_results.md` | — |
| A5.1 | Memo agreement code | extend `human_agreement.py`, `sample_for_human.py`, `score_human_agreement.py` with `--target memo` | `pytest` passes with fakes | $0 | code + tests | — |
| A5.2 | Human labels | user labels 30 memos from the blind CSV | 30 human pass/fail labels | $0 | `data/memo_bench/memo_human*.csv` | user does this personally |
| A5.3 | Agreement score | `.venv/bin/python scripts/score_human_agreement.py --target memo` | percent agreement + Cohen's kappa appended | $0 | `results/memo_results.md` | — |
| A6.1 | README results | rewrite Results from `results/results.md` + `results/memo_results.md` | headline with CIs, one table per benchmark, two charts, Limitations | $0 | `README.md` | every number matches a results file |
| A6.2 | Submission checklist | final checklist for the user | all items ticked or flagged | $0 | `docs/SUBMISSION-CHECKLIST.md` | video <3 min, Devpost, MIT repo, demo URL, spend posted |

## Spend log

| When | Step | Estimate | Actual | Lifetime after |
|---|---|---|---|---|
| Oct 25 | start | — | — | $5.5486 |
| Oct 25 | A2.1 full labeling | $19.1122 | $17.9709 | $23.5194 |

A2 chunk total: $2.53 before + $17.9709 = **$20.50 of $35**.

## Wave log

- **Wave 0: done.** Runbook, branch `a2-a6-results`, draft PR #5, lock `(4.0, 5)`, Tavily key path verified.
- **Wave 1 (A2): done.** 600/600 labels: tier 1 209 (34.8%), tier 2 168 (28.0%), tier 3 88 (14.7%), unresolved 135 (22.5%). Confusion (intended rows → label): T1 160/32/6/2, T2 21/76/42/61, T3 28/60/40/72. Tier 3 is 14.7%, marginally under the roadmap's 15% heuristic; no re-generation (spend already committed). Truncation audit: 19 truncated answers, all fail, 0 deciding passes, forcing all to fail changes 0 labels → label-neutral. Split: seed 42, 371 train / 121 test (test 20.2%; test mix 42/34/18/27), disjoint, calibration excluded, `split_meta.json` written.
- **Wave 2 (A3/A3b):** pending.
- **Wave 3 (A4):** pending.
- **Wave 4 (A5):** pending.
- **Wave 5 (A6):** pending.

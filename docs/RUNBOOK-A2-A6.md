# Runbook: A2–A6 (Person A, Oct 25–29)

Branch `a2-a6-results` (PR #5, merged 2026-10-27 by Bailey); A4–A6 continue on `a4-a6-memo-results`. Paid runs are serial and each one is estimated before it runs.
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
| A4.4b | Shared-evidence fix + dry run 2 | code: `gather_evidence` + `evidence=`; then `evaluate_memos.py --limit 3` (estimate, then `--yes --max-cost <remaining A4>`) | identical source URLs per question across strategies; citations ≤ len(sources); step costs sum | ≤$12 chunk | `data/memo_bench/evidence.jsonl` | stop if shared-source check fails |
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
| Oct 25 | A3.1 router evaluation | $0.0050 | $0.0036 | $23.5230 |
| Oct 25 | A3b.1 MMLU dry run (20) | $0.1908 | $0.0474 | $23.5704 |
| Oct 25 | A3b.2 MMLU full (300) | $2.6147 | $0.5492 | $24.1197 |
| Oct 25 | A4.3 memo prompt freeze | $0.0036 | $0.0125 | $24.1367 |
| Oct 26 | A4.4 memo dry run 1 (3 prompts; run `99c2608d9e4d`; archived, sources not shared) | $0.1376 | $0.2844 | $24.4211 |
| Oct 26 | A4.4b memo dry run 2 (3 prompts, shared evidence; run `f98b9ff53dc2`) | $0.1225 | $0.3203 | $24.7415 |
| Oct 26 | A4.5 memo full (37 prompts; run `9b5f0a8a4cf8`) | $1.5232 | $3.6967 | $28.4382 |
| Oct 26 | A4.5 resume pass (2 memos, 3 judgments; run `52743111c3ad`) | $0.0338 | $0.0599 | $28.4980 |

A2 chunk total: $2.53 before + $17.9709 = **$20.50 of $35**.
A3 total: **$0.0036 of $3**. A3b total: **$0.5966 of $3**. A4 total so far: **$4.3738 of $12**
(includes a $0.0036 generator call that returned 26 questions and was rejected before writing).

## Wave log

- **Wave 0: done.** Runbook, branch `a2-a6-results`, draft PR #5, lock `(4.0, 5)`, Tavily key path verified.
- **Wave 1 (A2): done.** 600/600 labels: tier 1 209 (34.8%), tier 2 168 (28.0%), tier 3 88 (14.7%), unresolved 135 (22.5%). Confusion (intended rows → label): T1 160/32/6/2, T2 21/76/42/61, T3 28/60/40/72. Tier 3 is 14.7%, marginally under the roadmap's 15% heuristic; no re-generation (spend already committed). Truncation audit: 19 truncated answers, all fail, 0 deciding passes, forcing all to fail changes 0 labels → label-neutral. Split: seed 42, 371 train / 121 test (test 20.2%; test mix 42/34/18/27), disjoint, calibration excluded, `split_meta.json` written.
- **Wave 2 (A3/A3b): done.** Router: accuracy 59.6% (95% CI 50.0–70.2), under-routing 9.6% (below 15%, no prompt edits), over-routing 30.9%; cost 58% of all-Ultra on resolved items (66% on all items); pass rate 79.8% vs all-Ultra 77.7% on resolved (62.0% vs 60.3% on all). MMLU-Pro: 300 frozen items, 1 unparseable tier answer (0.1%); all-Nano 66.7%, all-Super 88.7%, all-Ultra 88.7%, router 68.0% at 15% of all-Ultra cost, oracle 92.0%.
- **Wave 3 (A4): code + frozen prompts done; live blocked on B2.** 40 prompts (25 generated + 15 handwritten), unique, SHA256 `791ec5c5b60a966546c392f13173ae418610600dac48c4411f3c7c3eed0c495d`. Evaluation estimate for 120 memos + 120 judgments: **$1.8352**, inside A4. B1/B2 files still absent on Oct 25; gate at Oct 26 18:00.
- **Wave 3b (A4 fix, Oct 26).** B1/B2 landed (gate passed). Dry run 1 audit: strategies did not share sources (common URLs per question: 2/1/2 of 17–51), because B2 planned at each strategy's tier with temperature 0.3 and the cache is per query. Fix (user-approved): plan + search once per question with Nano, frozen to `data/memo_bench/evidence.jsonl`, reused by all three strategies; a shared-source gate blocks judging on mismatch. Dry run 1 also passed 0% under the locked rule (accuracy ≥ 5); user approved a disclosed memo rule, mean ≥ 4.0 and accuracy ≥ 4 (`memo_bench` in config.yaml), with the locked-rule pass rate reported alongside. `labeling.locked` unchanged. Dry-run 1 outputs archived to `data/memo_bench/archive/dryrun-2026-10-26/`. Dry run 2 (shared evidence): 3 evidence rows, 9 memos, 9 judgments, 0 failures; identical source URLs per question (22–41 sources), all citations in range, step costs sum, run spend in `logs/calls.jsonl` matches. Actual was 2.6x the estimate (extract inputs are much larger than the observed per-role averages), so the full-run cap is the remaining A4 budget.
- **Wave 3c (A4 full, Oct 26): done.** 40 prompts x 3 strategies; 119 of 120 memo/judgment pairs. First pass: 2 memos failed (extract JSON truncated at max_tokens: m019/all_nano, m026/routed) and 1 judgment (judge output truncated: m022/routed). One resume pass recovered m026/routed and both judgments; m019/all_nano failed the same way again and is excluded (counted as an all-Nano pipeline failure, not retried further). Audit: identical sources for all 40 questions (17–56 sources), step costs sum, spend matches `logs/calls.jsonl`; one all-Nano memo (m005) has no citations and is scored as is; one synthesis was truncated and is scored as is. Results in `results/memo_results.md` (memo rule + locked rule + paired routed-vs-Ultra table by routed tier).
- **Wave 4 (A5): code done; blocked on memo outputs.** `--target memo` sampling/scoring and tests are committed. Optional 50-answer test sample prepared at `data/splits/test_human.csv` (blank `human_pass`), key at `data/splits/test_human_key.jsonl`.
- **Wave 5 (A6): routing + MMLU README sections done.** Memo headline, tier-mix chart and final Limitations wait on B2/A4. Every README number currently matches `results/results.md` and `results/public_bench/results_zero_shot_nano.md`.

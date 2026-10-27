# Public benchmark: MMLU-Pro (business,economics), router zero_shot_nano

Items: 300 (seeded sample, frozen in data/public/mmlu_pro/items.jsonl; never used for training). Scored by exact match on the final letter, so no LLM judge is involved. Unparseable tier answers (counted wrong): 1.

In the table, Mean score and Pass rate both equal accuracy. Router cost includes the routing call.

| Strategy | n | Cost / query (USD) | 95% CI | Cost as % of all-Ultra | p50 latency (s) | p95 latency (s) | Mean score | 95% CI | Pass rate | 95% CI | Tier mix |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all-nano | 300 | 0.00003 | [0.00003, 0.00004] | 3% | 0.49 | 1.53 | 0.67 | [0.61, 0.72] | 66.7% | [61.0, 72.0]% | T1:300 |
| all-super | 300 | 0.00091 | [0.00072, 0.00112] | 90% | 2.61 | 17.57 | 0.89 | [0.85, 0.92] | 88.7% | [85.0, 92.0]% | T2:300 |
| all-ultra | 300 | 0.00101 | [0.00090, 0.00118] | 100% | 1.06 | 2.07 | 0.89 | [0.85, 0.92] | 88.7% | [85.0, 92.0]% | T3:300 |
| zero_shot_nano | 300 | 0.00015 | [0.00008, 0.00026] | 15% | 1.19 | 3.07 | 0.68 | [0.63, 0.73] | 68.0% | [62.7, 73.0]% | T1:289, T2:11 |
| oracle | 300 | 0.00050 | [0.00035, 0.00068] | 49% | 0.77 | 10.46 | 0.92 | [0.89, 0.95] | 92.0% | [88.7, 95.0]% | T1:200, T2:70, T3:30 |

![Cost vs accuracy](charts/cost_vs_accuracy_zero_shot_nano.png)

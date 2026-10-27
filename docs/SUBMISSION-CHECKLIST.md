# Submission checklist

Deadline: **Thu Oct 30, 10:00 AM PT**. Internal target: **Wed Oct 29**.
Track: **Best Apps and Agents**.

## Evidence and repo

- [ ] `results/results.md` exists and every README number matches it.
- [ ] `results/public_bench/results_zero_shot_nano.md` exists with 300 frozen items.
- [ ] `results/memo_results.md` exists with per-strategy scores, CIs and human agreement.
- [ ] `pytest` passes on a fresh clone.
- [ ] `git status` is clean on the submission branch; `data/` and `logs/` are not committed.
- [ ] `git log -p | grep -i "api_key"` is clean (no secrets in history).
- [ ] Repo is public and `LICENSE` is MIT.
- [ ] README has setup, results, limitations, demo URL and the Nemotron/Token Factory/Tavily credit.

## Demo

- [ ] Hosted demo URL works logged out and in an incognito window.
- [ ] Demo uses the dedicated low-limit Token Factory key and the Tavily key.
- [ ] A query after the session cap is refused politely.
- [ ] Demo URL is in the README and stays live through Dec 15.

## Video and Devpost

- [ ] YouTube video is public and **under 3:00**.
- [ ] Video shows a live memo run, the cost panel and the Evidence page.
- [ ] Devpost description covers inspiration, what it does, how it was built, challenges, accomplishments, learnings and next steps.
- [ ] Devpost includes candid feedback on Nebius Token Factory and NVIDIA Nemotron.
- [ ] Devpost notes the repo was created Oct 14, inside the submission period.
- [ ] Track selected: Best Apps and Agents.

## Spend and integrity

- [ ] Lifetime Token Factory spend is posted from `logs/calls.jsonl` (not estimated).
- [ ] Chunk spend is within A2 $35, A3 $3, A3b $3, A4 $12.
- [ ] Thresholds are still locked at `pass_threshold=4.0`, `min_accuracy=5`.
- [ ] Test split is frozen; no `--force` was used on `data/splits/test.jsonl` or `data/memo_bench/prompts.jsonl`.
- [ ] Any router-prompt change made after seeing test results is disclosed in `results/results.md`.
- [ ] The final PR is reviewed and merged by the user (the agent never merges).

## Final commands

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -c "from nemoroute.usage_log import UsageLog; from pathlib import Path; print(UsageLog(Path('logs/calls.jsonl')).total_spend())"
git status --short --branch
git log -p | grep -i "api_key" || echo "clean"
```

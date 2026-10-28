# Submission checklist

Deadline: **Thu Oct 30, 10:00 AM PT**. Internal target: **Wed Oct 29**.
Track: **Best Apps and Agents**.

## Evidence and repo

- [x] `results/results.md` exists and every README number matches it. *(Checked Oct 26. Two numbers are derived: 20.2% = 121/600 test rows, and the 20.7-point MMLU gap = 88.7 − 68.0.)*
- [x] `results/public_bench/results_zero_shot_nano.md` exists with 300 frozen items.
- [x] `results/memo_results.md` exists with per-strategy scores, CIs and human agreement. *(Checked Oct 27. Human agreement on the 30 blind memos: 70.0%, Cohen's κ = 0.335; machine-readable copy in `results/human_agreement_memo.json`.)*
- [x] `pytest` passes on a fresh clone. *(95 passed on a clean worktree of `a4-a6-memo-results`, Oct 26.)*
- [ ] `git status` is clean on the submission branch; `data/` and `logs/` are not committed. *(No file under `data/`, `logs/` or `.env` is tracked. Re-check `git status` on the final branch.)*
- [x] `git log -p | grep -i "api_key"` is clean (no secrets in history). *(That grep also matches the `api_key()` helper by name. The real check, run Oct 26, found neither key value in any commit on any branch.)*
- [x] Repo is public and `LICENSE` is MIT.
- [ ] README has setup, results, limitations, demo URL and the Nemotron/Token Factory/Tavily credit. *(The demo URL in "Deploying" is still a placeholder.)*

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
- [ ] Devpost notes the repo was created Oct 14, inside the submission period. *(GitHub says the repo was created 2026-10-13T20:13Z, which is Oct 13 PT. Check that against the submission-period start before writing a date.)*
- [ ] Track selected: Best Apps and Agents.

## Spend and integrity

- [x] Lifetime Token Factory spend is posted from `logs/calls.jsonl` (not estimated): **$28.4980** as of Oct 26 (see `docs/RUNBOOK-A2-A6.md`).
- [x] Chunk spend is within A2 $35, A3 $3, A3b $3, A4 $12 (A2 $20.50, A3 $0.0036, A3b $0.5966, A4 $4.3738).
- [x] Thresholds are still locked at `pass_threshold=4.0`, `min_accuracy=5`. *(The memo benchmark uses its own disclosed `memo_bench` rule.)*
- [x] Test split is frozen; no `--force` was used on `data/splits/test.jsonl` or `data/memo_bench/prompts.jsonl`. *(Prompts SHA256 is still `791ec5c5…c495d`.)*
- [x] Any router-prompt change made after seeing test results is disclosed in `results/results.md`. *(None was made.)*
- [ ] The final PR is reviewed and merged by the user (the agent never merges).

## Final commands

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -c "from nemoroute.usage_log import UsageLog; from pathlib import Path; print(UsageLog(Path('logs/calls.jsonl')).total_spend())"
git status --short --branch
git log -p | grep -i "api_key" || echo "clean"
```

# Handoff: A4–A6 finish (closed Oct 27)

Branch `a4-a6-memo-results`, PR #6 — **merged** into `main` by Bailey (merge commit `6db8b32`). The agent never merges.

## Done
- **A4 memo benchmark:** 40 frozen prompts × 3 strategies, 119 of 120 pairs; `results/memo_results.md`, `.csv` and two charts.
  - Shared Nano evidence in `data/memo_bench/evidence.jsonl`. All 40 questions have identical sources across strategies.
  - The headline uses the memo rule (mean ≥ 4.0, accuracy ≥ 4). The locked rule (4.0, 5) is unchanged and reported alongside.
  - m019/all_nano is excluded: extract was cut off at the token limit twice.
- **A5 sample:** 30 blind memos (10 per strategy) in `data/memo_bench/memo_human.csv`; key in `memo_human_key.jsonl`.
- **A5 labels:** all 30 `human_pass` values filled: 4 pass, 26 fail.
- **A6:**
  - README intro, memo headline, table, both charts and memo Limitations are done.
  - Human agreement (30 memos): **70.0%, Cohen's κ = 0.335**; 9 judge-only passes, 6 all-Ultra and 3 routed. Section in `results/memo_results.md`, JSON in `results/human_agreement_memo.json`, README line updated.
  - Checklist items that could be verified are ticked, each with its evidence.
- **Spend:** lifetime **$28.4980** from `logs/calls.jsonl`. A4 used $4.3738 of $12. No paid steps remain.

## Remaining (all $0)
1. **User:** review and merge open PRs; the agent never merges.
2. After merging, re-check `git status` on `main` and tick checklist line 12.
3. Demo deploy, video and Devpost are outside this handoff; see `docs/SUBMISSION-CHECKLIST.md`.

**Trap:** re-running `evaluate_memos.py` (even `--report-only`) rewrites `memo_results.md` and drops the human-agreement section. Re-run `scripts/score_human_agreement.py --target memo` after any re-render.

## Worktree notes
- The worktree is `.claude/worktrees/a4-a6-memo-results`.
- `data/`, `logs/`, `.env` and `.venv` are symlinks to the main checkout. They show as untracked, so never `git add -A`.
- Run Python as `PYTHONPATH=src .venv/bin/python …`. The venv's editable install points at the main checkout's `src/`.

## Finding for Person B
On the 25 memos routed to Ultra, routed and all-Ultra both synthesize with Ultra, yet routed scores 4.06 vs 4.33. The gap is mostly accuracy (3.04 vs 3.60) and grounding (2.68 vs 3.24). That points to the Nano extract step. Running extract at the routed tier, or on Super, is the obvious next experiment. It is a pipeline change, so it is B's call, and it would need a re-run. The 30 human labels sharpen this: the human rated routed and all-Ultra equally (2 passes each in the sample) while the judge put all-Ultra ahead 8 to 5, so treat the all-Ultra advantage as partly judge leniency.

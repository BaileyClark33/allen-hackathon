# Handoff: A4–A6 finish (as of Oct 26)

Branch `a4-a6-memo-results`, draft PR #6. PR #5 (A2–A3b) was merged by Bailey on Oct 27 UTC. Never merge; the user does.

## Done
- **A4 memo benchmark:** 40 frozen prompts × 3 strategies, 119 of 120 pairs; `results/memo_results.md`, `.csv` and two charts.
  - Shared Nano evidence in `data/memo_bench/evidence.jsonl`. All 40 questions have identical sources across strategies.
  - The headline uses the memo rule (mean ≥ 4.0, accuracy ≥ 4). The locked rule (4.0, 5) is unchanged and reported alongside.
  - m019/all_nano is excluded: extract was cut off at the token limit twice.
- **A5 sample:** 30 blind memos (10 per strategy) in `data/memo_bench/memo_human.csv`; key in `memo_human_key.jsonl` (don't open it while labeling).
- **A6:**
  - README intro, memo headline, table, both charts and memo Limitations are done. Only the human-agreement line is pending.
  - Checklist items that could be verified are ticked, each with its evidence.
- **Spend:** lifetime **$28.4980** from `logs/calls.jsonl`. A4 has used $4.3738 of $12. No paid steps remain.

## Remaining (all $0)
1. **User:** fill `human_pass` (1/0) for the 30 rows of `data/memo_bench/memo_human.csv`.
   - The question is: would you hand this memo to a client as is? The judge's bar is mean ≥ 4 and accuracy ≥ 4.
2. Run `PYTHONPATH=src .venv/bin/python scripts/score_human_agreement.py --target memo`.
   - This appends `## Human agreement` to `results/memo_results.md` and writes `results/human_agreement_memo.json`.
   - **Trap:** re-running `evaluate_memos.py` (even `--report-only`) rewrites `memo_results.md` and drops that section. Re-run step 2 after any re-render.
3. Replace the README line "Human agreement … *pending*" with the percent agreement and kappa from `results/memo_results.md`.
4. Check every README number against `results/` and tick the memo checklist item.
5. Commit with explicit paths, push, and mark PR #6 ready only if the user asks.

## Worktree notes
- The worktree is `.claude/worktrees/a4-a6-memo-results`.
- `data/`, `logs/`, `.env` and `.venv` are symlinks to the main checkout. They show as untracked, so never `git add -A`.
- Run Python as `PYTHONPATH=src .venv/bin/python …`. The venv's editable install points at the main checkout's `src/`.

## Finding for Person B
On the 25 memos routed to Ultra, routed and all-Ultra both synthesize with Ultra, yet routed scores 4.06 vs 4.33. The gap is mostly accuracy (3.04 vs 3.60) and grounding (2.68 vs 3.24). That points to the Nano extract step. Running extract at the routed tier, or on Super, is the obvious next experiment. It is a pipeline change, so it is B's call, and it would need a re-run.

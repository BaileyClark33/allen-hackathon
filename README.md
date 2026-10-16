# allen-hackathon

**Nemotron consulting router.** A small, fast NVIDIA Nemotron model reads each consulting or business-strategy question and routes it to the cheapest Nemotron model that can answer it well. All models are served at runtime by the **Nebius Token Factory** inference API.

Built for the Nebius x NVIDIA Global AI Hackathon (Best Apps and Agents track).

| Tier | Model (Token Factory ID) | Handles |
|---|---|---|
| Router + Tier 1 | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | definitions, factual lookups |
| Tier 2 | `nvidia/nemotron-3-super-120b-a12b` | frameworks, structured analysis |
| Tier 3 | `nvidia/Nemotron-3-Ultra-550b-a55b` | multi-step strategy, market sizing, ambiguous cases |

## Results

_Pending: populated from `results/results.md` after the held-out test evaluation._

Headline metrics: routing accuracy, under-routing rate, cost per query, latency, and answer quality, compared with an all-Ultra baseline, an all-Nano baseline, the zero-shot Nano router, the fine-tuned Nano router, a closed frontier reference model, and an oracle router (upper bound). Reported on two benchmarks:

- the frozen held-out split of our consulting dataset (LLM-judged, with a second judge as a bias check)
- a seeded MMLU-Pro business/economics sample (exact-match scored, no judge)

## How Nemotron and Token Factory are used

- **Inference.** Every model call goes through Token Factory's OpenAI-compatible endpoint (`https://api.tokenfactory.nebius.com/v1/`) using the `openai` Python SDK. See `src/nemoroute/client.py`.
- **Routing.** Nemotron 3 Nano classifies each query into tier 1, 2 or 3 (reasoning off). Two routers sit behind the `Router` protocol (`src/nemoroute/routers/base.py`) and are selected with `router.type` in `config.yaml` (or `--router`):
  - `zero_shot`: prompted Nano on Token Factory.
  - `finetuned`: Nano post-trained with LoRA on the labeled train split, run on a Nebius AI Cloud GPU and served with vLLM (`training/`).
- **Answering.** The chosen tier's Nemotron model answers the query.
- **Dataset generation.** Nemotron 3 Ultra writes realistic consulting queries across three difficulty levels, deduplicated with TF-IDF cosine similarity.
- **Labeling.** Nano, Super and Ultra all answer every query. Nemotron 3 Ultra (reasoning on) grades each answer blind against a written rubric: accuracy, completeness, structure and actionability. The label is the cheapest tier whose answer passes.
- **Cost accounting.** Every call logs model, input/output tokens, latency and cost (from Token Factory per-token prices) to `logs/calls.jsonl`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your Token Factory key to `.env` as `NEBIUS_API_KEY=...`. `.env` is gitignored.

All model IDs, prices, thresholds, paths and caps live in `config.yaml`.

## Run the demo app

```bash
streamlit run app/streamlit_app.py
```

Type a question. The app shows:
- the tier the router chose and its reasoning
- the answer
- that query's cost and latency

The sidebar tracks cumulative session cost and the estimated savings against sending everything to Ultra.

### Spend protection for the public demo

Set in `config.yaml` under `app:`:
- `session_query_cap`: queries allowed per browser session
- `daily_spend_cap_usd`: global spend cap per UTC day
- `total_spend_cap_usd`: lifetime cap
- `max_query_chars` and `answer_max_tokens`: bound the cost of a single query

App spend is logged to `logs/app_spend.jsonl`. The log is shared by every session in a single Streamlit process.

**Streamlit Community Cloud caveat:** the filesystem there is ephemeral, so `logs/app_spend.jsonl` resets whenever the app reboots. On that platform the daily and lifetime caps are best-effort only. For a hard backstop, give the deployed app a dedicated Token Factory API key with a low credit balance or spend limit. Provide the key through Streamlit secrets (`NEBIUS_API_KEY`).

## Pipeline

Every step that spends credits:
- prints a cost estimate first. It starts from config defaults and switches to observed per-role averages once a few real calls exist.
- shows lifetime spend against the $100 budget
- asks for confirmation before running
- takes `--limit N` for a dry run (default 20), or `--full` to process everything
- takes `--max-cost USD` as a hard stop mid-run

Every stage writes to append-only JSONL files and skips work that is already done, so interrupted runs resume without paying twice.

```bash
python scripts/probe_api.py
python scripts/generate_queries.py --dataset calibration --limit 20
python scripts/label_queries.py --dataset calibration --limit 20
python scripts/sample_for_human.py --target calibration
python scripts/tune_thresholds.py
python scripts/tune_thresholds.py --lock 4.0 4
python scripts/generate_queries.py --dataset full --full
python scripts/label_queries.py --dataset full --limit 20
python scripts/label_queries.py --dataset full --full
python scripts/split_dataset.py
python scripts/evaluate.py --full
python scripts/sample_for_human.py --target test
python scripts/score_human_agreement.py --target test
```

What each command does:
1. `probe_api.py` checks model IDs, hosts, the reasoning toggle and JSON support against the live API.
2. `generate_queries.py --dataset calibration --limit 20` generates the first 20 calibration queries (a dry run).
3. `label_queries.py --dataset calibration --limit 20` answers, judges and labels them, then prints the label distribution.
4. `sample_for_human.py --target calibration` exports calibration answers for human pass/fail labels.
5. `tune_thresholds.py` runs a grid search on the calibration set only. It makes no API calls.
6. `tune_thresholds.py --lock 4.0 4` locks the approved thresholds into `config.yaml`.
7. `generate_queries.py --dataset full --full` generates the full set, deduplicated against the calibration set.
8. `label_queries.py --dataset full --limit 20` is a dry run plus the class-balance check.
9. `label_queries.py --dataset full --full` labels the full set.
10. `split_dataset.py` makes a seeded, stratified 80/20 split and freezes the test set.
11. `evaluate.py --full` runs routing metrics, system metrics, charts and `results/results.md`.
12. `sample_for_human.py --target test` exports test answers for the human-agreement check.
13. `score_human_agreement.py --target test` computes percent agreement and Cohen's kappa between the human labels and the judge.

### Phase 2: fine-tuned router, frontier reference, public benchmark

Run these after the Phase 1 steps above.

```bash
python scripts/export_sft.py
# on the Nebius AI Cloud GPU VM (pip install -r training/requirements.txt):
python training/train_lora.py --smoke
python training/train_lora.py
ROUTER_FT_API_KEY=... bash training/serve.sh runs/router-ft/merged
# back on your machine: set models.nano_ft.base_url in config.yaml and ROUTER_FT_API_KEY in .env
python scripts/evaluate.py --router finetuned --full
python scripts/evaluate.py --full
python scripts/run_reference.py --full
python scripts/evaluate.py --full
python scripts/run_reference.py --judge reference_judge --full
python scripts/evaluate.py --full --judge reference_judge
python scripts/run_public_bench.py --full --reference
python scripts/run_public_bench.py --full --router finetuned
```

What each command does:
1. `export_sft.py` writes `data/sft/{train,val}.jsonl` from the train split. It makes no API calls. It refuses to run if any test-set, calibration or public-benchmark query would leak into training.
2. `train_lora.py --smoke` runs 10 training steps to check the GPU stack (PEFT on Nano's hybrid Mamba/MoE layers) before the full run.
3. `train_lora.py` trains with loss on the `{"tier": N}` completion only, early-stops on a 10% validation holdout, reports validation routing accuracy, and merges the adapter.
4. `serve.sh` serves the merged checkpoint as an OpenAI-compatible endpoint.
5. `evaluate.py --router finetuned` routes the test set with the fine-tuned router and writes `results/finetuned_nano/`.
6. `evaluate.py` (zero-shot) then includes every router that has saved predictions in the same `results/results.md` table.
7. `run_reference.py` answers the test set with the closed reference model (`models.reference`, default Claude Opus 5.5, `ANTHROPIC_API_KEY`) and grades those answers blind with the same Ultra judge and rubric. Re-running `evaluate.py` adds a `reference` row.
8. `run_reference.py --judge reference_judge` re-grades the reference answers and all Nemotron tier answers with the reference model as judge, a check on Ultra grading its own answers. `evaluate.py --judge reference_judge` writes that view to `results/judge_reference_judge/`.
9. `run_public_bench.py` samples 300 MMLU-Pro business/economics questions (frozen on first run). Every tier answers them, plus the reference model with `--reference`. It routes them and scores all strategies by exact match into `results/public_bench/`. Run it once per router.

**Billing.** The `$100` budget guard counts only Token Factory spend. Reference-model calls are billed by Anthropic, and fine-tuned router calls run on your own VM. Cost estimates list both separately, and every call is still logged. The fine-tuned router is priced at Nano's Token Factory rate, so router cost stays comparable with the zero-shot router.

### Evaluation hygiene

- **Calibration is kept separate.** Thresholds are tuned only on a separate 60-query calibration set, before the full set is labeled or split. Calibration queries never enter the test set, and `split_dataset.py` checks this.
- **The pipeline enforces the order.** `label_queries.py --full` (full set), `split_dataset.py` and `evaluate.py` all refuse to run until thresholds are locked.
- **Threshold changes are flagged.** Locked values are recorded in `config.yaml` and in `results.md`. If thresholds change after a test evaluation, `results.md` says so.
- **Unresolved queries are counted, not hidden.** A query where no tier passes is labeled `unresolved`, not tier 3. It is excluded from training and from routing accuracy, and its rate is reported separately.
- **The test set is frozen.** It is read only by `evaluate.py` and `sample_for_human.py`.
- **Baselines share the same answers.** Every strategy (all-Nano, all-Ultra, router, oracle) reuses the same cached answers and judge scores. Router cost and latency include the routing call.

### Limitations

- **Self-grading.** Ultra grades answers it also produced. The mitigations:
  - blind judging
  - absolute rubric scores rather than pairwise comparison
  - a human-agreement check on the test set
- **Labels depend on the judge.** They reflect this judge, rubric and threshold, not ground truth.
- **Fine-tuning runs on Nebius AI Cloud, not Token Factory.** Token Factory's fine-tuning catalog does not currently list Nemotron models. The router is trained on the open weights on a Nebius AI Cloud GPU and served with vLLM. If that endpoint is down, the demo app falls back to the zero-shot router.
- **The closed reference model is for evaluation only.** It never answers in the product path. It runs at its default thinking depth (effort `medium`), while the Nemotron tiers answer with reasoning off. Refusals are recorded as empty answers and score as fails.
- **The training set is small.** The router is fine-tuned on a few hundred labeled queries, so routing accuracy has wide confidence intervals. The report shows bootstrap CIs.

## Tests

```bash
pytest
```

Tests use a fake client and make no API calls.

## License

MIT. See [LICENSE](LICENSE).

# Sufficient Reasoning Demand: Difficulty, Capability, or Task Structure?

Companion code and data for the manuscript
*"Task structure governs the inference cost of large language models"*.

The project measures the **minimum reasoning a task requires** — the
*sufficient reasoning demand* r\*\_ε: the least costly configured reasoning
setting whose accuracy is within ε = 3 percentage points of the best attainable
accuracy for a task–model pair. We ask whether demand is determined by nominal
difficulty, model capability, or task structure, using one frontier model on
530 development problems from six benchmarks, two controlled task families,
and a prospective validation on 300 unseen problems.

## Repository layout

```text
code/   All source code: data pipeline, evaluators, analysis, plotting, tests, configs
data/   Published processed results: manuscript tables, figures, item-level data, run summaries
docs/   Manuscript draft, dataset documentation, experimental protocol
work/   Runtime artifacts (downloaded data, run logs, caches) — created when you run the pipeline; gitignored
```

The two folders that matter for inspection are **`code/`** (everything needed
to run or understand the pipeline) and **`data/`** (everything needed to verify
the numbers in the manuscript).

## Data policy

The six public benchmark datasets (MATH-500, ZebraLogicBench grid, Easy2Hard-Bench
AMC, AIME 2024, GPQA Diamond, LiveCodeBench) are **not** included in this
repository. They are fetched automatically at pinned HuggingFace revisions by

```bash
python code/scripts/download_datasets.py
```

Full provenance, revision hashes and licenses: [docs/DATASETS.md](docs/DATASETS.md).
Synthetic mechanism and prospective item sets are generated from fixed seeds by
the setup scripts in `code/scripts/` and therefore also need no shipping data.

## Key results

All numbers below are taken from `data/tables/*.json`; the file that backs each
table is listed in `data/README.md`. Figures 1–3 (SVG/PDF/PNG) are in
`data/figures/`.

### Benchmark overview (development set, flash model, ε = 3 pp)

| Benchmark | n | Accuracy low / high / max | r\*\_ε | ARR vs max |
| --- | ---: | --- | --- | ---: |
| MATH-500 | 100 | 81.0% / 83.6% / 79.6% | `low` | 50.6% |
| E2H-AMC | 100 | 75.0% / 78.2% / 76.4% | `high` | 36.0% |
| AIME 2024 | 30 | 96.7% / 94.7% / 96.7% | `low` | 53.0% |
| LiveCodeBench | 100 | 36.8% / 35.8% / 37.2% | `low` (easy/med), `max` (hard) | 68.5% |
| ZebraLogic grid | 100 | 10.6% / 14.4% / 11.8% | `max` (floor noise) | ≈0% |
| GPQA Diamond | 100 | 29.6% / 25.4% / 29.8% | `low` (floor control) | 51.7%† |

† Token-side only; near-chance accuracy at every setting.

### Structural controls (synthetic task families)

| Dimension | Sufficient setting r\*\_ε | Evidence |
| --- | --- | --- |
| Constraint density (clues 4/8/12) | moves `low` → `high` at 12 clues | `data/tables/mechanism_summary.json` |
| Search width (B = 2/4/8) | moves `low` → `high` at B = 8 | `data/tables/searchwidth_mid.json` |
| Sequential depth (8/16/24) | no shift (expenditure only: 315 → 909 → 1,853 tokens) | `data/tables/mechanism_summary.json` |
| Distractor load (0/2/4) | no shift (weak expenditure effect) | `data/tables/mechanism_summary.json` |
| State-tracking load (k = 2/4/8) | no shift (expenditure only: 591 → 1,404 → 2,793 tokens) | `data/tables/statetrack_mid.json` |

### Prospective validation (300 unseen problems, frozen policy, 3 repetitions)

| Family | n | ΔAcc (router − high) | One-sided 95% CI lower | Token saving | Identical / better / worse |
| --- | ---: | ---: | ---: | ---: | ---: |
| MATH-500 | 100 | −2.0 pp | −4.0 pp | −0.4% | 98 / 0 / 2 |
| E2H-AMC | 100 | −1.0 pp | −3.0 pp | 13.5% | 99 / 0 / 1 |
| ZebraLogic | 50 | +2.0 pp | 0.0 pp | 18.5% | 49 / 1 / 0 |
| LiveCodeBench | 50 | +4.0 pp | 0.0 pp | 49.7% | 48 / 2 / 0 |
| **Pooled** | **300** | **0.0 pp** | **−1.3 pp** | **31.9%** | **294 / 3 / 3** |

### Small-sample calibration protocol (stratum-level, safe default)

Pass rate / median token saving over 200 calibration draws, as reported in the
manuscript (Supplementary Table 1); exact per-draw values:
`data/tables/calibration_simulation.json`.

| Family | K = 10 | K = 20 | K = 30 |
| --- | --- | --- | --- |
| LiveCodeBench | 100% / 77.7% | 100% / 56.8% | 100% / 59.8% |
| GPQA Diamond | 100% / 24.1% | 99% / 24.6% | 96% / 25.1% |
| AIME | 100% / 29.0% | 67%† / 29.4% | — |
| MATH-500 | 91% / 0% | 95% / 0% | 96% / 0% |
| E2H-AMC | 36% / 18.5% | 52% / 14.9% | 62% / 13.4% |
| ZebraLogic | 17% / 29.2% | 42% / 21.1% | 57% / 20.5% |

† K = 20 leaves only 10 holdout items; the low pass rate reflects
certification power rather than calibration failure.

## Reproduction

### 1. Environment

Python 3.12 with:

```bash
pip install -r code/requirements.txt
```

Set your API key via the environment variable (never commit keys):

```bash
export DEEPSEEK_API_KEY=...
```

### 2. Fetch and prepare data

```bash
python code/scripts/download_datasets.py
python code/scripts/prepare_datasets.py --pilot-limit 100 --pilot-seed 42
python code/scripts/dataset_stats.py
```

### 3. Tests

```bash
python -m pytest code/tests -q
```

Tests that read downloaded data require step 2 to be completed first.

### 4. Re-run experiments (requires API budget)

```bash
python code/scripts/run_experiment.py --config code/configs/pilot_v1.yaml
```

### 5. Regenerate tables and figures

```bash
python code/scripts/analyze_flash_final.py
python code/scripts/paper_single_source.py
python code/scripts/make_figures.py
```

Runtime outputs (run logs, caches, regenerated tables) are written to `work/`
and are gitignored; the published snapshots in `data/` remain the reference
for the manuscript.

## Notes for reviewers

- `data/tables/*.json` is the single source of truth for every number in the
  manuscript; each file lists its generating script in `data/README.md`.
- Raw API generation logs (tens of thousands of JSONL rows) are not shipped;
  they can be regenerated with the pipeline or requested from the authors.
- Statistical conventions (ε = 3 pp, one-sided item-level bootstrap, paired
  non-inferiority) are defined in the Methods of `docs/NMI_MAIN_TEXT.md` and
  implemented in `code/src/reasoning_efficiency/stats.py`.

## License

Code: MIT (see `LICENSE`). Public benchmark data retain their original
licenses (see `docs/DATASETS.md`). Manuscript text and published result tables:
CC-BY-4.0.

# Sufficient Reasoning Demand: Difficulty, Capability, or Task Structure?

Companion code and data for the manuscript
*"Task structure—not difficulty alone—determines how much reasoning a language
model needs"* (submitted to *Nature Machine Intelligence*).

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

## Key results (all numbers in the manuscript come from `data/tables/`)

| Result | Value | File |
| --- | --- | --- |
| Prospective pooled accuracy difference vs always-`high` | 0.0 pp (one-sided 95% lower bound −1.3 pp ≥ −3 pp) | `data/tables/prospective_validation.json` |
| Prospective token saving vs always-`high` | 31.9% (294/300 items identical) | `data/tables/prospective_validation.json` |
| Development rule vs always-`high` | +1.1 pp, tokens −33.7% | `data/tables/dev_router_v3.json` |
| Benchmark-level available token reduction (ARR) vs always-`max` | 36.0–68.5% (floor benchmarks excluded) | `data/tables/paper_tables.json`, `stratum_demand.json` |
| Structural positive controls | constraint density and search width shift r\*\_ε `low`→`high` | `data/tables/mechanism_summary.json`, `searchwidth_mid.json` |
| Structural negative controls | depth / distractor / state-tracking load scale tokens but not r\*\_ε | `data/tables/mechanism_summary.json`, `statetrack_mid.json` |

Figures 1–3 (SVG/PDF/PNG): `data/figures/`.

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

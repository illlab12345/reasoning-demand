# code/

All source code for the project. Run every script from the repository root so
that `ROOT` resolves correctly (scripts are layout-agnostic: they resolve the
repo root as `parents[2]` and read/write `data/` for published outputs and
`work/` for runtime artifacts).

## Layout

```text
src/reasoning_efficiency/   core library
  adapters/                 API adapters (DeepSeek/OpenAI-compatible, mock)
  loaders/                  per-benchmark loaders (MATH-500, ZebraLogic, E2H, AIME, GPQA, LCB)
  eval/                     deterministic evaluators (symbolic, numeric, grid, multiple-choice, code execution)
  stats.py                  bootstrap, non-inferiority, sufficient-demand computations
  runner.py                 experiment runner (settings, repetitions, caching)
scripts/                    pipeline entry points (41 scripts, see below)
configs/                    YAML configuration (datasets, models, prompts, pricing, experiment stages)
tests/                      unit and integrity tests
```

## Pipeline stages

| Stage | Entry point |
| --- | --- |
| Download pinned public datasets | `scripts/download_datasets.py` |
| Convert to unified JSONL schema + pilot samples | `scripts/prepare_datasets.py`, `scripts/create_pilot_samples.py`, `scripts/expand_samples_to_100.py` |
| Generate synthetic mechanism/prospective items | `scripts/p1_probe_setup.py`, `scripts/p1_searchwidth_setup.py`, `scripts/p1_full_setup.py`, `scripts/create_statetrack_items.py` |
| Run model experiments | `scripts/run_experiment.py`, `scripts/run_p1_probe.py`, `scripts/probe_model_capabilities.py` |
| Build item-level datasets from run logs | `scripts/build_full_pilot_dataset.py`, `scripts/build_sample_level_all.py`, `scripts/build_router_dataset.py` |
| Analysis → manuscript tables | `scripts/analyze_flash_final.py`, `scripts/analyze_full_pilot.py`, `scripts/paper_single_source.py`, `scripts/stratum_demand.py`, `scripts/robustness_analysis.py`, `scripts/compute_merged_tables.py`, `scripts/compute_wasted_tokens.py`, `scripts/compute_oracle_router.py`, `scripts/simulate_calibration.py`, `scripts/summarize_*.py` |
| Figures | `scripts/make_figures.py` (uses `scripts/figure_style.py`) |
| Integrity / tests | `scripts/run_integrity_tests.py`, `python -m pytest tests` |

## Configuration

- `configs/datasets.yaml` — dataset registry with pinned HuggingFace revisions
- `configs/models.yaml` — model/provider registry (key via `DEEPSEEK_API_KEY`)
- `configs/prompts.yaml` — prompt templates
- `configs/pricing.yaml` — token pricing for cost estimates
- `configs/experiment.yaml`, `configs/pilot_v1.yaml`, `configs/p1_*.yaml` — experiment stage definitions

## API access

The API key is read from the environment variable `DEEPSEEK_API_KEY`
(see `src/reasoning_efficiency/adapters/openai_compatible.py`). Never commit
keys; the repository ignores `deepseek_key`, `*.key` and `.env`.

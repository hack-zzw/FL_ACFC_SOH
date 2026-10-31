# Battery SOH paper reproduction

Reproduce the current paper method, comparisons, ablations and external evaluations from the supplied processed inputs and saved models:

```text
conda activate pytorch
python main.py
```

The default command runs model inference, computes statistics from the newly generated predictions, draws Figs 2–4, and checks agreement with the frozen evidence. All new files go to `results/generated/`. Inputs and checkpoints are read-only. You may launch the entry point by absolute path from another working directory.

## Environment

The validated environment is Windows, Python 3.13.9 and PyTorch 2.9.0+cu128. Full main-model replay requires CUDA. To create a separate environment, use `conda env create -f environment.yml`, then `conda activate soh-paper-repro`. The environment has been recorded and reused; a fresh dependency installation has not been tested. See `configs/environment/`.

## Entry points

| Command | Purpose |
|---|---|
| `python main.py` | All saved-model replays, paper statistics and plots |
| `python train.py TASK` | Available source, head, comparison and adaptation training tasks |
| `python plot.py` | Redraw figures from a completed local run |

Use `python main.py --help` and `python train.py controls --help` for options. `main.py --suite main --strict-main` checks exact equality of the three main prediction components. Other suites are `native`, `readouts`, `controls`, `ablations`, `external` and `all`. Partial suites do not compute the complete paper summary. `main.py --verify-only` checks packaged file hashes; `--no-figures` omits rendering.

## Layout

- `configs/`: locked configurations, cohort/experiment indices and file checksums.
- `data/`: processed battery observations and observable features, prepared evaluation inputs and external query identities.
- `dataloader/`: loading, normalization and feature construction.
- `models/`: encoder, federated training, domain heads/inference, controlled comparisons and external adaptation.
- `utils/`: numerical reporting, replay orchestration, plotting and command support.
- `checkpoints/`: fixed trained states; main and control paths are flattened.
- `results/`: `reference/` for numerical comparisons, `source_data/` for released numerical data, and `generated/` for new runs.

The main implementation is in `models/encoder.py`, `models/federated.py`, `models/adaptation.py` and `models/domain/inference.py`. Representative battery identities and measured support observations are fixed study inputs in `data/figure_selection/`. Every displayed model prediction and accuracy value is generated in the current run; frozen predictions are used for verification only.

Generated outputs include `metrics/paper/`, `source_data/Table1.csv`, `source_data/Fig2.csv.gz`–`Fig4.csv.gz`, `figures/`, `validation/` and `run_summary.json`. Figs 1 and 5 are conceptual diagrams supplied as static vector graphics in `results/reference/figures/`.

## Evaluation definition

The source pool contains 303 batteries in 12 clients; the main target cohort contains 75 batteries, 20 labeled support records per battery, and 58,214 query records. Each of the three components completes adaptation and postprocessing before its prediction is averaged equally across all five domains. Main pooled MAE is 0.009347051198636335 and pooled R² is 0.963488486334106; battery-macro R² is 0.9182293056922992. SOH is a ratio; multiplying MAE by 100 yields SOH percentage points. MAPE is reported as a percentage. Protocol and result mappings are in `configs/method_lock.json` and `configs/result_map.csv`.

## Training

```text
python train.py source --seed 20260903
python train.py heads --domain NCA --seed 20260903
python train.py readouts --scope domain --seed 20260903
python train.py controls --all
python train.py comparisons --configuration fedavg --seed 20260903
python train.py external --retrain --dataset NASA --budget 20 --seed 20260903
python train.py external --retrain --dataset Oxford --budget 20 --seed 20260903
```

Source descriptors and embeddings are rebuilt automatically into `results/generated/feature_cache/`. `python train.py prepare` runs feature reconstruction explicitly. The `source` task fits the source encoder and initial domain states; `heads` fits registered initial conditional heads. Training all final domain residuals and routing states from scratch is not yet exposed as one integrated task. Their fitted states and complete inference implementation are supplied, covering the saved-model reproduction of all paper numerical experiments.

## Data and release information

See `data/README.md` for original sources, access links and processing scope, and `THIRD_PARTY_NOTICES.md` for third-party code attribution. Redistribution follows the original data-source conditions. Author names, the project code licence, public repository and DOI remain pending in `configs/release_metadata.json`. `AVAILABILITY.md` contains statement drafts. See `README_中文.md` for the Chinese guide.

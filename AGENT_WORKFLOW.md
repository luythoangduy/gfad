# AGENT_WORKFLOW

Read this file before making changes.

This repository is a research repo. Keep changes small, reproducible, and easy to audit.

## Repo Rules

- Keep the Kaggle notebook thin.
- Keep core logic in `src/` and the existing `foundad/` package structure unless a refactor is explicitly requested.
- Keep entrypoints in `scripts/`.
- Keep configs in `configs/`.
- The dataset already exists on Kaggle. Do not upload, duplicate, or commit dataset files.
- Never commit secrets, `.env`, `kaggle.json`, datasets, checkpoints, or large outputs.
- Use local `.env` profiles for Kaggle credentials.
- Push notebooks through the Kaggle CLI.
- Pull only whitelisted result files by default.
- Track experiments in `experiments/README.md`.

## Kaggle Workflow

- Pull the existing Kaggle notebook into `notebooks/kaggle_runner/`.
- Preserve the real Kaggle kernel ID in `kernel-metadata.json`.
- Update only the safe metadata fields when needed: `code_file`, `dataset_sources`, `enable_gpu`, `enable_internet`, and `is_private`.
- Treat Kaggle notebook metadata as the source of attached dataset links. Do not duplicate dataset attachment logic in notebook cells.
- Keep notebook code limited to environment setup, repo bootstrap, and a script invocation such as `python scripts/run_experiment.py --config configs/baseline.yaml`.
- Do not assume every Kaggle notebook can or should run `pip install -r requirements.txt`; dependency installation should stay optional.
- Write experiment artifacts to `/kaggle/working/outputs/<experiment_id>/`.
- Do not pull checkpoints unless explicitly requested.

## Safety Rules

- Never print Kaggle API keys.
- Never put secrets inside notebooks.
- Never make a Kaggle notebook public unless explicitly requested.
- Do not refactor model logic unless necessary for the requested workflow.

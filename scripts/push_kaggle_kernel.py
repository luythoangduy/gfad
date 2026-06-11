from __future__ import annotations

import argparse
from pathlib import Path

from kaggle_common import (
    DEFAULT_RUNNER_DIR,
    KaggleWorkflowError,
    build_kaggle_env,
    ensure_kaggle_cli,
    load_json,
    dump_json,
    print_profile_selection,
    run_command,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push notebooks/kaggle_runner back to Kaggle using the current kernel metadata.")
    parser.add_argument("--profile", default="main", choices=["main", "alt1", "alt2"])
    parser.add_argument(
        "--runner-dir",
        default=str(DEFAULT_RUNNER_DIR),
        help="Local notebook runner directory. Defaults to notebooks/kaggle_runner.",
    )
    return parser.parse_args()


def validate_metadata(metadata_path: Path) -> None:
    metadata = load_json(metadata_path)
    kernel_id = metadata.get("id", "")
    if not kernel_id or kernel_id.startswith("your_kaggle_username/"):
        raise KaggleWorkflowError(
            "kernel-metadata.json still contains a placeholder kernel id. Pull the real notebook first or update the metadata before pushing."
        )
    if metadata.get("code_file") != "notebook.ipynb":
        raise KaggleWorkflowError(
            "kernel-metadata.json must point to notebook.ipynb before pushing."
        )
    cleaned_sources = [source for source in metadata.get("dataset_sources", []) if source]
    if cleaned_sources != metadata.get("dataset_sources", []):
        metadata["dataset_sources"] = cleaned_sources
        dump_json(metadata_path, metadata)


def main() -> None:
    args = parse_args()
    runner_dir = Path(args.runner_dir).resolve()
    metadata_path = runner_dir / "kernel-metadata.json"

    try:
        kaggle_bin = ensure_kaggle_cli()
        if not metadata_path.exists():
            raise KaggleWorkflowError(
                f"Missing {metadata_path}. Pull the Kaggle notebook first."
            )
        validate_metadata(metadata_path)
        env, username, auth_mode = build_kaggle_env(args.profile)
        print_profile_selection(args.profile, username, auth_mode)
        run_command([kaggle_bin, "kernels", "push", "-p", str(runner_dir)], env=env)
        print(f"Pushed Kaggle kernel from {runner_dir}")
    except KaggleWorkflowError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()

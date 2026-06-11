from __future__ import annotations

import argparse
from pathlib import Path

from kaggle_common import (
    DEFAULT_RUNNER_DIR,
    KaggleWorkflowError,
    build_kaggle_env,
    ensure_kaggle_cli,
    normalize_kernel_metadata,
    print_profile_selection,
    run_command,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull an existing Kaggle notebook and metadata into notebooks/kaggle_runner.")
    parser.add_argument("--profile", default="main", choices=["main", "alt1", "alt2"])
    parser.add_argument("--kernel", required=True, help="Kaggle kernel reference like owner/kernel-slug")
    parser.add_argument(
        "--runner-dir",
        default=str(DEFAULT_RUNNER_DIR),
        help="Local notebook runner directory. Defaults to notebooks/kaggle_runner.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner_dir = Path(args.runner_dir).resolve()
    runner_dir.mkdir(parents=True, exist_ok=True)

    try:
        kaggle_bin = ensure_kaggle_cli()
        env, username, auth_mode = build_kaggle_env(args.profile)
        print_profile_selection(args.profile, username, auth_mode)
        run_command(
            [kaggle_bin, "kernels", "pull", args.kernel, "-p", str(runner_dir), "-m"],
            env=env,
        )
        metadata_path = runner_dir / "kernel-metadata.json"
        if not metadata_path.exists():
            raise KaggleWorkflowError(
                f"Expected pulled metadata at {metadata_path}, but the file was not created."
            )
        normalize_kernel_metadata(metadata_path)
        print(f"Pulled Kaggle kernel into {runner_dir}")
    except KaggleWorkflowError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()

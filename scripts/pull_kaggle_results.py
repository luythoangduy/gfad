from __future__ import annotations

import argparse
import re
import shutil
import tempfile
from pathlib import Path

from kaggle_common import (
    KaggleWorkflowError,
    REPO_ROOT,
    build_kaggle_env,
    ensure_kaggle_cli,
    print_profile_selection,
    run_command,
)


DEFAULT_FILES = [
    "metrics.json",
    "config.yaml",
    "git_commit.txt",
    "summary.csv",
    "result_table.csv",
    "log.txt",
    "AD_eval.csv",
    "train.csv",
    "params.yaml",
]
OPTIONAL_SMALL_EXTENSIONS = [".json", ".csv", ".txt", ".md", ".png", ".jpg", ".jpeg"]
CHECKPOINT_SUFFIXES = [".pth", ".pt", ".ckpt"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull selected Kaggle result files into experiments/results/<experiment_id>.")
    parser.add_argument("--profile", default="main", choices=["main", "alt1", "alt2"])
    parser.add_argument("--kernel", required=True, help="Kaggle kernel reference like owner/kernel-slug")
    parser.add_argument("--experiment_id", required=True, help="Experiment folder under /kaggle/working/outputs/")
    parser.add_argument(
        "--include-optional-small-files",
        action="store_true",
        help="Also pull extra small files such as png, md, txt, csv, and json.",
    )
    parser.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Explicitly allow checkpoint downloads.",
    )
    return parser.parse_args()


def build_file_pattern(experiment_id: str, include_optional_small_files: bool, include_checkpoints: bool) -> str:
    escaped_id = re.escape(experiment_id)
    exact_names = "|".join(re.escape(name) for name in DEFAULT_FILES)
    optional_extensions = "|".join(ext.lstrip(".") for ext in OPTIONAL_SMALL_EXTENSIONS)
    checkpoint_extensions = "|".join(ext.lstrip(".") for ext in CHECKPOINT_SUFFIXES)

    fragments = [rf".*/outputs/{escaped_id}/({exact_names})$"]
    if include_optional_small_files:
        fragments.append(rf".*/outputs/{escaped_id}/.*\.({optional_extensions})$")
    if include_checkpoints:
        fragments.append(rf".*/outputs/{escaped_id}/(checkpoint.*|.*\.({checkpoint_extensions}))$")
    return "|".join(fragments)


def flatten_results(download_root: Path, experiment_id: str, destination: Path) -> int:
    copied = 0
    destination.mkdir(parents=True, exist_ok=True)
    output_marker = f"outputs{Path.sep}{experiment_id}"

    for candidate in download_root.rglob("*"):
        if not candidate.is_file():
            continue

        parts = candidate.parts
        if "outputs" in parts:
            outputs_index = parts.index("outputs")
            if outputs_index + 1 < len(parts) and parts[outputs_index + 1] == experiment_id:
                relative_parts = parts[outputs_index + 2 :]
                target = destination / Path(*relative_parts) if relative_parts else destination / candidate.name
            else:
                continue
        else:
            target = destination / candidate.name

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, target)
        copied += 1

    return copied


def main() -> None:
    args = parse_args()
    destination = REPO_ROOT / "experiments" / "results" / args.experiment_id
    pattern = build_file_pattern(
        args.experiment_id,
        include_optional_small_files=args.include_optional_small_files,
        include_checkpoints=args.include_checkpoints,
    )

    try:
        kaggle_bin = ensure_kaggle_cli()
        env, username, auth_mode = build_kaggle_env(args.profile)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        print_profile_selection(args.profile, username, auth_mode)
        with tempfile.TemporaryDirectory(prefix="kaggle-results-") as temp_dir:
            temp_path = Path(temp_dir)
            run_command(
                [
                    kaggle_bin,
                    "kernels",
                    "output",
                    args.kernel,
                    "-p",
                    str(temp_path),
                    "--file-pattern",
                    pattern,
                ],
                env=env,
            )
            copied = flatten_results(temp_path, args.experiment_id, destination)
        print(f"Copied {copied} result file(s) into {destination}")
    except KaggleWorkflowError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()

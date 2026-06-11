from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from kaggle_common import REPO_ROOT, safe_git_commit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thin experiment wrapper for local and Kaggle runs.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    project = config.get("project", {})
    run_cfg = config.get("run", {})
    data_cfg = config.get("data", {})
    output_cfg = config.get("output", {})

    exp_name = project.get("exp_name")
    if not exp_name:
        raise SystemExit("Config is missing project.exp_name.")

    output_root = Path(output_cfg.get("dir", "outputs")).resolve()
    experiment_dir = output_root / exp_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    with (experiment_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    (experiment_dir / "git_commit.txt").write_text(safe_git_commit() + "\n", encoding="utf-8")

    devices = run_cfg.get("devices", ["cuda:0"])
    device_override = "[" + ",".join(str(device) for device in devices) + "]"
    app_name = run_cfg.get("app", "train_dinov3")
    diy_name = run_cfg.get("diy_name", f"_{exp_name}")
    mode = run_cfg.get("mode", "train")

    command = [
        sys.executable,
        str(REPO_ROOT / "foundad" / "main.py"),
        f"mode={mode}",
        f"app={app_name}",
        f"diy_name={diy_name}",
        f"devices={device_override}",
        f"data.dataset={data_cfg.get('dataset', 'mvtec')}",
        f"data.data_name={data_cfg.get('data_name', 'mvtec_1shot')}",
        f"data.data_path={data_cfg.get('data_path', '/kaggle/input/dataset_slug')}",
        f"data.test_root={data_cfg.get('test_root', data_cfg.get('data_path', '/kaggle/input/dataset_slug'))}",
        f"app.logging.folder={experiment_dir.as_posix()}",
    ]

    log_path = experiment_dir / "log.txt"
    start_time = datetime.now(timezone.utc)
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"Started: {start_time.isoformat()}\n")
        log_handle.write(f"Command: {' '.join(command)}\n\n")
        log_handle.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        end_time = datetime.now(timezone.utc)
        log_handle.write(f"\nFinished: {end_time.isoformat()}\n")
        log_handle.write(f"Return code: {completed.returncode}\n")

    metrics_payload = {
        "project": project.get("name", "unknown"),
        "experiment_id": exp_name,
        "seed": project.get("seed"),
        "status": "completed" if completed.returncode == 0 else "failed",
        "return_code": completed.returncode,
        "started_at": start_time.isoformat(),
        "finished_at": end_time.isoformat(),
        "log_file": "log.txt",
    }
    write_json(experiment_dir / "metrics.json", metrics_payload)

    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()

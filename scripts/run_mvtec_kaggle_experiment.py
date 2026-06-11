from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a full MVTec Kaggle train/eval experiment.")
    parser.add_argument("--config", required=True, help="Experiment YAML under configs/.")
    return parser.parse_args()


def run(command: list[str], *, cwd: Path, log_handle) -> None:
    log_handle.write(f"$ {' '.join(command)}\n")
    log_handle.flush()
    subprocess.run(command, cwd=cwd, stdout=log_handle, stderr=subprocess.STDOUT, text=True, check=True)


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_metrics_summary(eval_csv: Path, output_path: Path, config: dict, status: str) -> None:
    rows = []
    if eval_csv.exists():
        with eval_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

    mean_row = next((row for row in rows if row.get("class") == "Mean"), {})
    focus_rows = {row.get("class"): row for row in rows if row.get("class") in {"screw", "capsule"}}
    payload = {
        "experiment_id": config["project"]["exp_name"],
        "status": status,
        "git_commit": git_commit(),
        "mean": mean_row,
        "screw": focus_rows.get("screw", {}),
        "capsule": focus_rows.get("capsule", {}),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def hydra_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(hydra_value(item) for item in value) + "]"
    return str(value)


def flatten_hydra_overrides(prefix: str, payload: dict) -> list[str]:
    overrides = []
    for key, value in payload.items():
        path = f"{prefix}.{key}"
        if isinstance(value, dict):
            overrides.extend(flatten_hydra_overrides(path, value))
        else:
            overrides.append(f"{path}={hydra_value(value)}")
    return overrides


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_yaml(config_path)

    exp_name = config["project"]["exp_name"]
    run_cfg = config["run"]
    data_cfg = config["data"]
    kaggle_cfg = config["kaggle"]
    output_root = Path(config.get("output", {}).get("dir", "/kaggle/working/outputs"))
    output_dir = output_root / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "log.txt"
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"Started: {datetime.now(timezone.utc).isoformat()}\n")

        if kaggle_cfg.get("install_deps", False):
            run([sys.executable, "-m", "pip", "install", "hydra-core", "torchmetrics"], cwd=REPO_ROOT, log_handle=log_handle)

        sample_target = Path(kaggle_cfg["sample_target"])
        if not sample_target.exists():
            run(
                [
                    sys.executable,
                    "foundad/src/sample.py",
                    f"source={kaggle_cfg['dataset_root']}",
                    f"target={sample_target}",
                    f"seed={config['project'].get('seed', 0)}",
                    f"num_samples={kaggle_cfg.get('num_samples', 2)}",
                ],
                cwd=REPO_ROOT,
                log_handle=log_handle,
            )

        devices = "[" + ",".join(run_cfg.get("devices", ["cuda:0"])) + "]"
        data_overrides = data_cfg.get("augmentation_overrides", {})
        override_path = output_dir / "augmentation_overrides.yaml"
        override_path.write_text(yaml.safe_dump(data_overrides, sort_keys=False), encoding="utf-8")

        common_overrides = [
            f"data.dataset={data_cfg.get('dataset', 'mvtec')}",
            f"data.data_name={data_cfg.get('data_name', 'mvtec_tmp')}",
            f"data.data_path={data_cfg.get('data_path', '/kaggle/working')}",
            f"data.test_root={data_cfg.get('test_root', kaggle_cfg['dataset_root'])}",
            "data.num_workers=4",
            f"diy_name={run_cfg['diy_name']}",
            f"devices={devices}",
            f"dist.backend={run_cfg.get('dist_backend', 'gloo')}",
            "testing.segmentation_vis=False",
        ]

        train_cmd = [
            sys.executable,
            "foundad/main.py",
            "mode=train",
            f"app={run_cfg.get('app', 'train_dinov3')}",
            *common_overrides,
        ]
        if data_overrides:
            train_cmd.extend(flatten_hydra_overrides("data.augmentation_overrides", data_overrides))
        run(train_cmd, cwd=REPO_ROOT, log_handle=log_handle)

        eval_cmd = [
            sys.executable,
            "foundad/main.py",
            "mode=AD",
            "app=test",
            f"app.ckpt_step={run_cfg.get('ckpt_step', 2000)}",
            *common_overrides,
        ]
        run(eval_cmd, cwd=REPO_ROOT, log_handle=log_handle)

    log_dir = REPO_ROOT / "logs" / data_cfg.get("data_name", "mvtec_tmp") / f"dinov3{run_cfg['diy_name']}"
    eval_dir = log_dir / "eval" / str(run_cfg.get("ckpt_step", 2000))
    copy_if_exists(log_dir / "params.yaml", output_dir / "params.yaml")
    copy_if_exists(log_dir / "train.csv", output_dir / "train.csv")
    copy_if_exists(eval_dir / "AD_eval.csv", output_dir / "AD_eval.csv")
    shutil.copy2(config_path, output_dir / "config.yaml")
    (output_dir / "git_commit.txt").write_text(git_commit() + "\n", encoding="utf-8")
    write_metrics_summary(output_dir / "AD_eval.csv", output_dir / "metrics.json", config, "completed")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import site
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNNER_DIR = REPO_ROOT / "notebooks" / "kaggle_runner"


class KaggleWorkflowError(RuntimeError):
    """Raised for user-actionable Kaggle workflow errors."""


def _profile_key(profile: str, suffix: str) -> str:
    return f"KAGGLE_{profile.upper()}_{suffix}"


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        raise KaggleWorkflowError(
            f"Missing {path}. Copy .env.example to .env and fill in Kaggle credentials locally."
        )

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        values[key] = value
    return values


def resolve_profile(profile: str, env_values: dict[str, str]) -> dict[str, str]:
    api_token = env_values.get(_profile_key(profile, "API_TOKEN"), "").strip()
    access_token = env_values.get(_profile_key(profile, "ACCESS_TOKEN"), "").strip()
    username = env_values.get(_profile_key(profile, "USERNAME"), "").strip()
    api_key = env_values.get(_profile_key(profile, "KEY"), "").strip()

    if api_token or access_token:
        token = api_token or access_token
        return {
            "auth_mode": "api_token",
            "username": username or "(resolved by Kaggle token)",
            "api_token": token,
        }

    if username and api_key:
        return {
            "auth_mode": "legacy_key",
            "username": username,
            "api_key": api_key,
        }

    raise KaggleWorkflowError(
        "Profile "
        f"'{profile}' is incomplete in .env. Provide either "
        f"{_profile_key(profile, 'API_TOKEN')} (preferred) or the legacy pair "
        f"{_profile_key(profile, 'USERNAME')} and {_profile_key(profile, 'KEY')}."
    )


def build_kaggle_env(profile: str, dotenv_path: Path | None = None) -> tuple[dict[str, str], str, str]:
    env_values = load_dotenv(dotenv_path or REPO_ROOT / ".env")
    profile_config = resolve_profile(profile, env_values)
    env = os.environ.copy()
    env["KAGGLE_PROFILE"] = profile
    env.pop("KAGGLE_USERNAME", None)
    env.pop("KAGGLE_KEY", None)
    env.pop("KAGGLE_API_TOKEN", None)

    auth_mode = profile_config["auth_mode"]
    username = profile_config["username"]
    if auth_mode == "api_token":
        env["KAGGLE_API_TOKEN"] = profile_config["api_token"]
    else:
        env["KAGGLE_USERNAME"] = username
        env["KAGGLE_KEY"] = profile_config["api_key"]
    return env, username, auth_mode


def ensure_kaggle_cli() -> str:
    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin:
        return kaggle_bin

    candidates = []
    executable_dir = Path(sys.executable).resolve().parent
    candidates.append(executable_dir / "Scripts" / "kaggle.exe")
    candidates.append(executable_dir / "Scripts" / "kaggle")

    try:
        user_base = Path(site.getuserbase())
        candidates.append(user_base / "Scripts" / "kaggle.exe")
        candidates.append(user_base / "Scripts" / "kaggle")
    except Exception:
        pass

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise KaggleWorkflowError(
        "Kaggle CLI is not installed or not on PATH. Install it with `pip install kaggle` and make sure the executable is available."
    )


def run_command(command: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    try:
        subprocess.run(command, check=True, env=env, cwd=cwd or REPO_ROOT)
    except subprocess.CalledProcessError as exc:
        raise KaggleWorkflowError(
            f"Command failed with exit code {exc.returncode}: {' '.join(command)}"
        ) from exc


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def normalize_kernel_metadata(
    metadata_path: Path,
    *,
    code_file_name: str = "notebook.ipynb",
    dataset_sources: Iterable[str] | None = None,
    enable_gpu: bool = True,
    enable_internet: bool = True,
    is_private: bool = True,
) -> dict:
    metadata = load_json(metadata_path)

    original_code_file = metadata.get("code_file", code_file_name)
    if original_code_file != code_file_name:
        original_path = metadata_path.parent / original_code_file
        target_path = metadata_path.parent / code_file_name
        if original_path.exists() and original_path != target_path:
            original_path.replace(target_path)

    metadata["code_file"] = code_file_name
    metadata.setdefault("language", "python")
    metadata.setdefault("kernel_type", "notebook")
    metadata["enable_gpu"] = bool(metadata.get("enable_gpu", enable_gpu))
    metadata["enable_internet"] = bool(metadata.get("enable_internet", enable_internet))
    metadata["is_private"] = bool(metadata.get("is_private", is_private))
    sources = dataset_sources if dataset_sources is not None else metadata.get("dataset_sources", [])
    metadata["dataset_sources"] = [source for source in sources if source]
    metadata.setdefault("competition_sources", [])
    metadata.setdefault("kernel_sources", [])

    dump_json(metadata_path, metadata)
    return metadata


def safe_git_commit() -> str:
    git_bin = shutil.which("git")
    if not git_bin:
        return "unknown"
    try:
        result = subprocess.run(
            [git_bin, "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        return result.stdout.strip() or "unknown"
    except subprocess.CalledProcessError:
        return "unknown"


def print_profile_selection(profile: str, username: str, auth_mode: str) -> None:
    print(f"Selected Kaggle profile: {profile}")
    print(f"Kaggle username: {username}")
    print(f"Auth mode: {auth_mode}")
    if auth_mode == "api_token":
        print("KAGGLE_API_TOKEN loaded for child processes only.")
    else:
        print("KAGGLE_KEY loaded for child processes only.")


def fail(message: str, *, exit_code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(exit_code)

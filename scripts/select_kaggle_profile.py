from __future__ import annotations

import argparse

from kaggle_common import KaggleWorkflowError, build_kaggle_env, print_profile_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve a Kaggle API profile from the local .env file.")
    parser.add_argument("--profile", default="main", choices=["main", "alt1", "alt2"])
    parser.add_argument(
        "--shell",
        choices=["powershell", "bash", "cmd"],
        help="Print safe shell snippets for the selected username. The API key is never printed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        env, username, auth_mode = build_kaggle_env(args.profile)
    except KaggleWorkflowError as exc:
        raise SystemExit(str(exc)) from exc

    print_profile_selection(args.profile, username, auth_mode)
    if args.shell == "powershell":
        print(f"$env:KAGGLE_PROFILE='{args.profile}'")
        if auth_mode == "api_token":
            print("# KAGGLE_API_TOKEN is intentionally not printed.")
        else:
            print(f"$env:KAGGLE_USERNAME='{env['KAGGLE_USERNAME']}'")
            print("# KAGGLE_KEY is intentionally not printed.")
    elif args.shell == "bash":
        print(f"export KAGGLE_PROFILE='{args.profile}'")
        if auth_mode == "api_token":
            print("# KAGGLE_API_TOKEN is intentionally not printed.")
        else:
            print(f"export KAGGLE_USERNAME='{env['KAGGLE_USERNAME']}'")
            print("# KAGGLE_KEY is intentionally not printed.")
    elif args.shell == "cmd":
        print(f"set KAGGLE_PROFILE={args.profile}")
        if auth_mode == "api_token":
            print("REM KAGGLE_API_TOKEN is intentionally not printed.")
        else:
            print(f"set KAGGLE_USERNAME={env['KAGGLE_USERNAME']}")
            print("REM KAGGLE_KEY is intentionally not printed.")


if __name__ == "__main__":
    main()

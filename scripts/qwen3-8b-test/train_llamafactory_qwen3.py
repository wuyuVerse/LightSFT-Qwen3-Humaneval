import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch LLaMA-Factory full-parameter SFT for Qwen3 with FORCE_TORCHRUN"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../../config/llamafactory_sft.yaml",
        help="Path to LLaMA-Factory training config YAML",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="CUDA visible devices for single-node multi-GPU training (comma-separated)",
    )
    parser.add_argument(
        "--extra",
        type=str,
        default="",
        help="Extra CLI args appended to llamafactory-cli train (e.g. '--num_train_epochs 3.0')",
    )
    return parser.parse_args()


def ensure_uv() -> None:
    if shutil.which("uv") is None:
        print(
            "uv not found. Please source your env first: source /volume/pt-train/users/wzhang/wjj-workspace/.zshrc",
            file=sys.stderr,
        )
        sys.exit(127)


def main() -> int:
    args = parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 2

    ensure_uv()

    # Environment setup
    os.environ.setdefault("FORCE_TORCHRUN", "1")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    # Optional niceties
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    
    # Wandb offline mode
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = "../../wandb_logs"
    os.environ["WANDB_PROJECT"] = "LightSFT-Qwen3-Humaneval"
    os.environ["WANDB_RUN_NAME"] = f"qwen3-8b-sft-{args.devices.replace(',', '-')}gpu"

    cmd = [
        "uv",
        "run",
        "llamafactory-cli",
        "train",
        str(config_path),
    ]

    if args.extra.strip():
        cmd.extend(shlex.split(args.extra.strip()))

    print(f"Launching: {' '.join(shlex.quote(x) for x in cmd)}")
    print(f"FORCE_TORCHRUN={os.environ['FORCE_TORCHRUN']} CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

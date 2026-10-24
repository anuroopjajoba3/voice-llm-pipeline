#!/usr/bin/env python3
"""Copy HF checkpoint into Triton model repository weights/."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT / "deploy/triton/model_repository/intent_classifier/1/weights"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "outputs/finetune/best",
        help="HF save_pretrained directory",
    )
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()

    if not args.checkpoint.is_dir():
        raise SystemExit(f"Missing checkpoint: {args.checkpoint}")

    if args.dest.exists():
        shutil.rmtree(args.dest)
    shutil.copytree(args.checkpoint, args.dest)
    print(f"Exported {args.checkpoint} -> {args.dest}")


if __name__ == "__main__":
    main()

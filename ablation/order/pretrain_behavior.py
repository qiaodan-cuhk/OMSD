"""Pretrain the sequential diffusion behavior for a fixed order ablation."""
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omsd.cli import parse_stage_config
from pretrain_behavior import train_behavior


def main(argv=None):
    config = parse_stage_config("behavior", argv, variant="order")
    return train_behavior(config, variant="order")


if __name__ == "__main__":
    main()

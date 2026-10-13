"""Run fixed-order OMSD with the corresponding behavior and a main joint critic."""
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omsd.cli import parse_stage_config
from main import train_policy


def main(argv=None):
    config = parse_stage_config("policy", argv, variant="order")
    return train_policy(config, variant="order")


if __name__ == "__main__":
    main()

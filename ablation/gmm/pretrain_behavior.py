"""Pretrain GMM behavior for ablation; critic pretraining stays in the main workflow."""

from __future__ import annotations

import json
from pathlib import Path
import sys

# Permit ``python ablation/gmm/pretrain_behavior.py`` from any directory.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ablation.gmm.agents import GMMOfflineAgent
from omsd.cli import parse_stage_config
from omsd.runtime import prepare_experiment


def main(argv=None):
    config = parse_stage_config("behavior", argv, variant="gmm")
    with prepare_experiment(config, "behavior", agent_class=GMMOfflineAgent, variant="gmm") as experiment:
        step = 0
        total = config.num_epochs * experiment.batches_per_epoch
        for epoch in range(config.num_epochs):
            for samples in experiment.batch_epoch():
                step += 1
                metrics = experiment.agent.update_behavior(samples)
                metrics["epoch"] = epoch + 1
                experiment.log(step, metrics, final=step == total)
                if step % config.checkpoint_interval == 0 and step != total:
                    experiment.save(step)
        experiment.save(step, final=True)
        result = dict(checkpoint=str(experiment.artifacts.checkpoint_path), step=step,
                      dataset=experiment.dataset_summary)
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    main()

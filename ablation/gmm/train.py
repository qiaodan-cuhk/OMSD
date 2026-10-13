"""Policy training for the GMM ablation using its behavior and a main IQL critic."""

from __future__ import annotations

import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ablation.gmm.agents import GMMOfflineAgent
from omsd.cli import parse_stage_config
from omsd.runtime import prepare_experiment


def main(argv=None):
    config = parse_stage_config("policy", argv, variant="gmm")
    with prepare_experiment(config, "policy", agent_class=GMMOfflineAgent, variant="gmm") as experiment:
        for step in range(1, config.num_steps + 1):
            metrics = experiment.agent.update_policy(experiment.sample())
            if experiment.environment is not None and (
                    step % config.eval_interval == 0 or step == config.num_steps):
                metrics.update(experiment.evaluate())
            experiment.log(step, metrics)
            if step % config.checkpoint_interval == 0 and step != config.num_steps:
                experiment.save(step)
        experiment.save(config.num_steps, final=True)
        result = dict(checkpoint=str(experiment.artifacts.checkpoint_path), step=config.num_steps,
                      dataset=experiment.dataset_summary)
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    main()

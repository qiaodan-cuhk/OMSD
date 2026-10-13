"""OMSD/CTDE/IND policy optimization with frozen diffusion behavior and IQL."""
import json

from omsd.cli import parse_stage_config
from omsd.runtime import prepare_experiment


def train_policy(config, *, variant="main"):
    with prepare_experiment(config, "policy", variant=variant) as experiment:
        for step in range(1, config.num_steps + 1):
            metrics = experiment.agent.update_policy(experiment.sample())
            if experiment.environment is not None and (step % config.eval_interval == 0 or step == config.num_steps):
                metrics.update(experiment.evaluate())
            experiment.log(step, metrics, final=step == config.num_steps)
            if step % config.checkpoint_interval == 0 and step != config.num_steps:
                experiment.save(step)
        experiment.save(config.num_steps, final=True)
        result = dict(checkpoint=str(experiment.artifacts.checkpoint_path), step=config.num_steps,
                      dataset=experiment.dataset_summary)
    print(json.dumps(result))
    return result


def main(argv=None):
    return train_policy(parse_stage_config("policy", argv))


if __name__ == "__main__":
    main()

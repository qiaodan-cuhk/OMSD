"""Diffusion behavior pretraining: OMSD sequential conditions; CTDE/IND independent."""
import json

from omsd.cli import parse_stage_config
from omsd.runtime import prepare_experiment


def train_behavior(config, *, variant="main"):
    with prepare_experiment(config, "behavior", variant=variant) as experiment:
        step = 0
        total = config.num_epochs * experiment.batches_per_epoch
        for epoch in range(config.num_epochs):
            for samples in experiment.batch_epoch():
                step += 1
                metrics = experiment.agent.update_behavior(samples)
                metrics["epoch"] = epoch + 1
                experiment.log(step, metrics, final=step == total)
                # Snapshots use optimizer-step counts; users select the input file.
                if step % config.checkpoint_interval == 0 and step != total:
                    experiment.save(step)
        experiment.save(step, final=True)
        result = dict(checkpoint=str(experiment.artifacts.checkpoint_path), step=step,
                      dataset=experiment.dataset_summary)
    print(json.dumps(result))
    return result


def main(argv=None):
    return train_behavior(parse_stage_config("behavior", argv))


if __name__ == "__main__":
    main()

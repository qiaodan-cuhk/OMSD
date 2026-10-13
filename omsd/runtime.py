"""I/O helpers for independent research training scripts; no shared train loop."""
from contextlib import contextmanager
import json
from pathlib import Path
import random

import numpy as np
import torch

from .agents import OfflineAgent
from .artifacts import prepare_run, write_run_config
from .checkpoints import load_checkpoint, save_checkpoint
from .data import load_dataset
from .config import STAGE_DEFAULTS


def prepare_dataset(config):
    return load_dataset(config.dataset_path(), format=config.dataset_format, agent_ids=config.agent_ids,
        reward_scale=config.reward_scale, episode_length=config.episode_length,
        transition_mode=config.transition_mode, terminal_policy=config.terminal_policy,
        observation_dims=config.observation_dims, action_dims=config.action_dims,
        observation_prefix=config.observation_prefix, max_size=config.max_size)


def describe_dataset(dataset):
    return dict(size=dataset.size, num_agents=dataset.num_agents,
        agent_ids=list(dataset.agent_ids), obs_dims=list(dataset.obs_dims),
        action_dims=list(dataset.action_dims), state_dims=list(dataset.state_dims or []),
        source_format=dataset.source_format, metadata=dataset.metadata)


def inspect(config):
    result = describe_dataset(prepare_dataset(config))
    print(json.dumps(result, indent=2))
    return result


class Experiment:
    """Models, aligned batches and artifacts; stage scripts own optimization loops."""

    def __init__(self, config, stage, agent_class, variant):
        if stage not in {"behavior", "critic", "policy"}:
            raise ValueError("stage must be behavior, critic, or policy")
        if config.variant != variant:
            raise ValueError("Entrypoint and configured variant must match")
        self.config, self.stage, self.variant = config, stage, variant
        if stage in {"behavior", "critic"} and config.steps_per_epoch is None:
            config.steps_per_epoch = STAGE_DEFAULTS[stage]["steps_per_epoch"]
        self.dataset = prepare_dataset(config)
        self.dataset_summary = describe_dataset(self.dataset)
        self.environment, self.writer, self._metrics = None, None, None
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; select --device cpu explicitly")
        if stage == "policy" and (not config.behavior_checkpoint or not config.critic_checkpoint):
            raise ValueError("Policy training requires --behavior-checkpoint and --critic-checkpoint")
        torch.set_num_threads(config.num_threads)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        if config.conditional_order == "random":
            config.conditional_order = np.random.default_rng(config.seed).permutation(self.dataset.num_agents).tolist()
            config.ordered_updates = True
        self.agent = agent_class(config, self.dataset.obs_dims, self.dataset.action_dims,
            self.dataset.state_dims, state_layout=self.dataset.metadata.get("state_layout", "per_agent"),
            stage=stage, agent_ids=self.dataset.agent_ids)
        self.dependencies = {}
        components = ("behavior", "critic", "policy") if stage == "policy" else (stage,)
        for component in components:
            path = getattr(config, component + "_checkpoint", None)
            if path:
                load_checkpoint(path, self.agent, self.dataset, component)
                self.dependencies[component] = ([str(Path(item).expanduser().resolve()) for item in path]
                                                if isinstance(path, (list, tuple)) else str(Path(path).expanduser().resolve()))
        self.artifacts = None

    def start(self):
        if self.stage == "policy" and self.config.eval_interval:
            from .environments import make_environment, load_opponent_policy
            if self.config.agent_ids is None:
                self.config.agent_ids = list(self.dataset.agent_ids)
            self.environment = make_environment(self.config)
            if self.config.opponent_checkpoint:
                self.environment.other_action_fn = load_opponent_policy(self.config.opponent_checkpoint, self.environment)
                self.dependencies["opponent"] = self.config.opponent_checkpoint
            if self.environment.obs_dims != self.agent.obs_dims or self.environment.action_dims != self.agent.action_dims:
                raise ValueError("Environment dimensions differ from the dataset; check backend, partition and observation settings")
            if len(self.environment.agent_ids) != len(self.environment.full_agent_ids) and self.environment.other_action_fn is None:
                raise ValueError("Evaluation of a learner subset requires --opponent-checkpoint")
        self.artifacts = prepare_run(self.config, self.dataset, self.stage, variant=self.variant)
        write_run_config(self.artifacts, self.config, self.dataset_summary, dependencies=self.dependencies)
        self._metrics = self.artifacts.metrics_path.open("x", encoding="utf-8")
        if self.config.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.artifacts.directory / "tensorboard"))

    def batch_epoch(self):
        """Original omiga epoch: fixed updates from aligned buffer sampling."""
        for _ in range(self.config.steps_per_epoch):
            yield self.sample()

    @property
    def batches_per_epoch(self):
        return self.config.steps_per_epoch

    def sample(self):
        return self.dataset.sample(self.config.batch_size, device=self.config.device)

    def log(self, step, metrics, *, final=False):
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Non-finite training metric at step " + str(step))
        evaluation = any(name.startswith("evaluation/") for name in metrics)
        if step == 1 or step % self.config.logging_interval == 0 or final or evaluation:
            record = dict(step=step, **metrics)
            self._metrics.write(json.dumps(record) + "\n")
            self._metrics.flush()
            print(json.dumps(record))
            if self.writer is not None:
                for name, value in metrics.items():
                    self.writer.add_scalar(name, value, step)

    def save(self, step, *, final=False):
        path = self.artifacts.checkpoint_path if final else self.artifacts.directory / (self.stage + "_step" + str(step) + ".pt")
        save_checkpoint(path, self.agent, self.dataset, self.stage, step)
        return path

    def evaluate(self):
        if self.environment is None:
            raise RuntimeError("Evaluation requires a positive eval_interval")
        returns = []
        for _ in range(self.config.eval_episodes):
            obs = self.environment.reset()
            episode_return = 0.0
            for _ in range(self.config.episode_length):
                obs, rewards, dones, _ = self.environment.step(self.agent.act(obs))
                episode_return += float(np.mean(rewards))
                if np.all(dones):
                    break
            returns.append(episode_return)
        return {"evaluation/mean_return": float(np.mean(returns))}

    def close(self):
        if self.environment is not None:
            self.environment.close()
        if self.writer is not None:
            self.writer.close()
        if self._metrics is not None:
            self._metrics.close()


@contextmanager
def prepare_experiment(config, stage, *, agent_class=None, variant="main"):
    experiment = Experiment(config, stage, agent_class or OfflineAgent, variant)
    try:
        experiment.start()
        yield experiment
    finally:
        experiment.close()


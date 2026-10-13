"""Portable experiment configuration, independent of dataset and simulator APIs."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional, List, Union
import math

import yaml

from .specs import get_algorithm_spec
from .profiles import PROFILES

STAGE_DEFAULTS = {
    "behavior": dict(num_epochs=200, steps_per_epoch=10000, batch_size=512),
    "critic": dict(num_epochs=200, steps_per_epoch=5000, batch_size=1024),
    "policy": dict(batch_size=512),
    "inspect": {},
}


@dataclass
class ExperimentConfig:
    profile: str = "mpe"
    env_id: str = "simple_spread"
    backend: str = "mpe"
    agent_conf: Optional[str] = None
    agent_obsk: Optional[int] = None
    discrete_action: bool = False
    episode_length: int = 25
    data_path: Optional[str] = None
    dataset_dir: str = "datasets"
    dataset_format: str = "auto"
    data_type: str = "expert"
    dataset_num: int = 0
    transition_mode: str = "auto"
    terminal_policy: str = "provided"
    observation_prefix: Optional[int] = None
    observation_dims: Optional[List[int]] = None
    action_dims: Optional[List[int]] = None
    agent_ids: Optional[List[int]] = None
    pretrain_agent_id: Optional[int] = None
    reward_scale: float = 1.0
    max_size: Optional[int] = None
    algorithm: str = "OMSD"
    variant: str = "main"
    conditional_order: Optional[List[int]] = None
    ordered_updates: bool = True
    critic_input: str = "obs"
    device: str = "cuda:0"
    seed: int = 42
    num_steps: int = 100000
    num_epochs: int = 200
    steps_per_epoch: Optional[int] = None
    batch_size: int = 512
    gamma: float = 0.99
    tau: float = 0.7
    beta: float = 0.01
    WT: str = "VDS"
    regq: int = 0
    learning_rate: float = 3e-4
    dilac_lr: float = 3e-4
    iql_critic_lr: float = 3e-4
    policy_layer: int = 2
    q_layer: int = 2
    actor_blocks: int = 2
    t_GassProj_dims: int = 32
    t_embed_dims: int = 64
    sa_embed_dims: int = 32
    resnet_hidden_dim: int = 512
    n_policy_epochs: int = 100
    lr_anneal: bool = True
    n_gmm_components: int = 8
    gmm_hidden_dim: int = 512
    gmm_score_normalize: Optional[bool] = None
    output_root: str = "outputs"
    output_dir: Optional[str] = None
    run_id: Optional[str] = None
    behavior_checkpoint: Optional[Union[str, List[str]]] = None
    critic_checkpoint: Optional[Union[str, List[str]]] = None
    policy_checkpoint: Optional[Union[str, List[str]]] = None
    opponent_checkpoint: Optional[str] = None
    checkpoint_interval: int = 10000
    logging_interval: int = 500
    eval_interval: int = 0
    eval_episodes: int = 10
    num_threads: int = 2
    tensorboard: bool = False

    def __post_init__(self):
        # Preserve omiga's unit-normalized GMM guidance. Diffusion entrypoints
        # have no GMM setting; an explicit False still selects raw GMM score.
        if self.gmm_score_normalize is None:
            self.gmm_score_normalize = self.variant == "gmm"

    @property
    def learning_rates(self):
        return self.learning_rate

    @property
    def spec(self):
        return get_algorithm_spec(self.algorithm)

    @property
    def mode(self):
        """Internal decomposition name; derived from the public algorithm."""
        return self.spec.mode

    @property
    def behavior_model(self):
        """Entrypoints explicitly select diffusion or the separate GMM system."""
        return "gmm" if self.variant == "gmm" else "diffusion"

    def to_dict(self):
        return asdict(self)

    def validate(self):
        if self.variant != "order":
            self.ordered_updates = self.spec.mode == "SEQ"
        for name in ("reward_scale", "gamma", "tau", "beta", "learning_rate", "dilac_lr", "iql_critic_lr"):
            value = getattr(self, name)
            try:
                number = float(value)
            except (ValueError, TypeError) as exc:
                raise ValueError(name + " must be a finite number") from exc
            if isinstance(value, bool) or not math.isfinite(number):
                raise ValueError(name + " must be a finite number")
            setattr(self, name, number)
        for name in ("discrete_action", "ordered_updates", "lr_anneal", "tensorboard", "gmm_score_normalize"):
            value = getattr(self, name)
            if isinstance(value, str) and value.lower() in {"true", "false", "1", "0", "yes", "no"}:
                setattr(self, name, value.lower() in {"true", "1", "yes"})
            elif not isinstance(value, bool):
                raise ValueError(name + " must be true or false")
        if isinstance(self.device, int) and not isinstance(self.device, bool) and self.device >= 0:
            self.device = str(self.device)
        if not isinstance(self.device, str):
            raise ValueError("device must be cpu or a CUDA device such as cuda:0")
        self.algorithm = self.spec.name
        if self.variant not in {"main", "gmm", "order"}:
            raise ValueError("variant must be main, gmm, or order")
        if self.variant != "gmm" and self.gmm_score_normalize:
            raise ValueError("gmm_score_normalize is available only through ablation/gmm")
        if self.variant == "order" and self.algorithm != "OMSD":
            raise ValueError("The conditional-order ablation applies to OMSD")
        if self.profile not in PROFILES:
            raise ValueError("Unknown profile: " + self.profile)
        if self.agent_obsk is None:
            self.agent_obsk = 1
        if self.critic_input not in {"obs", "state"}:
            raise ValueError("critic_input must be obs or state")
        if self.dataset_format not in {"auto", "numpy", "omiga", "ogmarl", "hdf5"}:
            raise ValueError("dataset_format must be auto, numpy, omiga, ogmarl, or hdf5")
        if not 0 < self.tau < 1:
            raise ValueError("tau is the IQL expectile and must be between 0 and 1")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        for name in ("batch_size", "episode_length", "num_threads", "num_epochs", "num_steps",
                     "checkpoint_interval", "logging_interval", "eval_episodes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(name + " must be positive")
        if self.eval_interval < 0:
            raise ValueError("eval_interval must be nonnegative")
        if self.steps_per_epoch is not None and (isinstance(self.steps_per_epoch, bool)
                                                or not isinstance(self.steps_per_epoch, int)
                                                or self.steps_per_epoch <= 0):
            raise ValueError("steps_per_epoch must be a positive integer or null")
        if self.variant != "order" and self.conditional_order is not None:
            raise ValueError("conditional_order is available only through ablation/order")
        if isinstance(self.conditional_order, str) and self.conditional_order != "random":
            try:
                self.conditional_order = [int(x) for x in self.conditional_order.replace("-", ",").split(",")]
            except ValueError as exc:
                raise ValueError("conditional_order must be a permutation such as 2-0-1 or random") from exc
        if self.conditional_order is not None and self.conditional_order != "random":
            order = self.conditional_order
            if (not isinstance(order, list) or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in order)
                    or not order or len(set(order)) != len(order)):
                raise ValueError("conditional_order must be a list of distinct nonnegative agent indices")
        if self.agent_ids is not None and len(set(self.agent_ids)) != len(self.agent_ids):
            raise ValueError("agent_ids must not contain duplicates")
        if self.pretrain_agent_id is not None and (isinstance(self.pretrain_agent_id, bool)
                                                  or not isinstance(self.pretrain_agent_id, int)
                                                  or self.pretrain_agent_id < 0):
            raise ValueError("pretrain_agent_id must be a nonnegative dataset agent ID or null")
        if self.device.isdigit():
            self.device = "cuda:" + self.device
        if self.device == "cuda":
            self.device = "cuda:0"
        if self.device != "cpu" and not (self.device.startswith("cuda:") and self.device[5:].isdigit()):
            raise ValueError("device must be cpu or a CUDA device such as cuda:0")
        return self

    def dataset_path(self):
        if self.data_path:
            return Path(self.data_path).expanduser()
        root = Path(self.dataset_dir).expanduser()
        if self.dataset_format == "omiga" or self.profile == "omiga":
            partition = self.agent_conf or {
                "HalfCheetah-v2": "6x1", "Ant-v2": "2x4", "Hopper-v2": "3x1"
            }.get(self.env_id)
            if not partition:
                raise ValueError("Set --data-path or --agent-conf for this OMIGA environment")
            return root / "omiga" / (self.env_id + "-" + partition + "-" + self.data_type + ".hdf5")
        if self.dataset_format == "ogmarl":
            raise ValueError("OG-MARL layouts vary; specify --data-path to the episode directory")
        return root / self.env_id / self.data_type / ("seed_" + str(self.dataset_num) + "_data")


def load_config(path=None, overrides=None, *, variant="main", stage=None):
    """Apply original stage defaults, profile settings, then YAML/CLI overrides."""
    values = {}
    if path:
        with open(path, encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
            values = {} if loaded is None else loaded
        if not isinstance(values, dict):
            raise ValueError("Config must be a YAML mapping")
    values.update(overrides or {})
    if "mode" in values or "marltype" in values or "difftype" in values:
        raise ValueError("Select algorithm: OMSD, CTDE, or IND; mode/marltype/difftype are no longer configuration fields")
    if "behavior_model" in values:
        model = values.pop("behavior_model")
        expected = "gmm" if variant == "gmm" else "diffusion"
        if not isinstance(model, str) or model.lower() != expected:
            raise ValueError("GMM runs use the separate ablation/gmm entrypoints; main runs always use diffusion")
    if "variant" in values and values["variant"] != variant:
        raise ValueError("variant is determined by the entrypoint, not by the YAML config")
    values["variant"] = variant
    if variant != "order" and "ordered_updates" in values:
        expected = get_algorithm_spec(values.get("algorithm", "OMSD")).mode == "SEQ"
        value = values["ordered_updates"]
        if isinstance(value, str) and value.lower() in {"true", "1", "yes", "false", "0", "no"}:
            value = value.lower() in {"true", "1", "yes"}
        if value is not expected:
            raise ValueError("Custom ordered_updates is available only through ablation/order")
    known = {field.name for field in fields(ExperimentConfig)}
    unknown = set(values) - known
    if unknown:
        raise ValueError("Unknown configuration fields: " + ", ".join(sorted(unknown)))
    profile = values.get("profile", "mpe")
    if profile not in PROFILES:
        raise ValueError("Unknown profile: " + str(profile))
    if stage is not None and stage not in STAGE_DEFAULTS:
        raise ValueError("Unknown training stage: " + str(stage))
    if values.get("pretrain_agent_id") is not None:
        if stage in {"policy", "inspect"}:
            raise ValueError("pretrain_agent_id is only available for behavior or IND critic pretraining")
        if stage == "critic" and get_algorithm_spec(values.get("algorithm", "OMSD")).mode != "IND":
            raise ValueError("OMSD/CTDE use one joint critic; --pretrain-agent-id is only valid for IND critic pretraining")
    merged = dict(STAGE_DEFAULTS.get(stage, {}))
    merged.update(PROFILES[profile])
    merged.update(values)
    # Backend-specific partitions do not control model dimensions: datasets do.
    env_id = merged.get("env_id", "simple_spread")
    if profile == "omiga":
        merged.setdefault("agent_conf", {"HalfCheetah-v2": "6x1", "Ant-v2": "2x4",
                                         "Hopper-v2": "3x1"}.get(env_id))
    if profile == "mamujoco210":
        merged.setdefault("agent_conf", {"2ant": "2x4", "4ant": "4x2",
                                         "2halfcheetah": "2x3"}.get(env_id))
    if profile == "mpe" and env_id in {"simple_tag", "simple_world"}:
        # The original experiments learn predators and evaluate a fixed prey policy.
        merged.setdefault("agent_ids", [0, 1, 2])
    merged.setdefault("ordered_updates", get_algorithm_spec(merged.get("algorithm", "OMSD")).mode == "SEQ")
    return ExperimentConfig(**merged).validate()

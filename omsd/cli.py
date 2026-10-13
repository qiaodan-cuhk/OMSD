"""Argument parsing shared by the three independent research entrypoints."""

import argparse
import importlib
import sys

from .config import load_config, PROFILES
from .specs import ALGORITHM_SPECS


_STAGES = {
    "behavior": "behavior", "pretrain-behavior": "behavior",
    "critic": "critic", "pretrain-critic": "critic",
    "policy": "policy", "train": "policy", "inspect": "inspect",
}

_COMMON_FIELDS = {
    "profile", "env_id", "backend", "agent_conf", "agent_obsk", "discrete_action",
    "episode_length", "data_path", "dataset_dir", "dataset_format", "data_type",
    "dataset_num", "transition_mode", "terminal_policy", "observation_prefix",
    "observation_dims", "action_dims", "agent_ids", "reward_scale", "max_size",
    "algorithm", "critic_input", "device", "seed", "num_threads",
    "output_root", "output_dir", "run_id", "checkpoint_interval",
    "logging_interval", "tensorboard",
}
_DIFFUSION_FIELDS = {
    "actor_blocks", "t_GassProj_dims", "t_embed_dims", "sa_embed_dims", "resnet_hidden_dim",
}
_STAGE_FIELDS = {
    "behavior": {"num_epochs", "steps_per_epoch", "batch_size", "learning_rate", "lr_anneal", "pretrain_agent_id"},
    "critic": {"num_epochs", "steps_per_epoch", "batch_size", "gamma", "tau", "iql_critic_lr", "q_layer", "pretrain_agent_id"},
    "policy": {"num_steps", "batch_size", "beta", "WT", "regq", "dilac_lr",
               "policy_layer", "q_layer", "n_policy_epochs", "behavior_checkpoint",
               "critic_checkpoint", "policy_checkpoint", "opponent_checkpoint",
               "eval_interval", "eval_episodes"},
    "inspect": set(),
}
_INTEGER_FIELDS = {
    "agent_obsk", "episode_length", "dataset_num", "observation_prefix", "max_size",
    "seed", "num_steps", "num_epochs", "batch_size", "regq", "policy_layer", "q_layer",
    "actor_blocks", "t_GassProj_dims", "t_embed_dims", "sa_embed_dims", "resnet_hidden_dim",
    "n_policy_epochs", "steps_per_epoch", "n_gmm_components", "gmm_hidden_dim", "checkpoint_interval",
    "logging_interval", "eval_interval", "eval_episodes", "num_threads", "pretrain_agent_id",
}
_FLOAT_FIELDS = {"reward_scale", "gamma", "tau", "beta", "learning_rate", "dilac_lr", "iql_critic_lr"}
_LIST_FIELDS = {"observation_dims", "action_dims", "agent_ids"}
_BOOLEAN_FIELDS = {"discrete_action", "ordered_updates", "lr_anneal", "tensorboard", "gmm_score_normalize"}
_DESCRIPTIONS = {
    "algorithm": "Algorithm decomposition: OMSD sequential/joint, CTDE independent/joint, IND independent/local (default: OMSD)",
    "profile": "Dataset/environment convention: " + ", ".join(PROFILES),
    "backend": "Simulator backend for optional policy evaluation",
    "data_path": "Existing dataset file or directory; datasets are never downloaded",
    "dataset_format": "Data adapter: auto, numpy, omiga, ogmarl, hdf5",
    "agent_ids": "Physical dataset agent IDs in learner order, e.g. 0,1,2",
    "pretrain_agent_id": "Train only this physical dataset agent ID, retaining all SEQ prefix data; critic: IND only",
    "conditional_order": "Ablation only: fixed permutation of learner positions (2-0-1) or seeded random",
    "ordered_updates": "Ablation only: traverse agents in the fixed conditional order",
    "observation_prefix": "Number of leading observation features to remove from OG-MARL exports",
    "terminal_policy": "provided, combine, or fixed_horizon transition end masks",
    "transition_mode": "HDF5 convention: auto, explicit, omiga, adjacent",
    "tau": "IQL expectile, between zero and one",
    "critic_input": "obs (concatenated local observations) or explicit dataset state",
    "num_epochs": "Pretraining epochs, each containing fixed buffer-sampled updates (default: 200)",
    "steps_per_epoch": "Updates per epoch; defaults to original behavior=10000 / critic=5000",
    "num_steps": "Number of policy optimization updates",
    "eval_interval": "Policy updates between simulator evaluations; 0 disables evaluation",
    "behavior_checkpoint": "Matching behavior checkpoint, or one per-agent container/legacy file per dataset agent ID in learner order",
    "critic_checkpoint": "Matching joint IQL file, or one per-agent IND container/legacy file per dataset agent ID in learner order",
    "policy_checkpoint": "Optional initial policy weights or per-agent files; optimizer starts fresh",
    "opponent_checkpoint": "Prey policy for evaluating a predator subset",
    "device": "CUDA index (0, 1, ...), cuda:INDEX, or cpu (default: cuda:0)",
    "output_root": "Root for automatically organized artifacts (default: outputs)",
    "output_dir": "Optional explicit directory for this stage's artifacts",
    "run_id": "Optional experiment identifier; generated when omitted",
    "n_gmm_components": "GMM ablation: number of mixture components (default: 8)",
    "gmm_hidden_dim": "GMM ablation: hidden layer width (default: 512)",
    "gmm_score_normalize": "GMM ablation only: retain original unit-normalized score magnitude (default: true)",
}


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("Use true or false")


def parse_int_list(value):
    try:
        result = [int(x) for x in value.replace("-", ",").split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated integers") from exc
    if any(x < 0 for x in result):
        raise argparse.ArgumentTypeError("Dimensions and agent IDs cannot be negative")
    return result


def build_stage_parser(stage, *, variant="main"):
    if stage not in _STAGES:
        raise ValueError("Unknown training stage: " + str(stage))
    if variant not in {"main", "gmm", "order"}:
        raise ValueError("Unknown training variant: " + str(variant))
    phase = _STAGES[stage]
    label = "GMM ablation" if variant == "gmm" else ("order ablation" if variant == "order" else "diffusion")
    parser = argparse.ArgumentParser(description=f"OMSD {phase}: {label} pipeline")
    parser.add_argument("--config", help="Flat YAML config; explicit arguments override it")
    selected = _COMMON_FIELDS | _STAGE_FIELDS[phase]
    if variant == "gmm":
        selected.discard("WT")
        if phase == "policy":
            selected.add("gmm_score_normalize")
    if phase in {"behavior", "policy"}:
        selected |= {"n_gmm_components", "gmm_hidden_dim"} if variant == "gmm" else _DIFFUSION_FIELDS
    if variant == "order" and phase in {"behavior", "policy"}:
        selected |= {"conditional_order", "ordered_updates"}
    for name in sorted(selected):
        flags = ["--" + name.replace("_", "-").lower()]
        if name == "device":
            flags.append("--device-id")
        if name == "pretrain_agent_id" and phase == "behavior":
            flags.extend(["--seq-agent-id", "--seq_agent_id"])
        kwargs = dict(dest=name, default=argparse.SUPPRESS)
        if name in _DESCRIPTIONS:
            kwargs["help"] = _DESCRIPTIONS[name]
        if name == "algorithm":
            kwargs.update(type=str.upper, choices=tuple(ALGORITHM_SPECS))
        elif name in _INTEGER_FIELDS:
            kwargs["type"] = int
        elif name in _FLOAT_FIELDS:
            kwargs["type"] = float
        elif name in _LIST_FIELDS:
            kwargs["type"] = parse_int_list
        elif name in _BOOLEAN_FIELDS:
            kwargs.update(type=parse_bool, nargs="?", const=True)
        elif name in {"behavior_checkpoint", "critic_checkpoint", "policy_checkpoint"}:
            kwargs["nargs"] = "+"
        parser.add_argument(*flags, **kwargs)
    if "lr_anneal" in selected:
        parser.add_argument("--no-lr-anneal", dest="lr_anneal", action="store_false", default=argparse.SUPPRESS)
    if "gmm_score_normalize" in selected:
        parser.add_argument("--no-gmm-score-normalize", dest="gmm_score_normalize", action="store_false", default=argparse.SUPPRESS,
                            help="Use raw GMM score instead of the original unit normalization (research option)")
    return parser


def _config_from_args(parser, args, variant, stage):
    values = vars(args)
    path = values.pop("config", None)
    values.pop("command", None)
    for name in ("behavior_checkpoint", "critic_checkpoint", "policy_checkpoint"):
        value = values.get(name)
        if isinstance(value, list) and len(value) == 1:
            values[name] = value[0]
    try:
        return load_config(path, values, variant=variant, stage=stage)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


def parse_stage_config(stage, argv=None, *, variant="main"):
    parser = build_stage_parser(stage, variant=variant)
    return _config_from_args(parser, parser.parse_args(argv), variant, _STAGES[stage])


def build_parser():
    parser = argparse.ArgumentParser(description="OMSD: independent behavior, critic, and policy training stages")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "pretrain-behavior", "pretrain-critic", "train"):
        commands.add_parser(command, parents=[build_stage_parser(command)], add_help=False)
    return parser


def parse_config(argv=None):
    """Optional module dispatcher parser; training scripts parse their own stage."""
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command
    return command, _config_from_args(parser, args, "main", _STAGES[command])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in {"inspect", "pretrain-behavior", "pretrain-critic", "train"}:
        build_parser().parse_args(argv)
    command = argv[0]
    if command == "inspect":
        _, config = parse_config(argv)
        from .runtime import inspect
        return inspect(config)
    module = importlib.import_module({
        "pretrain-behavior": "pretrain_behavior",
        "pretrain-critic": "pretrain_critic", "train": "main",
    }[command])
    return module.main(argv[1:])

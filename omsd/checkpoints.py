"""Explicit component checkpoints with validated architecture and data contracts.

Loading never guesses a model type or changes the runtime configuration.
Original tensor-only state_dicts load directly into the explicitly constructed
model; their environment and order settings remain the caller's responsibility.
GMM's separate entrypoints use the same container contract without cross-model
conversion.
"""

import inspect
import os
from pathlib import Path
import warnings

import torch


FORMAT_VERSION = 2
COMPONENTS = {"behavior", "critic", "policy"}


def _component(component):
    if component not in COMPONENTS:
        raise ValueError("component must be behavior, critic, or policy")


def _algorithm(agent):
    return getattr(agent.config, "algorithm", {"SEQ": "OMSD", "CTDE": "CTDE", "IND": "IND"}.get(agent.mode, agent.mode))


def model_metadata(agent, dataset):
    """Record effective dimensions/semantics, not only user supplied overrides."""
    config = agent.config
    metadata = getattr(dataset, "metadata", {})
    terminal_policy = metadata.get("terminal_policy", config.terminal_policy)
    if terminal_policy == "auto":
        terminal_policy = "provided"
    semantics = dict(
        reward_scale=float(metadata.get("reward_scale", config.reward_scale)),
        terminal_policy=terminal_policy,
        transition_mode=metadata.get("transition_mode", {
            "numpy": "explicit", "npy": "explicit", "memory": "explicit",
            "ogmarl": "adjacent", "omiga": "omiga",
        }.get(dataset.source_format, config.transition_mode)),
        source_format=dataset.source_format,
        terminal_next_obs=metadata.get("terminal_next_obs", "provided"),
    )
    if terminal_policy in {"combine", "fixed_horizon"}:
        semantics["episode_length"] = metadata.get("episode_length", config.episode_length)
    policy_semantics = dict(
        prefix_actions="fixed-round-snapshot",
        behavior_vector="negative-analytic-score" if agent.behavior_model == "gmm" else "weighted-noise",
    )
    if agent.behavior_model == "gmm":
        policy_semantics["gmm_score_normalize"] = bool(getattr(config, "gmm_score_normalize", True))
    return dict(
        obs_dims=list(dataset.obs_dims), action_dims=list(dataset.action_dims),
        state_dims=list(dataset.state_dims or []), agent_ids=list(dataset.agent_ids),
        algorithm=_algorithm(agent),
        behavior_layout="seq" if agent.mode == "SEQ" else "ind",
        critic_layout="ind" if agent.mode == "IND" else "joint",
        behavior_model=agent.behavior_model,
        conditional_order=list(agent.conditional_order),
        ordered_updates=bool(config.ordered_updates),
        critic_input=config.critic_input,
        environment=dict(profile=config.profile, backend=config.backend,
                         env_id=config.env_id, agent_conf=config.agent_conf,
                         agent_obsk=config.agent_obsk,
                         discrete_action=bool(config.discrete_action)),
        observation_prefix=metadata.get("observation_prefix", 0),
        state_layout=metadata.get("state_layout", "per_agent"),
        data_semantics=semantics,
        policy_semantics=policy_semantics,
    )


def component_architecture(agent, component):
    """Describe only constructor settings relevant to the selected component."""
    _component(component)
    config = agent.config
    if component == "behavior":
        description = dict(condition_dims=list(agent.behavior_condition_dims),
                           output_dims=list(agent.action_dims), model=agent.behavior_model)
        names = ("n_gmm_components", "gmm_hidden_dim") if agent.behavior_model == "gmm" else (
            "actor_blocks", "t_GassProj_dims", "t_embed_dims", "sa_embed_dims", "resnet_hidden_dim")
        description.update({name: getattr(config, name) for name in names})
        return description
    if component == "critic":
        dimensions = agent.obs_dims if config.critic_input == "obs" else agent.state_dims
        if agent.mode == "IND":
            conditions = list(dimensions) if len(dimensions) > 1 else list(dimensions) * agent.n_agents
            outputs = list(agent.action_dims)
        else:
            shared_state = config.critic_input == "state" and agent.state_layout == "shared"
            conditions = [dimensions[0] if shared_state else sum(dimensions)]
            outputs = [sum(agent.action_dims)]
        return dict(condition_dims=conditions, action_dims=outputs,
                    model="iql", q_layer=config.q_layer, hidden_dim=256, value_layers=2)
    return dict(condition_dims=list(agent.obs_dims), action_dims=list(agent.action_dims),
                model="deterministic", policy_layer=config.policy_layer, hidden_dim=256)


def save_checkpoint(path, agent, dataset, component, step=0):
    _component(component)
    weights = agent.component_state_dict(component)
    if not weights:
        raise ValueError("Cannot save a component that was not constructed: " + component)
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(format_version=FORMAT_VERSION, component=component,
                   metadata=model_metadata(agent, dataset), config=agent.config.to_dict(),
                   architecture=component_architecture(agent, component),
                   step=int(step), weights=weights)
    index = getattr(agent, "pretrain_agent_index", None)
    if index is not None:
        if agent.config.pretrain_agent_id != dataset.agent_ids[index]:
            raise ValueError("Selected model and dataset agent IDs differ")
        payload["selection"] = dict(agent_index=index, agent_id=dataset.agent_ids[index])
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_metadata(actual, expected, component):
    if not isinstance(actual, dict):
        raise ValueError("Checkpoint requires explicit metadata")
    names = ["obs_dims", "action_dims", "agent_ids", "environment", "observation_prefix"]
    if component == "behavior":
        names += ["behavior_layout", "behavior_model"]
        if expected["behavior_layout"] == "seq":
            names += ["conditional_order"]
    elif component == "critic":
        names += ["critic_layout", "critic_input", "data_semantics"]
        if expected["critic_input"] == "state":
            names += ["state_dims", "state_layout"]
    else:
        names += ["algorithm", "behavior_model", "behavior_layout", "ordered_updates", "policy_semantics"]
        if expected["behavior_layout"] == "seq":
            names += ["conditional_order"]
    for name in names:
        if name not in actual:
            raise ValueError("Checkpoint requires metadata field: " + name)
        actual_value, expected_value = actual[name], expected[name]
        if name == "data_semantics" and isinstance(actual_value, dict) and isinstance(expected_value, dict):
            # The file/container label is provenance, not a training convention.
            # Identical OMIGA transitions may be read via omiga, hdf5, or auto.
            actual_value = {key: value for key, value in actual_value.items() if key != "source_format"}
            expected_value = {key: value for key, value in expected_value.items() if key != "source_format"}
        if actual_value != expected_value:
            raise ValueError("Checkpoint " + name + " mismatch: expected " +
                             str(expected[name]) + ", found " + str(actual[name]))


def _validate_weights(weights, expected):
    """Reject invalid weights before PyTorch can partially change a component."""
    if not isinstance(weights, dict) or set(weights) != set(expected):
        raise ValueError("Checkpoint weight keys do not match the configured component")
    for key, tensor in weights.items():
        target = expected[key]
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != target.shape or
                tensor.dtype != target.dtype or tensor.layout != target.layout):
            raise ValueError("Checkpoint weight shape or dtype mismatch: " + key)
        if not torch.isfinite(tensor).all():
            raise ValueError("Checkpoint contains non-finite weights: " + key)


def load_tensor_payload(path):
    """Read checkpoint payloads with the installed PyTorch serialization API.

    Newer releases support weights-only loading; PyTorch 1.10 uses its original
    loader. Component and opponent callers validate the returned tensor payload.
    """
    options = {"weights_only": True} if "weights_only" in inspect.signature(torch.load).parameters else {}
    return torch.load(path, map_location="cpu", **options)


def _read_file(path):
    root = Path(path).expanduser()
    if not root.exists():
        raise FileNotFoundError("Checkpoint not found: " + str(root))
    if not root.is_file():
        raise ValueError("Checkpoint must be an explicit file; automatic directory selection is unsupported")
    return load_tensor_payload(root)


def _raw_model_weights(weights, module, agent, component):
    """Match an original model or its known wrapper without guessing its type.

    Published/OMIGA pretraining wrappers contain only diffusion_behavior.* or
    gmm_score_model.*. Their saved critics contain q0.*, q0_target.*, and vf.*;
    the auxiliary IQL actor is not registered in that saved critic. Unexpected
    sibling components are rejected rather than silently discarded.
    """
    if not isinstance(weights, dict) or not weights or any(
            not isinstance(key, str) or not isinstance(value, torch.Tensor)
            for key, value in weights.items()):
        raise ValueError("Legacy checkpoint must be a nonempty tensor-only state_dict")
    expected = module.state_dict()
    if set(weights) == set(expected):
        selected = weights
    else:
        prefixes = {
            "behavior": ["gmm_score_model." if agent.behavior_model == "gmm" else "diffusion_behavior."],
            "critic": [],
            "policy": ["SRPO_policy."],
        }[component]
        selected = None
        for prefix in prefixes:
            if all(key.startswith(prefix) for key in weights):
                selected = {key[len(prefix):]: value for key, value in weights.items()}
                break
        if selected is None:
            raise ValueError("Legacy " + component + " keys do not match the explicitly configured model")
    _validate_weights(selected, expected)
    return selected


def _load_legacy(paths, agent, component, *, payload=None):
    modules = getattr(agent, {"behavior": "behaviors", "critic": "critics", "policy": "policies"}[component])
    if not modules:
        raise ValueError("Cannot load a component that was not constructed: " + component)
    expected = agent.component_state_dict(component)
    if payload is not None and isinstance(payload, dict) and set(payload) == set(expected):
        # Also support an explicitly saved ModuleList.state_dict(), with its
        # deterministic 0./1./... prefixes and all agents in one tensor dict.
        _validate_weights(payload, expected)
        combined = payload
    else:
        if len(paths) != len(modules):
            raise ValueError("Legacy " + component + " checkpoint requires exactly " + str(len(modules)) +
                             " explicitly selected per-agent files in dataset agent-ID order")
        combined = {}
        for index, source, module in zip(agent.component_indices(component), paths, modules):
            weights = payload if payload is not None else _read_file(source)
            selected = _raw_model_weights(weights, module, agent, component)
            combined.update({str(index) + "." + key: value for key, value in selected.items()})
        _validate_weights(combined, expected)
    # Validate every file/key/tensor before performing the first model mutation.
    warnings.warn("Loading legacy tensor-only " + component + " weights: architecture and tensors were checked; "
                  "environment, dataset semantics, and conditional order use the supplied configuration "
                  "because these files contain no compatibility metadata.", UserWarning, stacklevel=3)
    agent.load_component_state_dict(component, combined)
    return 0


def _validate_container(payload, agent, dataset, component):
    if not isinstance(payload, dict) or payload.get("format_version") not in {1, FORMAT_VERSION}:
        raise ValueError("Unsupported checkpoint format; use checkpoints produced by the maintained training entrypoints")
    if payload.get("component") != component:
        raise ValueError("Expected a " + component + " checkpoint")
    if not isinstance(payload.get("config"), dict) or not isinstance(payload.get("architecture"), dict):
        raise ValueError("Checkpoint requires explicit config and architecture metadata; legacy conversion is unsupported")
    step = payload.get("step", 0)
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("Checkpoint step must be a nonnegative integer")
    _validate_metadata(payload.get("metadata"), model_metadata(agent, dataset), component)
    if payload["architecture"] != component_architecture(agent, component):
        raise ValueError("Checkpoint architecture mismatch; configure the component explicitly before loading")
    if "selection" in payload and (not isinstance(payload["selection"], dict)
                                    or payload["config"].get("pretrain_agent_id") != payload["selection"].get("agent_id")):
        raise ValueError("Checkpoint agent selection disagrees with its pretraining config")


def _validate_selection(selection, index, dataset, component, agent):
    if component == "critic" and agent.mode != "IND":
        raise ValueError("Joint critic checkpoints cannot contain a single-agent selection")
    expected = dict(agent_index=index, agent_id=dataset.agent_ids[index])
    if (not isinstance(selection, dict) or set(selection) != set(expected)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in selection.values())
            or selection != expected):
        raise ValueError("Checkpoint agent selection mismatch: expected " + str(expected) + ", found " + str(selection))


def _load_agent_files(paths, agent, dataset, component):
    """Validate the complete ordered list, then mutate the component once."""
    modules = agent._component(component)
    indices = agent.component_indices(component)
    if len(paths) != len(modules):
        raise ValueError(component + " checkpoint requires exactly " + str(len(modules)) +
                         " explicitly selected per-agent files in dataset agent-ID order")
    combined, legacy, steps = {}, False, []
    for source, index, module in zip(paths, indices, modules):
        payload = _read_file(source)
        if isinstance(payload, dict) and "format_version" not in payload:
            selected = _raw_model_weights(payload, module, agent, component)
            combined.update({str(index) + "." + key: value for key, value in selected.items()})
            legacy = True
            steps.append(0)
        else:
            _validate_container(payload, agent, dataset, component)
            if "selection" not in payload:
                raise ValueError("Per-agent checkpoint lists require single-agent containers; use one full component container directly")
            _validate_selection(payload["selection"], index, dataset, component, agent)
            expected = {str(index) + "." + key: value for key, value in module.state_dict().items()}
            _validate_weights(payload.get("weights"), expected)
            combined.update(payload["weights"])
            steps.append(payload.get("step", 0))
    _validate_weights(combined, agent.component_state_dict(component))
    if legacy:
        warnings.warn("Loading legacy tensor-only " + component + " weights: architecture and tensors were checked; "
                      "environment, dataset semantics, and conditional order use the supplied configuration "
                      "because these files contain no compatibility metadata.", UserWarning, stacklevel=3)
    step = min(steps)
    agent.load_component_state_dict(component, combined)
    return step


def load_checkpoint(path, agent, dataset, component):
    """Load a container, a raw joint component, or explicit per-agent files.

    A list/tuple selects one single-agent container or original tensor-only file
    for each model in dataset agent-ID order. It never searches filenames.
    Bare state_dicts do not provide the semantic validation of new containers.
    """
    _component(component)
    if isinstance(path, (list, tuple)):
        if len(path) == 1 and len(agent._component(component)) == 1:
            return load_checkpoint(path[0], agent, dataset, component)
        return _load_agent_files(path, agent, dataset, component)
    payload = _read_file(path)
    if isinstance(payload, dict) and "format_version" not in payload:
        return _load_legacy([path], agent, component, payload=payload)
    _validate_container(payload, agent, dataset, component)
    if "selection" in payload:
        indices = agent.component_indices(component)
        if len(indices) != 1:
            raise ValueError("A single-agent checkpoint requires the complete per-agent file list in dataset agent-ID order")
        _validate_selection(payload["selection"], indices[0], dataset, component, agent)
    elif getattr(agent, "pretrain_agent_index", None) is not None:
        raise ValueError("Single-agent pretraining requires a matching single-agent checkpoint")
    expected_weights = agent.component_state_dict(component)
    if not expected_weights:
        raise ValueError("Cannot load a component that was not constructed: " + component)
    _validate_weights(payload.get("weights"), expected_weights)
    agent.load_component_state_dict(component, payload["weights"])
    return payload.get("step", 0)

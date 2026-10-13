"""Dataset adapters with one transition schema and aligned multi-agent sampling.

OMIGA files store observations of consecutive timesteps: ``d[t]`` marks a
terminal observation and a transition uses ``r[t]`` and ``d[t + 1]``. Legacy
NumPy files already store transitions. These conventions are intentionally
separate; using the wrong convention changes the training targets.
"""

from pathlib import Path
import re

import h5py
import numpy as np
import torch


class DatasetError(ValueError):
    """A dataset does not match its declared schema."""


def _finite(value, name):
    if value.dtype.kind not in "biuf":
        raise DatasetError(f"{name}: expected numeric data, got {value.dtype}")
    for start in range(0, len(value), 65536):
        if not np.isfinite(value[start:start + 65536]).all():
            raise DatasetError(f"{name}: contains NaN or infinity")


def _array(value, name, scalar=False):
    value = np.asarray(value)
    if scalar:
        if value.ndim == 2 and value.shape[1] == 1:
            value = value[:, 0]
        if value.ndim != 1:
            raise DatasetError(f"{name}: expected [transitions] or [transitions, 1], got {value.shape}")
    else:
        if value.ndim == 1:
            value = value[:, None]
        if value.ndim != 2 or value.shape[1] == 0:
            raise DatasetError(f"{name}: expected [transitions, features], got {value.shape}")
    _finite(value, name)
    with np.errstate(over="ignore", invalid="ignore"):
        value = value.astype(np.float32, copy=False)
    _finite(value, name)
    return value


def _binary(value, name):
    value = _array(value, name, scalar=True)
    if not np.isin(value, [0, 1]).all():
        raise DatasetError(f"{name}: terminal flags must be 0 or 1")
    return value


def _capacity(size, max_size):
    if size <= 0:
        raise DatasetError("dataset contains no usable transitions")
    if max_size is not None:
        if int(max_size) != max_size or max_size <= 0:
            raise DatasetError("max_size must be a positive integer")
        if size > max_size:
            raise DatasetError(f"dataset has {size} transitions, exceeding capacity {max_size}; increase capacity explicitly")


def _ids(ids, available):
    chosen = list(available if ids is None else ids)
    if not chosen or len(set(chosen)) != len(chosen):
        raise DatasetError("agent_ids must be nonempty and unique")
    missing = [i for i in chosen if i not in available]
    if missing:
        raise DatasetError(f"agent_ids {missing} do not exist; available IDs: {list(available)}")
    return chosen


def _declared_dims(agents, field, dims):
    if dims is None:
        return
    dims = list(dims)
    if len(dims) != len(agents):
        raise DatasetError(f"{field} dimensions must contain one value per selected agent")
    paired = {"obs": "next_obs", "action": "next_action"}[field]
    for i, (agent, dim) in enumerate(zip(agents, dims)):
        if not isinstance(dim, (int, np.integer)) or dim <= 0:
            raise DatasetError(f"agent {i} {field}: declared dimension must be a positive integer")
        for key in (field, paired):
            if key not in agent:
                continue
            array = agent[key]
            if dim > array.shape[1]:
                raise DatasetError(f"agent {i} {key}: declared {dim} features, data has {array.shape[1]}")
            if dim < array.shape[1]:
                if np.any(array[:, dim:] != 0):
                    raise DatasetError(f"agent {i} {key}: cannot trim nonzero features from {array.shape[1]} to {dim}")
                agent[key] = array[:, :dim]


def _terminals(agents, policy, episode_length):
    if policy == "auto":
        policy = "provided"
    if policy not in {"provided", "combine", "fixed_horizon"}:
        raise DatasetError("terminal_policy must be auto, provided, combine, or fixed_horizon")
    if policy == "provided":
        return
    if not isinstance(episode_length, (int, np.integer)) or episode_length <= 0:
        raise DatasetError("fixed horizon terminals require a positive episode_length")
    # Some original MPE exports have all-zero dones and use a fixed 25-step
    # rollout. This is opt-in because a dimension or filename is not evidence
    # that an arbitrary dataset has that horizon.
    horizon = ((np.arange(len(agents[0]["done"])) + 1) % episode_length == 0).astype(np.float32)
    for agent in agents:
        agent["done"] = np.maximum(agent["done"], horizon) if policy == "combine" else horizon.copy()


def _infer_next_actions(agents):
    """Use adjacent actions only where all agents confirm the same successor."""
    size = len(agents[0]["obs"])
    aligned = np.zeros(size, dtype=bool)
    aligned[:-1] = True
    for agent in agents:
        aligned &= agent["done"] == 0
        for start in range(0, size - 1, 65536):
            end = min(start + 65536, size - 1)
            aligned[start:end] &= np.all(agent["next_obs"][start:end] == agent["obs"][start + 1:end + 1], axis=1)
    for agent in agents:
        if "next_action" not in agent:
            agent["next_action"] = np.zeros_like(agent["action"])
            indices = np.flatnonzero(aligned)
            agent["next_action"][indices] = agent["action"][indices + 1]
            agent["next_action_valid"] = aligned.astype(np.float32)
        else:
            # Explicit next actions are valid even for shuffled transitions.
            agent["next_action"] = agent["next_action"].copy()
            agent["next_action_valid"] = np.ones(size, dtype=np.float32)
        terminal = agent["done"] != 0
        agent["next_action"][terminal] = 0
        agent["next_action_valid"][terminal] = 0


class Dataset:
    """Validated per-agent arrays; sampled rows always refer to the same timestep.

    ``next_action_valid`` distinguishes a measured/inferred next action from a
    zero placeholder. Explicit transitions may be shuffled, so adjacency must
    be verified instead of taking the next row or wrapping at the file end.
    """

    def __init__(self, agents, agent_ids=None, source_format="memory", metadata=None):
        self.agents = []
        required = {"obs", "action", "rewards", "next_obs", "done"}
        expected_size = None
        for index, original in enumerate(agents):
            missing = required - original.keys()
            if missing:
                raise DatasetError(f"agent {index}: missing fields {sorted(missing)}")
            agent = {}
            for key, array in original.items():
                if key in {"done", "next_action_valid"}:
                    agent[key] = _binary(array, f"agent {index} {key}")
                else:
                    agent[key] = _array(array, f"agent {index} {key}", scalar=key in {"rewards", "next_action_valid"})
            size = len(agent["obs"])
            if expected_size is None:
                expected_size = size
            for key, array in agent.items():
                if len(array) != expected_size:
                    raise DatasetError(f"agent {index} {key}: length {len(array)} differs from aligned length {expected_size}")
            if agent["obs"].shape != agent["next_obs"].shape:
                raise DatasetError(f"agent {index}: obs and next_obs shapes differ")
            if ("state" in agent) != ("next_state" in agent):
                raise DatasetError(f"agent {index}: state and next_state must be supplied together")
            if "state" in agent and agent["state"].shape != agent["next_state"].shape:
                raise DatasetError(f"agent {index}: state and next_state shapes differ")
            if "next_action" in agent and agent["action"].shape != agent["next_action"].shape:
                raise DatasetError(f"agent {index}: action and next_action shapes differ")
            if "next_action_valid" in agent and "next_action" not in agent:
                raise DatasetError(f"agent {index}: next_action_valid requires next_action")
            self.agents.append(agent)
        _capacity(expected_size or 0, None)
        if any("state" in a for a in self.agents) and not all("state" in a for a in self.agents):
            raise DatasetError("state fields must be present for every selected agent or none")
        self.num_agents = len(self.agents)
        self.agent_ids = list(range(self.num_agents)) if agent_ids is None else list(agent_ids)
        if len(self.agent_ids) != self.num_agents or len(set(self.agent_ids)) != self.num_agents:
            raise DatasetError("agent_ids must contain one unique ID per agent")
        self.size = expected_size
        self.obs_dims = [a["obs"].shape[1] for a in self.agents]
        self.action_dims = [a["action"].shape[1] for a in self.agents]
        self.state_dims = [a["state"].shape[1] for a in self.agents] if "state" in self.agents[0] else None
        self.source_format = source_format
        self.metadata = dict(metadata or {})
        if not all("next_action_valid" in a for a in self.agents):
            _infer_next_actions(self.agents)
        self.metadata["missing_nonterminal_next_actions"] = [
            int(np.count_nonzero((a["done"] == 0) & (a["next_action_valid"] == 0))) for a in self.agents]

    def __len__(self):
        return self.size

    def sample(self, batch_size, device="cpu"):
        if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
            raise DatasetError("batch_size must be a positive integer")
        indices = np.random.randint(self.size, size=batch_size)
        return self.batch(indices, device=device)

    def batch(self, indices, device="cpu"):
        """Fetch declared rows for an epoch permutation without replacement."""
        indices = np.asarray(indices)
        if indices.ndim != 1 or indices.size == 0 or indices.dtype.kind not in "iu":
            raise DatasetError("batch indices must be a nonempty one-dimensional integer sequence")
        if np.any(indices < 0) or np.any(indices >= self.size):
            raise DatasetError(f"batch indices must lie in [0, {self.size})")
        return [{key: torch.as_tensor(array[indices], device=device) for key, array in agent.items()}
                for agent in self.agents]


def _legacy_numpy(path, agent_ids, max_size):
    available = sorted(int(match.group(1)) for file in path.glob("obs_*.npy")
                       for match in [re.fullmatch(r"obs_(\d+)\.npy", file.name)] if match)
    selected = _ids(agent_ids, available)
    names = {"obs": "obs", "action": "acs", "rewards": "rews", "next_obs": "next_obs", "done": "dones"}
    agents = []
    length = None
    for agent_id in selected:
        agent = {}
        fields = dict(names)
        if (path / f"states_{agent_id}.npy").exists() or (path / f"next_states_{agent_id}.npy").exists():
            fields.update(state="states", next_state="next_states")
        if (path / f"next_acs_{agent_id}.npy").exists():
            fields["next_action"] = "next_acs"
        for key, prefix in fields.items():
            file = path / f"{prefix}_{agent_id}.npy"
            if not file.is_file():
                raise DatasetError(f"agent {agent_id}: missing {file.name}")
            try:
                array = np.load(file, mmap_mode="r", allow_pickle=False)
            except (ValueError, OSError) as error:
                raise DatasetError(f"{file.name}: {error}") from error
            if array.ndim == 0:
                raise DatasetError(f"{file.name}: missing transition axis")
            if length is None:
                length = len(array)
                _capacity(length, max_size)
            if len(array) != length:
                raise DatasetError(f"{file.name}: length {len(array)} differs from aligned length {length}")
            agent[key] = _binary(array, file.name) if key == "done" else _array(array, file.name, scalar=key == "rewards")
        agents.append(agent)
    return agents, selected, {}


_KEYS = {
    "obs": ("observations", "obs", "o"), "action": ("actions", "action", "a"),
    "rewards": ("rewards", "reward", "r"), "done": ("terminals", "dones", "done", "d"),
    "next_obs": ("next_observations", "next_obs", "next_o"),
    "state": ("states", "state", "s"), "next_state": ("next_states", "next_state", "next_s"),
    "next_action": ("next_actions", "next_action", "next_a"),
    "timeouts": ("timeouts", "truncated", "truncations"),
}


def _field(file, field, required=False):
    matches = [file[key] for key in _KEYS[field] if key in file and isinstance(file[key], h5py.Dataset)]
    if len(matches) > 1:
        raise DatasetError(f"ambiguous HDF5 {field} keys; use one of {_KEYS[field]}")
    if not matches and required:
        raise DatasetError(f"HDF5 missing {field}; expected one of {_KEYS[field]}")
    return matches[0] if matches else None


def _features(field, indices, agent_id, agent_count, name, shared=False):
    if field.ndim == 3:
        if field.shape[1] != agent_count:
            raise DatasetError(f"{name}: agent axis {field.shape[1]} differs from observations {agent_count}")
        return _array(field[indices, agent_id, :], name)
    if field.ndim == 2 and (agent_count == 1 or shared):
        return _array(field[indices, :], name)
    if field.ndim == 2 and field.shape[1] == agent_count and name in {"action", "next_action"}:
        return _array(field[indices, agent_id], name)
    if field.ndim == 1 and agent_count == 1:
        return _array(field[indices], name)
    raise DatasetError(f"{name}: expected [steps, agents, features], got {field.shape}")


def _scalars(field, indices, agent_id, agent_count, name, terminal=False):
    shape = field.shape
    if field.ndim == 1 or (field.ndim == 2 and shape[1] == 1):
        value = field[indices]
    elif field.ndim == 2 and shape[1] == agent_count:
        value = field[indices, agent_id]
    elif field.ndim == 3 and shape[1:] == (agent_count, 1):
        value = field[indices, agent_id, 0]
    else:
        raise DatasetError(f"{name}: expected shared scalar or one scalar per agent, got {shape}")
    return _binary(value, name) if terminal else _array(value, name, scalar=True)


def _hdf5(path, agent_ids, max_size, mode):
    with h5py.File(path, "r") as file:
        obs = _field(file, "obs", True)
        if obs.ndim not in {2, 3}:
            raise DatasetError(f"{obs.name}: expected [steps, features] or [steps, agents, features], got {obs.shape}")
        actions = _field(file, "action", True)
        rewards = _field(file, "rewards", True)
        done = _field(file, "done", True)
        next_obs = _field(file, "next_obs")
        state = _field(file, "state")
        next_state = _field(file, "next_state")
        next_action = _field(file, "next_action")
        timeouts = _field(file, "timeouts")
        count = obs.shape[1] if obs.ndim == 3 else 1
        selected = _ids(agent_ids, range(count))
        length = len(obs)
        fields = [actions, rewards, done, next_obs, state, next_state, next_action, timeouts]
        for field in fields:
            if field is not None and (field.ndim == 0 or len(field) != length):
                raise DatasetError(f"{field.name}: expected aligned length {length}, got {field.shape}")
        if mode == "auto":
            mode = "explicit" if next_obs is not None else ("omiga" if {"s", "o", "a", "r", "d"}.issubset(file) else None)
            if mode is None:
                raise DatasetError("HDF5 has no explicit next observations; specify transition_mode='adjacent' for current-row terminals")
        if mode == "explicit":
            if next_obs is None:
                raise DatasetError("explicit transitions require next_obs/next_observations")
            if (state is None) != (next_state is None):
                raise DatasetError("explicit transitions require both state and next_state or neither")
            indices = np.arange(length)
        else:
            if next_obs is not None or next_state is not None:
                raise DatasetError(f"{mode} snapshot mode conflicts with explicit next observations/states")
            if length < 2:
                raise DatasetError("snapshot datasets need at least two timesteps")
            # Episode resets must never become a successor. In OMIGA d[t]
            # means observation t is terminal. Current-row convention uses
            # d[t] to end transition t, so the final transition cannot be
            # reconstructed without an explicit next observation.
            boundary = np.zeros(length, dtype=bool)
            for i in range(count):
                boundary |= _scalars(done, slice(None), i, count, "done", True).astype(bool)
                if timeouts is not None:
                    boundary |= _scalars(timeouts, slice(None), i, count, "timeouts", True).astype(bool)
            indices = np.flatnonzero(~boundary[:-1])
        _capacity(len(indices), max_size)
        agents = []
        for agent_id in selected:
            agent = {
                "obs": _features(obs, indices, agent_id, count, "obs"),
                "action": _features(actions, indices, agent_id, count, "action"),
                "rewards": _scalars(rewards, indices, agent_id, count, "rewards"),
                "done": _scalars(done, indices + 1 if mode == "omiga" else indices, agent_id, count, "done", True),
                "next_obs": _features(next_obs if mode == "explicit" else obs,
                                      indices if mode == "explicit" else indices + 1, agent_id, count, "next_obs"),
            }
            if timeouts is not None:
                agent["done"] = np.maximum(agent["done"], _scalars(timeouts, indices + 1 if mode == "omiga" else indices,
                                                                 agent_id, count, "timeouts", True))
            if state is not None:
                agent["state"] = _features(state, indices, agent_id, count, "state", shared=True)
                agent["next_state"] = _features(next_state if mode == "explicit" else state,
                                                indices if mode == "explicit" else indices + 1, agent_id, count, "next_state", shared=True)
            if mode != "explicit" or next_action is not None:
                agent["next_action"] = _features(next_action if mode == "explicit" else actions,
                                                 indices if mode == "explicit" else indices + 1, agent_id, count, "next_action")
            agents.append(agent)
        return agents, selected, {"transition_mode": mode, "source_steps": length, "discarded_steps": length - len(indices),
                                  "state_layout": "shared" if state is not None and state.ndim == 2 else "per_agent"}


def _ogmarl(path, agent_ids, max_size, observation_prefix):
    names = ("obs", "actions", "rewards", "path_lengths", "discounts")
    arrays = {}
    for name in names:
        try:
            arrays[name] = np.load(path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise DatasetError(f"{name}.npy: {error}") from error
    obs, actions, rewards, lengths, discounts = (arrays[name] for name in names)
    if obs.ndim != 3:
        raise DatasetError(f"obs.npy: expected [steps, agents, features], got {obs.shape}")
    size, count, _ = obs.shape
    _capacity(size, max_size)
    selected = _ids(agent_ids, range(count))
    for name, array in (("actions", actions), ("rewards", rewards), ("discounts", discounts)):
        if array.ndim == 0 or len(array) != size:
            raise DatasetError(f"{name}.npy: expected aligned length {size}, got {array.shape}")
    lengths = np.asarray(lengths)
    if lengths.ndim != 1:
        raise DatasetError("path_lengths must be a one-dimensional array")
    _finite(lengths, "path_lengths")
    if np.any(lengths <= 0) or np.any(lengths != np.floor(lengths)) or int(lengths.sum()) != size:
        raise DatasetError("path_lengths must be positive integers summing to the number of timesteps")
    ends = np.cumsum(lengths.astype(np.int64)) - 1
    done = np.zeros(size, dtype=np.float32)
    done[ends] = 1
    if observation_prefix is None:
        observation_prefix = count
        if obs.shape[2] <= count:
            raise DatasetError("OG-MARL observations have no features after agent ID; specify observation_prefix explicitly")
        for start in range(0, size, 65536):
            if not np.array_equal(obs[start:start + 65536, :, :count],
                                  np.broadcast_to(np.eye(count), (len(obs[start:start + 65536]), count, count))):
                raise DatasetError("OG-MARL agent ID prefix does not match one-hot IDs; specify observation_prefix=0 to retain unprefixed observations")
    if not isinstance(observation_prefix, (int, np.integer)) or not 0 <= observation_prefix < obs.shape[2]:
        raise DatasetError("observation_prefix must be an integer within the observation width")
    agents = []
    indices = np.arange(size)
    for agent_id in selected:
        agent_discount = _scalars(discounts, indices, agent_id, count, "discounts")
        if not np.array_equal(agent_discount, 1 - done):
            raise DatasetError("discounts do not match path_lengths episode boundaries")
        agent_obs = _array(obs[:, agent_id, observation_prefix:], "obs")
        action = _features(actions, indices, agent_id, count, "action")
        successor = np.zeros_like(agent_obs)
        next_action = np.zeros_like(action)
        valid = np.flatnonzero(done == 0)
        successor[valid] = agent_obs[valid + 1]
        next_action[valid] = action[valid + 1]
        agents.append({"obs": agent_obs, "action": action, "rewards": _scalars(rewards, indices, agent_id, count, "rewards"),
                       "done": done.copy(), "next_obs": successor, "next_action": next_action,
                       "next_action_valid": 1 - done})
    return agents, selected, {"episode_count": len(lengths), "observation_prefix": observation_prefix,
                              "terminal_next_obs": "zeros; export contains no terminal successor"}


def load_dataset(path, format="auto", agent_ids=None, reward_scale=1.0, episode_length=25,
                 transition_mode="auto", observation_dims=None, action_dims=None,
                 max_size=None, observation_prefix=None, terminal_policy="auto"):
    """Load a local dataset without downloading data or guessing feature widths.

    Formats: ``numpy`` (legacy per-agent NPY), ``omiga`` (snapshot HDF5),
    ``hdf5`` (explicit transitions or a declared snapshot convention), and
    ``ogmarl`` (trajectory NPY export). ``auto`` detects the local file schema.
    Agent IDs select physical agents in the requested order. Declared widths
    may remove verified zero padding; they never discard nonzero features.
    Fixed-horizon MPE terminals require ``terminal_policy='combine'`` or
    ``'fixed_horizon'``. Auto preserves supplied terminal flags.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise DatasetError(f"dataset path does not exist: {path}")
    if not np.isscalar(reward_scale) or np.asarray(reward_scale).dtype.kind not in "biuf" or not np.isfinite(reward_scale):
        raise DatasetError("reward_scale must be a finite scalar")
    if transition_mode not in {"auto", "explicit", "omiga", "adjacent"}:
        raise DatasetError("transition_mode must be auto, explicit, omiga, or adjacent")
    if format == "auto":
        if path.is_file() and h5py.is_hdf5(path):
            format = "hdf5"
        elif path.is_dir() and (path / "path_lengths.npy").exists():
            format = "ogmarl"
        elif path.is_dir() and any(path.glob("obs_*.npy")):
            format = "numpy"
        else:
            raise DatasetError("cannot detect dataset format; expected legacy NPY directory or HDF5 file")
    if format in {"numpy", "npy"}:
        if transition_mode not in {"auto", "explicit"}:
            raise DatasetError("legacy NumPy files already contain explicit transitions")
        agents, selected, metadata = _legacy_numpy(path, agent_ids, max_size)
    elif format in {"omiga", "hdf5"}:
        if format == "omiga":
            if transition_mode not in {"auto", "omiga"}:
                raise DatasetError("format='omiga' requires OMIGA snapshot terminal semantics")
            transition_mode = "omiga"
        agents, selected, metadata = _hdf5(path, agent_ids, max_size, transition_mode)
    elif format == "ogmarl":
        if transition_mode not in {"auto", "adjacent"}:
            raise DatasetError("OG-MARL trajectory exports use current-row terminal semantics")
        agents, selected, metadata = _ogmarl(path, agent_ids, max_size, observation_prefix)
    else:
        raise DatasetError(f"unknown dataset format {format!r}")
    _declared_dims(agents, "obs", observation_dims)
    _declared_dims(agents, "action", action_dims)
    _terminals(agents, terminal_policy, episode_length)
    metadata["reward_scale"] = float(reward_scale)
    metadata["terminal_policy"] = "provided" if terminal_policy == "auto" else terminal_policy
    if terminal_policy in {"combine", "fixed_horizon"}:
        metadata["episode_length"] = episode_length
    for agent in agents:
        agent["rewards"] = agent["rewards"] * reward_scale
    dataset = Dataset(agents, selected, format, metadata)
    # Mask stored successor actions after any explicit terminal override.
    for agent in dataset.agents:
        terminal = agent["done"] != 0
        agent["next_action"][terminal] = 0
        agent["next_action_valid"][terminal] = 0
    return dataset

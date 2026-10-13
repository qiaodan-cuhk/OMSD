"""Explicit, lazy environment backends and a small canonical multi-agent API.

Main profiles preserve MPE and OG-MARL/MADiff MuJoCo 2.1 conventions.
Importing this module does not import Gym or a simulator.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Callable, Optional

import numpy as np


class EnvironmentUnavailable(RuntimeError):
    """An explicitly selected environment backend cannot be constructed."""


_MPE_SCENARIOS = {"simple_spread", "simple_tag", "simple_world"}
_MUJOCO_210 = {
    "2ant": ("Ant-v2", "2x4"),
    "4ant": ("Ant-v2", "4x2"),
    "2halfcheetah": ("HalfCheetah-v2", "2x3"),
}


def get_env_defaults(profile: str, env_id: str) -> dict:
    """Return the branch-specific defaults, without requiring a simulator.

    ``scenario`` names the underlying Gym task, while ``env_id`` may be a
    dataset alias such as ``2ant``. Explicit configuration overrides these
    defaults; unknown task/profile combinations fail rather than guess dims.
    """
    profile = {"mujoco210": "mamujoco210"}.get(profile, profile)
    if env_id in _MPE_SCENARIOS:
        return {
            "backend": "mpe",
            "scenario": env_id,
            "episode_length": 25,
            "discrete_action": False,
        }
    if profile == "mamujoco210" and env_id in _MUJOCO_210:
        scenario, agent_conf = _MUJOCO_210[env_id]
        return {
            "backend": "mamujoco210",
            "scenario": scenario,
            "agent_conf": agent_conf,
            "agent_obsk": 1,
            "episode_length": 1000,
            "global_categories": "qvel,qpos",
        }
    raise ValueError(
        f"No historical environment defaults for profile={profile!r}, "
        f"env_id={env_id!r}. Set an explicit supported backend and its settings."
    )


def _space_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    spaces = getattr(value, "spaces", None)
    if isinstance(spaces, Mapping):
        return list(spaces.values())
    if spaces is not None:
        return list(spaces)
    return [value]


def _space_dim(space) -> int:
    # A discrete action is represented as a one-hot vector by the historical MPE.
    if getattr(space, "nvec", None) is not None:
        return int(np.asarray(space.nvec).sum())
    if getattr(space, "n", None) is not None:
        return int(space.n)
    shape = getattr(space, "shape", None)
    if shape is not None and len(shape):
        return int(np.prod(shape))
    children = getattr(space, "spaces", None)
    if children is not None:
        if isinstance(children, Mapping):
            children = children.values()
        return sum(_space_dim(child) for child in children)
    raise ValueError(f"Cannot infer a dimension from space {type(space).__name__}")


def _shape_dims(value, count: int):
    if value is None:
        return None
    if np.isscalar(value):
        return [int(value)] * count
    if isinstance(value, list) and len(value) == count:
        return [int(np.prod(dim)) for dim in value]
    return [int(np.prod(value))] * count


def _agent_values(value, count: int, dtype, label: str):
    array = np.asarray(value, dtype=dtype)
    if array.size == 1:
        return [array.reshape(-1)[0].item()] * count
    if array.size != count:
        raise ValueError(f"{label} has {array.size} entries; expected {count}")
    return array.reshape(count).tolist()


class Environment:
    """Adapt 3/4/5/6-item APIs while retaining heterogeneous agent dimensions.

    ``agent_ids`` optionally selects the learner team. A subset rollout needs
    ``other_action_fn(full_obs)`` returning a mapping of the remaining agent IDs
    to actions. No implicit random or pretrained opponent policy is introduced.
    All returned observations/rewards/dones follow ``agent_ids`` order.
    """

    def __init__(
        self,
        backend,
        *,
        episode_length: Optional[int] = None,
        agent_ids=None,
        other_action_fn: Optional[Callable] = None,
    ):
        self.backend = backend
        self.episode_length = episode_length
        self.other_action_fn = other_action_fn
        self._steps = 0
        self.last_info = {}
        self._last_full_obs = None
        info_fn = getattr(backend, "get_env_info", None)
        self.env_info = dict(info_fn() or {}) if callable(info_fn) else {}
        self._full_action_spaces = _space_list(
            getattr(backend, "action_space", self.env_info.get("action_spaces"))
        )
        observation_spaces = _space_list(getattr(backend, "observation_space", None))
        count = self.env_info.get("n_agents")
        if count is None:
            for name in ("n_agents", "n", "num_agents"):
                count = getattr(backend, name, None)
                if count is not None:
                    break
        if count is None:
            spaces = self._full_action_spaces or observation_spaces
            if spaces is None:
                raise ValueError("Backend must expose agent count or per-agent spaces")
            count = len(spaces)
        self._count = int(count)
        agents = getattr(backend, "agents", None)
        if agents is not None and len(agents) == self._count:
            self.full_agent_ids = [
                str(getattr(agent, "name", f"agent_{i}"))
                for i, agent in enumerate(agents)
            ]
        else:
            self.full_agent_ids = [f"agent_{i}" for i in range(self._count)]
        if len(set(self.full_agent_ids)) != self._count:
            raise ValueError("Backend exposes duplicate agent IDs")
        self.agent_ids = (
            [self.full_agent_ids[int(name)] if isinstance(name, (int, np.integer)) else name
             for name in agent_ids]
            if agent_ids is not None else self.full_agent_ids[:]
        )
        if not self.agent_ids or len(set(self.agent_ids)) != len(self.agent_ids):
            raise ValueError("Select a nonempty list of unique agent IDs")
        unknown = set(self.agent_ids) - set(self.full_agent_ids)
        if unknown:
            raise ValueError(f"Unknown environment agent IDs: {sorted(unknown)}")
        self._indices = [self.full_agent_ids.index(name) for name in self.agent_ids]
        policy_indices = (
            [i for i, agent in enumerate(agents) if getattr(agent, "adversary", False)]
            if agents is not None
            else []
        )
        self.policy_indices = policy_indices or list(range(self._count))
        self.policy_agent_ids = [self.full_agent_ids[i] for i in self.policy_indices]
        self.agent_types = (
            ["adversary" if getattr(agents[i], "adversary", False) else "agent"
             for i in self._indices]
            if agents is not None else ["agent"] * len(self.agent_ids)
        )
        self.nagents = len(self.agent_ids)
        self.n = self.nagents
        self.n_agents = self.nagents

        if self._full_action_spaces:
            if len(self._full_action_spaces) != self._count:
                raise ValueError("Backend action spaces do not match agent count")
            self._full_action_dims = [_space_dim(s) for s in self._full_action_spaces]
            self.action_space = [self._full_action_spaces[i] for i in self._indices]
        else:
            self._full_action_dims = _shape_dims(
                self.env_info.get("action_dims", self.env_info.get("n_actions")),
                self._count,
            )
            if self._full_action_dims is None:
                raise ValueError("Backend must expose action spaces or action dimensions")
            self.action_space = None
        self.action_dims = [self._full_action_dims[i] for i in self._indices]
        self.observation_space = (
            [observation_spaces[i] for i in self._indices]
            if observation_spaces is not None
            else None
        )
        # MaMuJoCo's compatibility observation_space incorrectly has n_agents
        # dimensions. Its actual observations must take precedence.
        obs_fn = getattr(backend, "get_obs", None)
        if callable(obs_fn):
            self._record_observations(obs_fn())
        else:
            dims = _shape_dims(self.env_info.get("obs_shape"), self._count)
            if dims is None and observation_spaces is not None:
                dims = [_space_dim(space) for space in observation_spaces]
            self._full_obs_dims = dims
            self.obs_dims = [dims[i] for i in self._indices] if dims else None
        state_dims = _shape_dims(self.env_info.get("state_shape"), self._count)
        self.state_dims = [state_dims[i] for i in self._indices] if state_dims else None

    def _joint_observations(self, observations):
        if isinstance(observations, Mapping):
            observations = [observations[name] for name in self.full_agent_ids]
        elif self._count == 1:
            array = np.asarray(observations)
            if array.ndim == 1:
                observations = [array]
        if len(observations) != self._count:
            raise ValueError(
                f"Backend returned {len(observations)} observations for {self._count} agents"
            )
        return [np.asarray(obs, dtype=np.float32).reshape(-1) for obs in observations]

    def _record_observations(self, observations):
        obs = self._joint_observations(observations)
        dims = [item.size for item in obs]
        previous = getattr(self, "_full_obs_dims", None)
        if previous is not None and previous != dims:
            raise ValueError(f"Observation dimensions changed: expected {previous}, got {dims}")
        self._full_obs_dims = dims
        self.obs_dims = [dims[i] for i in self._indices]
        self._last_full_obs = obs
        return [obs[i] for i in self._indices]

    def _looks_joint(self, value):
        # Do not confuse a tuple of two individual observations with (obs, info).
        if isinstance(value, Mapping):
            return all(name in value for name in self.full_agent_ids)
        if isinstance(value, (list, tuple)):
            return len(value) == self._count and all(np.asarray(x).ndim > 0 for x in value)
        return isinstance(value, np.ndarray) and value.ndim >= 2 and len(value) == self._count

    def reset(self, **kwargs):
        result = self.backend.reset(**kwargs)
        self._steps = 0
        self.last_info = {}
        if isinstance(result, tuple):
            if len(result) == 2 and isinstance(result[1], Mapping):
                result, reset_info = result
                self.last_info.update(reset_info)
            elif len(result) in (2, 3) and self._looks_joint(result[0]):
                self.last_info["state"] = result[1]
                if len(result) == 3:
                    self.last_info["available_actions"] = result[2]
                result = result[0]
        if result is None:
            result = self.backend.get_obs()
        return self._record_observations(result)

    def get_obs(self):
        obs_fn = getattr(self.backend, "get_obs", None)
        if callable(obs_fn):
            return self._record_observations(obs_fn())
        if self._last_full_obs is None:
            return self.reset()
        return [self._last_full_obs[i] for i in self._indices]

    def get_state(self):
        getter = getattr(self.backend, "get_state", None)
        if callable(getter):
            return getter()
        return self.last_info.get("state")

    def _joint_actions(self, actions):
        if len(actions) != self.nagents:
            raise ValueError(f"Expected {self.nagents} actions, received {len(actions)}")
        for i, (action, dim) in enumerate(zip(actions, self.action_dims)):
            if np.asarray(action).size != dim:
                raise ValueError(f"Action for {self.agent_ids[i]} has wrong dimension; expected {dim}")
        full = [None] * self._count
        for index, action in zip(self._indices, actions):
            full[index] = np.asarray(action).reshape(-1)
        if len(self._indices) != self._count:
            if self.other_action_fn is None:
                remaining = [self.full_agent_ids[i] for i, action in enumerate(full) if action is None]
                raise EnvironmentUnavailable(
                    f"A rollout for the selected learner team needs an explicit opponent "
                    f"action provider for {remaining}. Supply other_action_fn; no policy "
                    "checkpoint is loaded implicitly."
                )
            extra = self.other_action_fn(self._last_full_obs)
            for i, action in enumerate(full):
                if action is None:
                    name = self.full_agent_ids[i]
                    if name not in extra:
                        raise ValueError(f"Opponent action provider omitted {name}")
                    action = np.asarray(extra[name]).reshape(-1)
                    if action.size != self._full_action_dims[i]:
                        raise ValueError(f"Opponent action for {name} has wrong dimension")
                    full[i] = action
        return full

    def step(self, actions):
        result = self.backend.step(self._joint_actions(actions))
        if not isinstance(result, (tuple, list)):
            raise ValueError("Backend step must return a tuple or list")
        truncated = None
        native_terminated = None
        if len(result) == 3:
            rewards, dones, info = result
            observations = self.backend.get_obs()
        elif len(result) == 4:
            observations, rewards, dones, info = result
        elif len(result) == 5:
            observations, rewards, terminated, truncated, info = result
            dones = np.logical_or(terminated, truncated)
            native_terminated = _agent_values(terminated, self._count, bool, "Terminated")
        elif len(result) == 6:
            observations, state, rewards, dones, info, available_actions = result
        else:
            raise ValueError(f"Unsupported backend step tuple length: {len(result)}")
        info = dict(info) if isinstance(info, Mapping) else {"agent_info": info}
        if len(result) == 6:
            info["state"] = state
            info["available_actions"] = available_actions
        full_rewards = _agent_values(rewards, self._count, np.float32, "Rewards")
        full_dones = _agent_values(dones, self._count, bool, "Dones")
        if truncated is None:
            timeout = info.get("TimeLimit.truncated", info.get("episode_limit", info.get("bad_transition", False)))
            full_truncated = _agent_values(timeout, self._count, bool, "Truncated")
            agent_info = info.get("agent_info")
            if isinstance(agent_info, (list, tuple)):
                if len(agent_info) != self._count:
                    raise ValueError("Per-agent info length differs from backend agent count")
                markers = [bool(item.get("TimeLimit.truncated", item.get("bad_transition", False)))
                           if isinstance(item, Mapping) else False for item in agent_info]
                full_truncated = [timeout or marker for timeout, marker in zip(full_truncated, markers)]
            full_truncated = [done and timeout for done, timeout in zip(full_dones, full_truncated)]
        else:
            full_truncated = _agent_values(truncated, self._count, bool, "Truncated")
        self._steps += 1
        if self.episode_length is not None and self._steps >= self.episode_length:
            full_truncated = [timeout or not done for timeout, done in zip(full_truncated, full_dones)]
            full_dones = [True] * self._count
        full_terminated = native_terminated or [
            done and not timeout for done, timeout in zip(full_dones, full_truncated)
        ]
        info["terminated"] = [full_terminated[i] for i in self._indices]
        info["truncated"] = [full_truncated[i] for i in self._indices]
        info["bootstrap_mask"] = [float(not full_terminated[i]) for i in self._indices]
        self.last_info = info
        return (
            self._record_observations(observations),
            [full_rewards[i] for i in self._indices],
            [full_dones[i] for i in self._indices],
            info,
        )

    def seed(self, seed: int):
        # Historical MaMuJoCo.seed is a no-op. Seed its actual Gym wrapper too.
        if hasattr(self.backend, "timelimit_env"):
            target = self.backend.timelimit_env
        else:
            target = self.backend
        seed_fn = getattr(target, "_seed", None)
        if callable(seed_fn):
            return seed_fn(seed)
        seed_fn = getattr(target, "seed", None)
        if callable(seed_fn):
            return seed_fn(seed)
        return target.reset(seed=seed)

    def close(self):
        # MPE inherits an old Gym close implementation that may not close viewers.
        for viewer in getattr(self.backend, "viewers", []):
            if viewer is not None:
                viewer.close()
        close_fn = getattr(self.backend, "close", None)
        if callable(close_fn):
            try:
                return close_fn()
            except NotImplementedError:
                pass
        wrapped = getattr(self.backend, "wrapped_env", None)
        if wrapped is not None and callable(getattr(wrapped, "close", None)):
            return wrapped.close()


def _make_mpe(settings):
    scenario_name = settings["scenario"]
    if scenario_name not in _MPE_SCENARIOS:
        raise ValueError(f"Unsupported historical MPE scenario: {scenario_name}")
    module = importlib.import_module(f"omsd._vendor.mpe.scenarios.{scenario_name}")
    env_module = importlib.import_module("omsd._vendor.mpe.environment")
    scenario = module.Scenario()
    world = scenario.make_world()
    return env_module.MultiAgentEnv(
        world,
        scenario.reset_world,
        scenario.reward,
        scenario.observation,
        getattr(scenario, "benchmark_data", None) if scenario_name == "simple_spread" else None,
        discrete_action=settings.get("discrete_action", False),
    )


def _make_mujoco(settings, module_path):
    module = importlib.import_module(module_path)
    env_args = {
        "scenario": settings["scenario"],
        "episode_limit": settings["episode_length"],
        "agent_conf": settings["agent_conf"],
        "agent_obsk": settings.get("agent_obsk"),
    }
    for field in ("global_categories", "k_categories", "obs_add_global_pos"):
        if field in settings:
            env_args[field] = settings[field]
    return module.MujocoMulti(env_args=env_args)


def _make_210(settings):
    return _make_mujoco(settings, "third_party.multiagent_mujoco.multiagent_mujoco.mujoco_multi")


BACKENDS = {
    "mpe": _make_mpe,
    "mamujoco210": _make_210,
}


def load_opponent_policy(path, environment: Environment) -> Callable:
    """Load a user-supplied historical DDPG prey policy as an action callback.

    Supported files contain the original ``agent_params`` list (all agents in
    backend order or only the unselected opponents), a single ``policy`` state
    dict, or a raw single-policy state dict. Only policy weights are used; critic
    and optimizer state are unnecessary for evaluation. The payload must contain
    tensor state dictionaries, using weights-only loading when PyTorch supports it.
    """
    import torch
    from .checkpoints import load_tensor_payload
    from utils.networks import MLPNetwork

    remaining = [i for i in range(environment._count) if i not in environment._indices]
    if not remaining:
        raise ValueError("An opponent checkpoint requires an environment with an unselected agent team")
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Opponent checkpoint does not exist: {path}")
    try:
        payload = load_tensor_payload(path)
    except Exception as exc:
        raise ValueError(
            "Opponent checkpoint must contain tensor state dictionaries; "
            f"model objects are not supported. Cause: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Opponent checkpoint must contain a policy state dict or agent_params mapping")
    if "agent_params" in payload:
        entries = payload["agent_params"]
        if not isinstance(entries, (tuple, list)):
            raise ValueError("Opponent agent_params must be a list in backend agent order")
        if len(entries) == environment._count:
            entries = [entries[index] for index in remaining]
        elif len(entries) != len(remaining):
            raise ValueError(
                f"Opponent checkpoint has {len(entries)} agent entries; expected "
                f"{environment._count} full-team or {len(remaining)} opponent-only entries"
            )
    elif len(remaining) == 1:
        entries = [payload]
    else:
        raise ValueError("Multiple opponent policies need an agent_params list")

    if environment._full_obs_dims is None:
        environment.reset()
    policies = []
    for index, entry in zip(remaining, entries):
        if not isinstance(entry, Mapping):
            raise ValueError("Every opponent agent entry must contain policy tensor weights")
        state_dict = entry.get("policy", entry)
        if not isinstance(state_dict, Mapping) or not isinstance(state_dict.get("fc1.weight"), torch.Tensor):
            raise ValueError("Opponent policy must be a historical MLPNetwork state dict with fc1.weight")
        required = ("fc1.weight", "fc2.weight", "fc3.weight")
        if any(not isinstance(state_dict.get(name), torch.Tensor) for name in required):
            raise ValueError("Opponent MLPNetwork checkpoint is missing linear-layer tensor weights")
        hidden_dim, obs_dim = state_dict["fc1.weight"].shape
        action_dim = state_dict["fc3.weight"].shape[0]
        expected_obs = environment._full_obs_dims[index]
        expected_action = environment._full_action_dims[index]
        if (obs_dim, action_dim) != (expected_obs, expected_action):
            name = environment.full_agent_ids[index]
            raise ValueError(
                f"Opponent policy dimensions for {name} are ({obs_dim}, {action_dim}); "
                f"environment expects ({expected_obs}, {expected_action})"
            )
        action_space = environment._full_action_spaces[index] if environment._full_action_spaces else None
        discrete = getattr(action_space, "n", None) is not None
        policy = MLPNetwork(obs_dim, action_dim, hidden_dim=hidden_dim,
                            constrain_out=True, discrete_action=discrete)
        policy.load_state_dict(state_dict, strict=True)
        policy.eval()
        policies.append((index, policy))

    def opponent_actions(full_observations):
        if full_observations is None:
            raise ValueError("Reset the environment before requesting opponent actions")
        actions = {}
        with torch.no_grad():
            for index, policy in policies:
                observation = torch.as_tensor(full_observations[index], dtype=torch.float32).reshape(1, -1)
                # Match historical DDPGAgent.step(explore=False), including tanh
                # for continuous policies and the final [-1, 1] clamp.
                action = policy(observation).clamp(-1, 1).squeeze(0).numpy()
                actions[environment.full_agent_ids[index]] = action
        return actions

    return opponent_actions


def make_environment(config, *, other_action_fn=None) -> Environment:
    """Create only the selected backend, then normalize its public API.

    Both an ExperimentConfig-like object and a dict are accepted. Optional
    backends are never imported until this function is called.
    """
    values = dict(config) if isinstance(config, Mapping) else vars(config)
    env_id = values["env_id"]
    profile = values.get("profile") or values.get("backend") or (
        "mpe" if env_id in _MPE_SCENARIOS else "mamujoco210"
    )
    try:
        settings = get_env_defaults(profile, env_id)
    except ValueError:
        if not values.get("backend"):
            raise
        settings = {"scenario": env_id, "episode_length": 1000}
    for key in ("backend", "agent_conf", "agent_obsk", "episode_length", "discrete_action", "global_categories"):
        if values.get(key) is not None and values.get(key) != "auto":
            settings[key] = values[key]
    backend_name = settings.get("backend", profile)
    if backend_name not in BACKENDS:
        raise ValueError(f"Unknown backend {backend_name!r}; choose {', '.join(BACKENDS)}")
    try:
        backend = BACKENDS[backend_name](settings)
    except Exception as exc:
        if backend_name == "mpe":
            requirement = "Install the main dependencies: pip install -r requirements.txt (Gym 0.12.1)."
        else:
            requirement = (
                "Install requirements.txt and the external MuJoCo 2.1 "
                "runtime/build libraries. No simulator binary is downloaded by OMSD."
            )
        raise EnvironmentUnavailable(f"Cannot start backend {backend_name!r}. {requirement} Cause: {exc}") from exc
    try:
        environment = Environment(
            backend,
            episode_length=settings.get("episode_length"),
            agent_ids=values.get("agent_ids"),
            other_action_fn=other_action_fn,
        )
        environment.seed(values.get("seed", 0))
        return environment
    except Exception:
        close = getattr(backend, "close", None)
        if callable(close):
            close()
        raise

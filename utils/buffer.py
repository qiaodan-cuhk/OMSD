"""Compatibility facade for the historical offline ReplayBuffer API.

Storage now grows with the loaded dataset instead of allocating ``max_steps``
rows per field. Capacity remains an explicit limit, never a truncation rule.
New code should use :mod:`omsd.data` directly.
"""

import numpy as np
import torch

from omsd.data import DatasetError, load_dataset


class ReplayBuffer:
    def __init__(self, max_steps, num_agents, obs_dims, ac_dims, device,
                 is_mamujoco=False, state_dims=None, store_on_gpu=False,
                 dtype=np.float32):
        if not isinstance(max_steps, (int, np.integer)) or max_steps <= 0:
            raise DatasetError("max_steps must be a positive integer")
        if num_agents <= 0 or len(obs_dims) != num_agents or len(ac_dims) != num_agents:
            raise DatasetError("obs_dims and ac_dims must describe each agent")
        if state_dims is not None and len(state_dims) != num_agents:
            raise DatasetError("state_dims must describe each agent")
        if np.dtype(dtype).kind != "f":
            raise DatasetError("ReplayBuffer dtype must be floating point")
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.obs_dims = list(obs_dims)
        self.ac_dims = list(ac_dims)
        self.state_dims = None if state_dims is None else list(state_dims)
        self.device = device
        self.is_mamujoco = is_mamujoco
        self.store_on_gpu = store_on_gpu
        self.dtype = dtype
        self.filled_i = 0
        self.curr_i = 0
        self.dataset = None
        self._device_agents = None
        self._publish_buffers([])

    def __len__(self):
        return self.filled_i

    def _publish_buffers(self, agents):
        mapping = {"obs_buffs": "obs", "ac_buffs": "action", "rew_buffs": "rewards",
                   "next_obs_buffs": "next_obs", "done_buffs": "done",
                   "state_buffs": "state", "next_state_buffs": "next_state"}
        for name, key in mapping.items():
            setattr(self, name, [a[key] for a in agents if key in a])

    def load_dataset(self, path, format="auto", rew_scale=1.0, **kwargs):
        dataset = load_dataset(path, format=format, reward_scale=rew_scale,
                               observation_dims=self.obs_dims, action_dims=self.ac_dims,
                               max_size=self.max_steps, **kwargs)
        if dataset.num_agents != self.num_agents:
            raise DatasetError(f"expected {self.num_agents} agents, loaded {dataset.num_agents}")
        if self.state_dims is not None and dataset.state_dims is not None and dataset.state_dims != self.state_dims:
            raise DatasetError(f"expected state dimensions {self.state_dims}, loaded {dataset.state_dims}")
        self.dataset = dataset
        self.filled_i = dataset.size
        self.curr_i = dataset.size % self.max_steps
        self.sum_reward = float(dataset.agents[0]["rewards"].sum())
        completed = float(dataset.agents[0]["done"].sum())
        self.ave_reward = self.sum_reward / completed if completed else float("nan")
        self._device_agents = None
        if self.store_on_gpu:
            self._tensorize_all()
        else:
            self._publish_buffers(dataset.agents)
        return dataset

    def load_batch_data(self, dir, rew_scale=1.0):
        # Historical MPE exports used a fixed 25-step horizon; MaMuJoCo
        # transitions already include reliable terminals.
        return self.load_dataset(dir, format="numpy", rew_scale=rew_scale,
                                 terminal_policy="provided" if self.is_mamujoco else "combine")

    def load_batch_data_omiga(self, dir, rew_scale=1.0):
        return self.load_dataset(dir, format="omiga", rew_scale=rew_scale)

    def load_batch_data_ogmarl(self, dir, rew_scale=1.0, load_to_gpu=False):
        self.store_on_gpu = load_to_gpu
        return self.load_dataset(dir, format="ogmarl", rew_scale=rew_scale)

    def _tensorize_all(self):
        if self.dataset is None:
            raise DatasetError("load a dataset before tensorizing the buffer")
        self._device_agents = [{key: torch.tensor(value, dtype=torch.float32, device=self.device)
                                for key, value in agent.items()} for agent in self.dataset.agents]
        self._publish_buffers(self._device_agents)

    def sample(self, N, to_gpu=False):
        if self.dataset is None:
            raise DatasetError("load a dataset before sampling")
        if self.store_on_gpu:
            if not isinstance(N, (int, np.integer)) or N <= 0:
                raise DatasetError("batch_size must be a positive integer")
            indices = torch.randint(self.filled_i, (N,), device=self.device)
            return [{key: value.index_select(0, indices) for key, value in agent.items()}
                    for agent in self._device_agents]
        return self.dataset.sample(N, device=self.device if to_gpu else "cpu")

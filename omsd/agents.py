"""Dataset-independent implementations of the OMIGA OMSD training route.

Policies always receive observations. SEQ factorizes the behavior distribution
in ``conditional_order`` while the centralized critic keeps actions in agent-id
order. Observation and action widths are explicit for every agent.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from algorithms.model import Dirac_Policy, ScoreNet_IDQL, TwinQ, ValueFunction


def marginal_prob_std(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """OMIGA's variance-preserving diffusion schedule (beta_0=.1, beta_1=20)."""
    log_alpha = -0.25 * t.square() * (20.0 - 0.1) - 0.5 * t * 0.1
    return log_alpha.exp(), (1.0 - (2.0 * log_alpha).exp()).sqrt()


def _dimensions(values: Sequence[int], name: str) -> List[int]:
    if not values or any(isinstance(v, bool) or int(v) != v or int(v) <= 0 for v in values):
        raise ValueError(f"{name} must contain positive integer dimensions")
    return [int(v) for v in values]


def parse_conditional_order(order: Optional[Union[str, Sequence[int]]], count: int) -> List[int]:
    """Parse the original ``2-0-1`` syntax and reject incomplete permutations."""
    if order is None or order == "":
        return list(range(count))
    if isinstance(order, str):
        try:
            order = [int(part) for part in order.split("-")]
        except ValueError as exc:
            raise ValueError("conditional_order must be an agent permutation such as 2-0-1") from exc
    order = list(order)
    if len(order) != count or any(isinstance(v, bool) or not isinstance(v, int) for v in order) or set(order) != set(range(count)):
        raise ValueError(f"conditional_order must contain every agent id 0..{count - 1} exactly once")
    return order


class IQLCritic(nn.Module):
    """Twin Q and expectile value updates, with a separately frozen Q target."""

    def __init__(self, action_dim: int, condition_dim: int, config: Any, trainable: bool = True):
        super().__init__()
        self.q0 = TwinQ(action_dim, condition_dim, layers=config.q_layer)
        self.q0_target = copy.deepcopy(self.q0).requires_grad_(False)
        self.vf = ValueFunction(condition_dim)
        self.trainable = trainable
        self.q_optimizer = torch.optim.Adam(self.q0.parameters(), lr=config.iql_critic_lr) if trainable else None
        self.value_optimizer = torch.optim.Adam(self.vf.parameters(), lr=config.iql_critic_lr) if trainable else None
        if not trainable:
            self.requires_grad_(False)
        self.expectile = config.tau
        self.discount = config.gamma

    def update(self, condition: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
               next_condition: torch.Tensor, done: torch.Tensor) -> Dict[str, float]:
        if not self.trainable:
            raise RuntimeError("A frozen policy-stage critic cannot be pretrained")
        with torch.no_grad():
            target_q = self.q0_target(action, condition)
            next_value = self.vf(next_condition)
        value = self.vf(condition)
        advantage = target_q - value
        value_loss = (torch.abs(self.expectile - (advantage < 0).float()) * advantage.square()).mean()
        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        self.value_optimizer.step()
        target = reward[:, None] + (1.0 - done[:, None]) * self.discount * next_value
        q_values = self.q0.both(action, condition)
        q_loss = sum(F.mse_loss(q_value, target) for q_value in q_values) / 2.0
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()
        # OMIGA uses .005 for Polyak averaging; config.tau is the IQL expectile.
        with torch.no_grad():
            for source, destination in zip(self.q0.parameters(), self.q0_target.parameters()):
                destination.copy_(0.005 * source + 0.995 * destination)
        return {"q_loss": float(q_loss.detach()), "value_loss": float(value_loss.detach()),
                "q": float(target_q.mean()), "value": float(value.detach().mean()),
                "advantage": float(advantage.detach().mean())}


class OfflineAgent(nn.Module):
    """OMSD variants sharing the same dataset-independent batch contract.

    ``samples`` is an aligned list of dictionaries in actual agent-id order.
    Each dictionary has ``obs``, ``action``, ``rewards``, ``next_obs``, ``done``.
    ``critic_input='state'`` additionally requires ``state`` and ``next_state``.
    A single ``state_dims`` entry means a shared global state in sample 0;
    one entry per agent means concatenated per-agent states for a joint critic.
    ``state_layout='shared'`` explicitly selects sample 0 for a joint critic
    when a dataset exposes the same global state in every agent dictionary.
    """

    behavior_model = "diffusion"

    def __init__(self, config: Any, obs_dims: Sequence[int], action_dims: Sequence[int],
                 state_dims: Optional[Sequence[int]] = None, state_layout: str = "per_agent",
                 stage: str = "all", agent_ids: Optional[Sequence[int]] = None):
        super().__init__()
        self.config = config
        if stage not in {"behavior", "critic", "policy", "all"}:
            raise ValueError("stage must be behavior, critic, policy, or all")
        self.stage = stage
        self.obs_dims = _dimensions(obs_dims, "obs_dims")
        self.action_dims = _dimensions(action_dims, "action_dims")
        self.n_agents = len(self.obs_dims)
        self.nagents = self.n_agents
        if len(self.action_dims) != self.n_agents:
            raise ValueError("obs_dims and action_dims must describe the same agents")
        self.mode = str(config.mode).upper()
        if self.mode not in {"SEQ", "IND", "CTDE", "JAL"}:
            raise ValueError("mode must be SEQ, IND, CTDE, or JAL")
        if str(config.behavior_model).lower() != self.behavior_model:
            raise ValueError(f"{type(self).__name__} only accepts {self.behavior_model} behavior models; "
                             "use the explicit ablation entrypoint for other models")
        self.behavior_layout = {"SEQ": "sequential", "JAL": "joint"}.get(self.mode, "independent")
        self.critic_layout = "independent" if self.mode == "IND" else "joint"
        self.agent_ids = list(range(self.n_agents)) if agent_ids is None else list(agent_ids)
        if len(self.agent_ids) != self.n_agents or len(set(self.agent_ids)) != self.n_agents:
            raise ValueError("agent_ids must describe every learner exactly once")
        selected_id = getattr(config, "pretrain_agent_id", None)
        self.pretrain_agent_index = None
        if selected_id is not None:
            if stage not in {"behavior", "critic"}:
                raise ValueError("pretrain_agent_id is only available for behavior or IND critic pretraining")
            if stage == "critic" and self.mode != "IND":
                raise ValueError("OMSD/CTDE use one joint critic; --pretrain-agent-id is only valid for IND critic pretraining")
            if self.mode == "JAL":
                raise ValueError("A joint behavior model cannot select a single agent")
            if selected_id not in self.agent_ids:
                raise ValueError("pretrain_agent_id must belong to the loaded dataset agent IDs: " + str(self.agent_ids))
            self.pretrain_agent_index = self.agent_ids.index(selected_id)
        self.device = torch.device(config.device)
        self.conditional_order = parse_conditional_order(config.conditional_order, self.n_agents)
        self.agent_to_stage = {agent_id: stage for stage, agent_id in enumerate(self.conditional_order)}
        self.critic_input = getattr(config, "critic_input", "obs")
        if self.critic_input not in {"obs", "state"}:
            raise ValueError("critic_input must be obs or state")
        self.state_dims = _dimensions(state_dims, "state_dims") if state_dims is not None else None
        if self.state_dims is not None and len(self.state_dims) not in {1, self.n_agents}:
            raise ValueError("state_dims needs one shared global dimension or one dimension per agent")
        if state_layout not in {"shared", "per_agent"}:
            raise ValueError("state_layout must be shared or per_agent")
        self.state_layout = state_layout
        if self.state_dims is not None and state_layout == "shared" and len(set(self.state_dims)) != 1:
            raise ValueError("shared states must have matching state dimensions")
        if stage != "behavior" and self.critic_input == "state" and self.state_dims is None:
            raise ValueError("critic_input=state requires dataset state_dims")
        if not 0.0 < config.tau < 1.0:
            raise ValueError("tau is the IQL expectile and must be between zero and one")
        if not 0.0 <= config.gamma <= 1.0:
            raise ValueError("gamma must be between zero and one")
        if config.n_policy_epochs <= 0:
            raise ValueError("n_policy_epochs must be positive")
        if stage != "critic":
            self._validate_behavior_config()

        if self.mode == "JAL":
            policy_conditions = [sum(self.obs_dims)]
            behavior_conditions = policy_conditions
            output_dims = [sum(self.action_dims)]
        else:
            policy_conditions = self.obs_dims
            behavior_conditions = [self.obs_dims[i] + sum(self.action_dims[j] for j in self._prefix(i))
                                   if self.mode == "SEQ" else self.obs_dims[i] for i in range(self.n_agents)]
            output_dims = self.action_dims
        self.behavior_condition_dims = list(behavior_conditions)
        self.behavior_indices = ([self.pretrain_agent_index] if stage == "behavior" and self.pretrain_agent_index is not None
                                 else list(range(len(behavior_conditions))))
        self.critic_indices = ([self.pretrain_agent_index] if stage == "critic" and self.pretrain_agent_index is not None
                               else list(range(self.n_agents if self.mode == "IND" else 1)))
        self.policies = nn.ModuleList()
        self.behaviors = nn.ModuleList()
        self.critics = nn.ModuleList()
        if stage in {"policy", "all"}:
            self.policies.extend(Dirac_Policy(action, condition, layer=config.policy_layer)
                                 for condition, action in zip(policy_conditions, output_dims))
        if stage in {"behavior", "policy", "all"}:
            self.behaviors.extend(self._make_behavior(behavior_conditions[index], output_dims[index])
                                  for index in self.behavior_indices)
        if stage in {"critic", "policy", "all"}:
            critic_dims = self.obs_dims if self.critic_input == "obs" else self.state_dims
            trainable = stage != "policy"
            if self.mode == "IND":
                conditions = critic_dims * self.n_agents if len(critic_dims) == 1 else critic_dims
                self.critics.extend(IQLCritic(self.action_dims[index], conditions[index], config, trainable=trainable)
                                    for index in self.critic_indices)
            else:
                condition = (critic_dims[0] if self.critic_input == "state" and self.state_layout == "shared"
                             else sum(critic_dims))
                self.critics.append(IQLCritic(sum(self.action_dims), condition, config, trainable=trainable))
        if stage == "policy":
            self.behaviors.requires_grad_(False).eval()
            self.critics.eval()
        self.to(self.device)
        self.behavior_optimizers = ([torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
                                     for model in self.behaviors] if stage in {"behavior", "all"} else [])
        self.policy_optimizers = [torch.optim.Adam(model.parameters(), lr=config.dilac_lr) for model in self.policies]
        self.behavior_schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000000, eta_min=1e-5)
                                    for optimizer in self.behavior_optimizers] if config.lr_anneal else []
        self.policy_schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_policy_epochs * 10000, eta_min=0.0)
                                  for optimizer in self.policy_optimizers]

    def _validate_behavior_config(self) -> None:
        if self.config.t_GassProj_dims < 2 or self.config.t_GassProj_dims % 2:
            raise ValueError("t_GassProj_dims must be a positive even number")

    def _make_behavior(self, condition_dim: int, action_dim: int) -> nn.Module:
        return ScoreNet_IDQL(condition_dim + action_dim, action_dim, marginal_prob_std,
                             embed_dim=self.config.t_GassProj_dims, args=self.config)

    def _require_stage(self, stage: str) -> None:
        if self.stage not in {stage, "all"}:
            raise RuntimeError(f"{stage} updates require a {stage}-stage trainer, got {self.stage}")

    def _prefix(self, agent_id: int) -> List[int]:
        return self.conditional_order[:self.agent_to_stage[agent_id]]

    def _matrix(self, value: Any, width: int, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if tensor.ndim != 2 or tensor.shape[1] != width:
            raise ValueError(f"{name} must have shape (batch, {width}), got {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values")
        return tensor

    def _batch(self, samples: Sequence[Mapping[str, Any]], next_observations: bool = False, critic_state: bool = False) -> List[Dict[str, torch.Tensor]]:
        if len(samples) != self.n_agents:
            raise ValueError(f"expected samples for {self.n_agents} aligned agents")
        batch = []
        count = None
        for i, sample in enumerate(samples):
            result = {"obs": self._matrix(sample["obs"], self.obs_dims[i], f"agent {i} obs")}
            if "action" in sample:
                result["action"] = self._matrix(sample["action"], self.action_dims[i], f"agent {i} action")
            if next_observations:
                result["next_obs"] = self._matrix(sample["next_obs"], self.obs_dims[i], f"agent {i} next_obs")
                for key in ("rewards", "done"):
                    vector = torch.as_tensor(sample[key], dtype=torch.float32, device=self.device)
                    if vector.ndim == 2 and vector.shape[1] == 1:
                        vector = vector[:, 0]
                    if vector.ndim != 1 or not torch.isfinite(vector).all():
                        raise ValueError(f"agent {i} {key} must be a finite batch vector")
                    result[key] = vector
                if ((result["done"] < 0) | (result["done"] > 1)).any():
                    raise ValueError("done values must be between zero and one")
            if critic_state and self.critic_input == "state" and (len(self.state_dims) != 1 or i == 0):
                width = self.state_dims[0] if len(self.state_dims) == 1 else self.state_dims[i]
                result["state"] = self._matrix(sample["state"], width, f"agent {i} state")
                if next_observations:
                    result["next_state"] = self._matrix(sample["next_state"], width, f"agent {i} next_state")
            count = result["obs"].shape[0] if count is None else count
            if count == 0 or any(tensor.shape[0] != count for tensor in result.values()):
                raise ValueError("all agent batch fields must have the same nonempty aligned batch size")
            batch.append(result)
        return batch

    def _critic_condition(self, batch: List[Dict[str, torch.Tensor]], agent_id: Optional[int] = None,
                          next_observations: bool = False) -> torch.Tensor:
        key = ("next_" if next_observations else "") + self.critic_input
        if self.critic_input == "state" and (len(self.state_dims) == 1 or (self.state_layout == "shared" and agent_id is None)):
            return batch[0][key]
        if agent_id is not None:
            return batch[agent_id][key]
        return torch.cat([sample[key] for sample in batch], dim=-1)

    def _behavior_condition(self, batch: List[Dict[str, torch.Tensor]], actions: List[torch.Tensor], agent_id: int) -> torch.Tensor:
        if self.mode == "JAL":
            return torch.cat([sample["obs"] for sample in batch], dim=-1)
        prefix = self._prefix(agent_id) if self.mode == "SEQ" else []
        return torch.cat([batch[agent_id]["obs"]] + [actions[i] for i in prefix], dim=-1)

    def update_behavior(self, samples: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
        self._require_stage("behavior")
        batch = self._batch(samples)
        actions = [sample["action"] for sample in batch]
        metrics = {}
        for position, (index, behavior) in enumerate(zip(self.behavior_indices, self.behaviors)):
            condition = self._behavior_condition(batch, actions, index)
            action = torch.cat(actions, dim=-1) if self.mode == "JAL" else actions[index]
            behavior.train()
            loss = self._behavior_loss(behavior, action, condition)
            optimizer = self.behavior_optimizers[position]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if self.behavior_schedulers:
                self.behavior_schedulers[position].step()
            metrics[f"behavior_{index}/loss"] = float(loss.detach())
        return metrics

    def update_critic(self, samples: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
        self._require_stage("critic")
        batch = self._batch(samples, next_observations=True, critic_state=True)
        metrics = {}
        for index, critic in zip(self.critic_indices, self.critics):
            individual = self.mode == "IND"
            condition = self._critic_condition(batch, index if individual else None)
            next_condition = self._critic_condition(batch, index if individual else None, next_observations=True)
            action = batch[index]["action"] if individual else torch.cat([sample["action"] for sample in batch], dim=-1)
            # OMIGA's joint IQL pretraining uses the first agent's team reward/done.
            reward = batch[index if individual else 0]["rewards"]
            done = batch[index if individual else 0]["done"]
            result = critic.update(condition, action, reward, next_condition, done)
            metrics.update({f"critic_{index}/{key}": value for key, value in result.items()})
        return metrics

    def _behavior_loss(self, behavior: nn.Module, action: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        t = torch.rand(action.shape[0], device=self.device) * (1.0 - 1e-3) + 1e-3
        alpha, std = marginal_prob_std(t)
        noise = torch.randn_like(action)
        prediction = behavior(action * alpha[:, None] + noise * std[:, None], t, condition)
        return (prediction - noise).square().sum(dim=-1).mean()

    def _behavior_vector(self, behavior: nn.Module, action: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        t = torch.rand(action.shape[0], device=self.device) * 0.96 + 0.02
        alpha, std = marginal_prob_std(t)
        noise = torch.randn_like(action)
        with torch.no_grad():
            prediction = behavior(action.detach() * alpha[:, None] + noise * std[:, None], t, condition)
            if "noise" in self.config.WT:
                prediction = prediction - noise
            if "VDS" in self.config.WT:
                weight = std.square()
            elif "stable" in self.config.WT:
                weight = torch.ones_like(std)
            elif "score" in self.config.WT:
                weight = alpha / std
            else:
                raise ValueError("WT must contain VDS, stable, or score")
        return prediction * weight[:, None]

    def update_policy(self, samples: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
        self._require_stage("policy")
        batch = self._batch(samples, critic_state=True)
        metrics = {}
        with torch.no_grad():
            if self.mode == "JAL":
                joint_observations = torch.cat([sample["obs"] for sample in batch], dim=-1)
                actions = list(self.policies[0](joint_observations).split(self.action_dims, dim=-1))
            else:
                actions = [policy(sample["obs"]).detach() for policy, sample in zip(self.policies, batch)]
        indices = [0] if self.mode == "JAL" else (self.conditional_order if getattr(self.config, "ordered_updates", False) else range(self.n_agents))
        for index in indices:
            behavior = self.behaviors[index]
            previous_training = behavior.training
            behavior.eval()
            try:
                policy = self.policies[index]
                observation = joint_observations if self.mode == "JAL" else batch[index]["obs"]
                action = policy(observation)
                condition = self._behavior_condition(batch, actions, index)
                behavior_vector = self._behavior_vector(behavior, action, condition)
                detached_action = action.detach().requires_grad_(True)
                if self.mode in {"JAL", "IND"}:
                    critic_action = detached_action
                    critic = self.critics[index]
                else:
                    actions[index] = detached_action
                    critic_action = torch.cat(actions, dim=-1)
                    critic = self.critics[0]
                critic_condition = self._critic_condition(batch, index if self.mode == "IND" else None)
                q1, q2 = critic.q0_target.both(critic_action, critic_condition)
                q = (q1 + q2) / 2.0
                guidance = torch.autograd.grad(q.sum(), detached_action)[0].detach()
                if self.config.regq:
                    guidance = guidance / guidance.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
                loss = ((behavior_vector * action).sum(dim=-1) - self.config.beta * (guidance * action).sum(dim=-1)).mean()
                optimizer = self.policy_optimizers[index]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                self.policy_schedulers[index].step()
                if self.mode in {"SEQ", "CTDE"}:
                    # OMIGA update/update_ordered retain the actor samples from
                    # before this minibatch's optimizer steps for both behavior
                    # prefixes and joint-Q inputs. Do not re-sample updated actors.
                    actions[index] = action.detach()
                metrics[f"policy_{index}/loss"] = float(loss.detach())
                metrics[f"policy_{index}/q"] = float(q.detach().mean())
                metrics[f"policy_{index}/guidance_norm"] = float(guidance.norm(dim=-1).mean())
                if all("action" in sample for sample in batch):
                    target_action = torch.cat([sample["action"] for sample in batch], dim=-1) if self.mode == "JAL" else batch[index]["action"]
                    metrics[f"policy_{index}/action_error"] = float((target_action - action.detach()).mean())
            finally:
                behavior.train(previous_training)
        return metrics

    @torch.no_grad()
    def act(self, observations: Sequence[Any]) -> List[Any]:
        """Return bounded actions; retain NumPy/Tensor representation per agent."""
        self._require_stage("policy")
        if len(observations) != self.n_agents:
            raise ValueError(f"expected observations for {self.n_agents} agents")
        batched = []
        single = []
        for index, value in enumerate(observations):
            tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            single.append(tensor.ndim == 1)
            batched.append(self._matrix(tensor[None] if tensor.ndim == 1 else tensor, self.obs_dims[index], f"agent {index} obs"))
        if len(set(tensor.shape[0] for tensor in batched)) != 1 or len(set(single)) != 1:
            raise ValueError("observations must use matching batch dimensions")
        if self.mode == "JAL":
            actions = self.policies[0](torch.cat(batched, dim=-1)).split(self.action_dims, dim=-1)
        else:
            actions = [policy(observation) for policy, observation in zip(self.policies, batched)]
        result = []
        for index, action in enumerate(actions):
            action = action[0] if single[index] else action
            result.append(action if isinstance(observations[index], torch.Tensor) else action.cpu().numpy())
        return result

    def _component(self, component: str) -> nn.ModuleList:
        names = {"behavior": self.behaviors, "behaviors": self.behaviors,
                 "critic": self.critics, "critics": self.critics,
                 "policy": self.policies, "policies": self.policies, "actor": self.policies}
        if component not in names:
            raise ValueError("component must be behavior, critic, or policy")
        if not names[component]:
            raise RuntimeError(f"{component} is unavailable in the {self.stage} stage")
        return names[component]

    def component_state_dict(self, component: str) -> Mapping[str, torch.Tensor]:
        modules = self._component(component)
        return {str(index) + "." + key: value for index, module in zip(self.component_indices(component), modules)
                for key, value in module.state_dict().items()}

    def load_component_state_dict(self, component: str, state: Mapping[str, torch.Tensor]) -> Any:
        indices = self.component_indices(component)
        remapped = {str(position) + key[len(str(index)):]: value
                    for position, index in enumerate(indices) for key, value in state.items()
                    if key.startswith(str(index) + ".")}
        if len(remapped) != len(state):
            raise ValueError("Checkpoint includes weights for an unconstructed agent")
        return self._component(component).load_state_dict(remapped, strict=True)

    def component_indices(self, component: str) -> List[int]:
        """Learner positions represented by each constructed submodel."""
        if component in {"behavior", "behaviors"}:
            return self.behavior_indices
        if component in {"critic", "critics"}:
            return self.critic_indices
        return list(range(len(self._component(component))))



class BehaviorTrainer(OfflineAgent):
    """Only behavior models and their pretraining optimizers."""

    def __init__(self, config: Any, obs_dims: Sequence[int], action_dims: Sequence[int],
                 state_dims: Optional[Sequence[int]] = None, state_layout: str = "per_agent", agent_ids=None):
        super().__init__(config, obs_dims, action_dims, state_dims, state_layout, stage="behavior", agent_ids=agent_ids)


class CriticTrainer(OfflineAgent):
    """Only IQL critics and their pretraining optimizers."""

    def __init__(self, config: Any, obs_dims: Sequence[int], action_dims: Sequence[int],
                 state_dims: Optional[Sequence[int]] = None, state_layout: str = "per_agent", agent_ids=None):
        super().__init__(config, obs_dims, action_dims, state_dims, state_layout, stage="critic", agent_ids=agent_ids)


class PolicyTrainer(OfflineAgent):
    """Policies with frozen pretrained behavior models and IQL critics."""

    def __init__(self, config: Any, obs_dims: Sequence[int], action_dims: Sequence[int],
                 state_dims: Optional[Sequence[int]] = None, state_layout: str = "per_agent", agent_ids=None):
        super().__init__(config, obs_dims, action_dims, state_dims, state_layout, stage="policy", agent_ids=agent_ids)

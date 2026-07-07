# -*- coding: utf-8 -*-
# 2026-03-28, add GMM model for SRPO, replace the diffusion behavior with GMM model
import copy
import torch
import torch.nn as nn
from algorithms.model import *

import math
import torch.nn.functional as F


### GMM Score model ###

class ConditionalGMM(nn.Module):
    """
    条件对角高斯混合（MDN）。
    cond: (B, cond_dim) -> logits (B, K), mu (B, K, D), log_std (B, K, D)
    log p(a|cond) 用 log-sum-exp，对 a 可微（用于 ∇_a log p）。
    """

    def __init__(self, cond_dim, action_dim, n_components, hidden_dim=256, min_std=1e-4):
        super().__init__()
        self.K = n_components
        self.action_dim = action_dim
        self.min_std = min_std
        out_dim = n_components + n_components * action_dim + n_components * action_dim
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, cond):
        h = self.net(cond)
        logits = h[:, : self.K]
        mu = h[:, self.K : self.K + self.K * self.action_dim].view(
            -1, self.K, self.action_dim
        )
        log_std = h[:, self.K + self.K * self.action_dim :].view(
            -1, self.K, self.action_dim
        )
        return logits, mu, log_std

    @staticmethod
    def mixture_log_prob(a, logits, mu, log_std, min_std=1e-4):
        """
        a: (B, D)
        logits: (B, K), mu / log_std: (B, K, D)
        return: (B,)  each log p(a|cond)
        """
        log_pi = F.log_softmax(logits, dim=-1)
        std = F.softplus(log_std) + min_std
        a_exp = a.unsqueeze(1)
        log_n = -0.5 * (
            ((a_exp - mu) ** 2) / (std ** 2)
            + 2 * torch.log(std)
            + math.log(2 * math.pi)
        )
        log_n = log_n.sum(-1)
        return torch.logsumexp(log_pi + log_n, dim=-1)


def scale_gmm_behavior_vec(behavior_vec: torch.Tensor, weight=1.0, l2norm=False) -> torch.Tensor:
    """
    GMM 的 ∇_a log p 范数通常远大于扩散里 episilon*wt；先缩放（可选 L2 归一化）再进 SRPO loss，
    使 (behavior·a) 与 β*(guidance·a) 可比，β 扫描才有意义。
    - gmm_behavior_weight: 全局标量，默认 0.02
    - gmm_behavior_l2norm: True 时按样本 L2 归一化后再乘 weight（保方向、分量比例）
    - flip_gmm_behavior_score: True 时用 -∇log p（与扩散 score 符号不一致时可试）
    """
    # sign = -1.0 if getattr(args, "flip_gmm_behavior_score", False) else 1.0
    w = float(weight)
    # v = sign * behavior_vec
    if l2norm:
        denom = behavior_vec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        behavior_vec = behavior_vec / denom
    return w * behavior_vec



def gmm_score_at_a(gmm_module: ConditionalGMM, detach_a: torch.Tensor, cond: torch.Tensor):
    """
    策略更新用：在 detach_a 上算 ∇_a log p(a|cond)。
    cond 先过 GMM 且 detach，避免 score 反传到 GMM 参数。
    detach_a 需 requires_grad=True。
    """
    gmm_module.eval()
    with torch.no_grad():
        logits, mu, log_std = gmm_module(cond)
    log_p = ConditionalGMM.mixture_log_prob(
        detach_a, logits, mu, log_std, min_std=gmm_module.min_std
    )
    vec = torch.autograd.grad(
        log_p.sum(), detach_a, retain_graph=True, create_graph=False
    )[0]

    # 处理GMM的归一化问题
    # print(f"vec norm: {vec.norm(dim=-1, keepdim=True).mean()}")
    vec = scale_gmm_behavior_vec(vec, weight=1.0, l2norm=True)
    # print(f"vec norm: {vec.norm(dim=-1, keepdim=True).mean()}")
    return vec.detach()


def gmm_nll_loss(gmm_module: ConditionalGMM, a_data: torch.Tensor, cond: torch.Tensor):
    """预训练用：最大化 log p，即最小化 -mean(log p)。"""
    gmm_module.train()
    logits, mu, log_std = gmm_module(cond)
    log_p = ConditionalGMM.mixture_log_prob(
        a_data, logits, mu, log_std, min_std=gmm_module.min_std
    )
    return -log_p.mean()



# SRPO for IND & JAL：独立 Q(s,a) + 条件 GMM 行为 score（cond 为局部观测或 JAL 拼接观测）
class SRPO_GMM(nn.Module):
    def __init__(self, input_dim, output_dim, marginal_prob_std, args=None):
        super().__init__()
        cond_dim = input_dim - output_dim
        n_comp = getattr(args, "n_gmm_components", 8)
        hidden = getattr(args, "gmm_hidden_dim", 512)
        self.gmm_score_model = ConditionalGMM(
            cond_dim, output_dim, n_comp, hidden_dim=hidden
        ).to(args.device)
        self.gmm_optimizer = torch.optim.AdamW(
            self.gmm_score_model.parameters(), lr=args.learning_rates
        )
        self.gmm_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.gmm_optimizer, T_max=2000000, eta_min=1e-5
        )

        self.SRPO_policy = Dirac_Policy(
            output_dim, input_dim - output_dim, layer=args.policy_layer
        ).to(args.device)
        dilac_lr = getattr(args, "dilac_lr", 3e-4)
        self.SRPO_policy_optimizer = torch.optim.Adam(
            self.SRPO_policy.parameters(), lr=dilac_lr
        )
        self.SRPO_policy_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.SRPO_policy_optimizer,
            T_max=args.n_policy_epochs * 10000,
            eta_min=0.0,
        )

        self.marginal_prob_std = marginal_prob_std  # 与旧构造兼容，策略更新中不用
        self.args = args
        self.output_dim = output_dim
        self.step = 0
        self.q = []
        self.q.append(
            IQL_Critic(adim=output_dim, sdim=input_dim - output_dim, args=args)
        )

    def update_SRPO_policy(self, data):
        s = data["s"]
        a = self.SRPO_policy(s)
        detach_a = a.detach().requires_grad_(True)

        behavior_vec = gmm_score_at_a(self.gmm_score_model, detach_a, s)

        qs = self.q[0].q0_target.both(detach_a, s)
        q = (qs[0].squeeze() + qs[1].squeeze()) / 2.0
        self.SRPO_policy.q = torch.mean(q)

        guidance = torch.autograd.grad(torch.sum(q), detach_a)[0].detach()

        if self.args.regq:
            guidance_norm = torch.mean(guidance ** 2, dim=-1, keepdim=True).sqrt()
            guidance = guidance / guidance_norm

        loss = (behavior_vec * a).sum(-1) - (guidance * a).sum(-1) * self.args.beta
        loss = loss.mean()

        self.SRPO_policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.SRPO_policy_optimizer.step()
        self.SRPO_policy_lr_scheduler.step()

        error_a = torch.mean(data["a"] - a)
        episilon = behavior_vec

        if self.args.env_id == "bandit":
            return loss, episilon, guidance, error_a, a
        else:
            return loss, episilon, guidance, error_a
    



# SRPO for Seq_MASRPO：联合 Q + 条件 GMM 行为 score（OMSD / BRPO-CTDE 第一个 agent，cond 仅 s）
class SRPO_GMM_CTDE(nn.Module):
    def __init__(self, input_dim, output_dim, marginal_prob_std, args=None):
        super().__init__()
        # 与 ScoreNet 一致：condition 维 = input_dim - output_dim（此处即局部观测 s）
        cond_dim = input_dim - output_dim
        n_comp = getattr(args, "n_gmm_components", 8)
        hidden = getattr(args, "gmm_hidden_dim", 512)
        self.gmm_score_model = ConditionalGMM(
            cond_dim, output_dim, n_comp, hidden_dim=hidden
        ).to(args.device)
        self.gmm_optimizer = torch.optim.AdamW(
            self.gmm_score_model.parameters(), lr=args.learning_rates
        )
        self.gmm_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.gmm_optimizer, T_max=2000000, eta_min=1e-5
        )

        self.SRPO_policy = Dirac_Policy(
            output_dim, input_dim - output_dim, layer=args.policy_layer
        ).to(args.device)
        self.SRPO_policy_optimizer = torch.optim.Adam(
            self.SRPO_policy.parameters(), lr=3e-4
        )
        self.SRPO_policy_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.SRPO_policy_optimizer,
            T_max=args.n_policy_epochs * 10000,
            eta_min=0.0,
        )

        self.marginal_prob_std = marginal_prob_std  # 仅保留调用约定，GMM 策略更新不用
        self.args = args
        self.output_dim = output_dim
        self.step = 0

        n_agent_numbers = len(args.alg_types)
        self.q = []
        self.q.append(
            IQL_Critic(
                adim=output_dim * n_agent_numbers,
                sdim=(input_dim - output_dim) * n_agent_numbers,
                args=args,
            )
        )

    def update_SRPO_policy(self, data, agent_id):
        s = data["s"]
        s_joint = data["s_joint"]
        a_joint = data["a_joint"]

        a = self.SRPO_policy(s)
        detach_a = a.detach().requires_grad_(True)
        a_joint[agent_id] = detach_a
        detach_a_joint = torch.cat(a_joint, dim=1)

        behavior_vec = gmm_score_at_a(self.gmm_score_model, detach_a, s)

        qs = self.q[0].q0_target.both(detach_a_joint, s_joint)
        q = (qs[0].squeeze() + qs[1].squeeze()) / 2.0
        self.SRPO_policy.q = torch.mean(q)

        guidance = torch.autograd.grad(torch.sum(q), detach_a)[0].detach()

        if self.args.regq:
            guidance_norm = torch.mean(guidance ** 2, dim=-1, keepdim=True).sqrt()
            guidance = guidance / guidance_norm

        # wt = getattr(self.args, "gmm_score_wt", 1.0)
        loss = (behavior_vec * a).sum(-1) - (guidance * a).sum(
            -1
        ) * self.args.beta
        loss = loss.mean()

        self.SRPO_policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.SRPO_policy_optimizer.step()
        self.SRPO_policy_lr_scheduler.step()

        error_a = torch.mean(data["a"] - a)
        episilon = behavior_vec  # 占位：原 diffusion episilon，便于日志/返回值兼容

        if self.args.env_id == "bandit":
            return loss, episilon, guidance, error_a, a
        else:
            return loss, episilon, guidance, error_a

    def update_SRPO_policy_ordered(self, data, agent_id, update_order_idx):
        s = data["s"]
        s_joint = data["s_joint"]
        a_joint = data["a_joint"]
        a_joint_ordered = data["a_joint_ordered"]

        a = self.SRPO_policy(s)
        detach_a = a.detach().requires_grad_(True)
        a_joint[agent_id] = detach_a
        a_joint_ordered[update_order_idx] = a.detach().clone()

        detach_a_joint = torch.cat(a_joint, dim=1)

        # 与扩散版一致：有序训练时行为条件仍用当前局部 s（与 update_SRPO_policy_ordered 里扩散用 s 相同）
        behavior_vec = gmm_score_at_a(self.gmm_score_model, detach_a, s)

        qs = self.q[0].q0_target.both(detach_a_joint, s_joint)
        q = (qs[0].squeeze() + qs[1].squeeze()) / 2.0
        self.SRPO_policy.q = torch.mean(q)

        guidance = torch.autograd.grad(torch.sum(q), detach_a)[0].detach()

        if self.args.regq:
            guidance_norm = torch.mean(guidance ** 2, dim=-1, keepdim=True).sqrt()
            guidance = guidance / guidance_norm

        # wt = getattr(self.args, "gmm_score_wt", 1.0)
        loss = (behavior_vec * a).sum(-1) - (guidance * a).sum(
            -1
        ) * self.args.beta
        loss = loss.mean()

        self.SRPO_policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.SRPO_policy_optimizer.step()
        self.SRPO_policy_lr_scheduler.step()

        error_a = torch.mean(data["a"] - a)
        episilon = behavior_vec

        if self.args.env_id == "bandit":
            return loss, episilon, guidance, error_a, a
        else:
            return loss, episilon, guidance, error_a



# 第二个agent的policy维度需要调整，policy维度并没有变化
# 顺序分解中后续 agent：p(a_i | s, a_<i)，与扩散版 SSD 相同维数约定
class SRPO_GMM_ssd(nn.Module):
    def __init__(self, input_dim, output_dim, marginal_prob_std, agent_idx, args=None):
        super().__init__()
        cond_dim = input_dim - output_dim
        n_comp = getattr(args, "n_gmm_components", 8)
        hidden = getattr(args, "gmm_hidden_dim", 512)
        self.gmm_score_model = ConditionalGMM(
            cond_dim, output_dim, n_comp, hidden_dim=hidden
        ).to(args.device)
        self.gmm_optimizer = torch.optim.AdamW(
            self.gmm_score_model.parameters(), lr=args.learning_rates
        )
        self.gmm_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.gmm_optimizer, T_max=2000000, eta_min=1e-5
        )

        self.SRPO_policy = Dirac_Policy(
            output_dim, input_dim - (agent_idx + 1) * output_dim, layer=args.policy_layer
        ).to(args.device)
        self.SRPO_policy_optimizer = torch.optim.Adam(
            self.SRPO_policy.parameters(), lr=3e-4
        )
        self.SRPO_policy_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.SRPO_policy_optimizer,
            T_max=args.n_policy_epochs * 10000,
            eta_min=0.0,
        )

        self.marginal_prob_std = marginal_prob_std
        self.args = args
        self.output_dim = output_dim
        self.agent_idx = agent_idx
        self.step = 0

        n_agent_numbers = len(args.alg_types)
        self.q = []
        self.q.append(
            IQL_Critic(
                adim=output_dim * n_agent_numbers,
                sdim=(input_dim - (agent_idx + 1) * output_dim) * n_agent_numbers,
                args=args,
            )
        )

    def update_SRPO_policy(self, data, agent_id):
        s = data["s"]
        s_joint = data["s_joint"]
        a_joint = data["a_joint"]

        a = self.SRPO_policy(s)

        prefix_actions = torch.cat(a_joint[:agent_id], dim=1)
        s_condition = torch.cat((s, prefix_actions), dim=1).to(self.args.device)

        detach_a = a.detach().requires_grad_(True)
        a_joint[agent_id] = detach_a
        detach_a_joint = torch.cat(a_joint, dim=1)

        behavior_vec = gmm_score_at_a(self.gmm_score_model, detach_a, s_condition)

        qs = self.q[0].q0_target.both(detach_a_joint, s_joint)
        q = (qs[0].squeeze() + qs[1].squeeze()) / 2.0
        self.SRPO_policy.q = torch.mean(q)

        guidance = torch.autograd.grad(torch.sum(q), detach_a)[0].detach()

        if self.args.regq:
            guidance_norm = torch.mean(guidance ** 2, dim=-1, keepdim=True).sqrt()
            guidance = guidance / guidance_norm

        loss = (behavior_vec * a).sum(-1) - (guidance * a).sum(-1) * self.args.beta
        loss = loss.mean()

        self.SRPO_policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.SRPO_policy_optimizer.step()
        self.SRPO_policy_lr_scheduler.step()

        error_a = torch.mean(data["a"] - a)
        episilon = behavior_vec

        if self.args.env_id == "bandit":
            return loss, episilon, guidance, error_a, a
        else:
            return loss, episilon, guidance, error_a

    def update_SRPO_policy_ordered(self, data, agent_id, update_order_idx):
        s = data["s"]
        s_joint = data["s_joint"]
        a_joint = data["a_joint"]
        a_joint_ordered = data["a_joint_ordered"]

        a = self.SRPO_policy(s)

        prefix_actions = torch.cat(a_joint_ordered[:update_order_idx], dim=1)
        s_condition = torch.cat((s, prefix_actions), dim=1).to(self.args.device)

        detach_a = a.detach().requires_grad_(True)
        a_joint[agent_id] = detach_a
        a_joint_ordered[update_order_idx] = a.detach().clone()

        detach_a_joint = torch.cat(a_joint, dim=1)

        behavior_vec = gmm_score_at_a(self.gmm_score_model, detach_a, s_condition)

        qs = self.q[0].q0_target.both(detach_a_joint, s_joint)
        q = (qs[0].squeeze() + qs[1].squeeze()) / 2.0
        self.SRPO_policy.q = torch.mean(q)

        guidance = torch.autograd.grad(torch.sum(q), detach_a)[0].detach()

        if self.args.regq:
            guidance_norm = torch.mean(guidance ** 2, dim=-1, keepdim=True).sqrt()
            guidance = guidance / guidance_norm

        loss = (behavior_vec * a).sum(-1) - (guidance * a).sum(-1) * self.args.beta
        loss = loss.mean()

        self.SRPO_policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.SRPO_policy_optimizer.step()
        self.SRPO_policy_lr_scheduler.step()

        error_a = torch.mean(data["a"] - a)
        episilon = behavior_vec

        if self.args.env_id == "bandit":
            return loss, episilon, guidance, error_a, a
        else:
            return loss, episilon, guidance, error_a
    



class MASRPO_Behavior_GMM(nn.Module):
    """预训练条件 GMM 行为模型：最大化 log p(a|obs)，接口对齐原 MASRPO_Behavior。"""

    def __init__(self, input_dim, output_dim, marginal_prob_std, args=None):
        super().__init__()
        cond_dim = input_dim - output_dim
        n_comp = getattr(args, "n_gmm_components", 8)
        hidden = getattr(args, "gmm_hidden_dim", 512)

        self.gmm_score_model = ConditionalGMM(
            cond_dim, output_dim, n_comp, hidden_dim=hidden
        ).to(args.device)
        self.gmm_optimizer = torch.optim.AdamW(
            self.gmm_score_model.parameters(), lr=args.learning_rates
        )
        if getattr(args, "lr_anneal", False):
            self.gmm_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.gmm_optimizer, T_max=2000000, eta_min=1e-5
            )
        else:
            self.gmm_lr_scheduler = None

        self.marginal_prob_std = marginal_prob_std  # 仅占位，与旧构造签名兼容
        self.args = args
        self.output_dim = output_dim
        self.step = 0
        self.device = args.device

    def update_behavior(self, data):
        self.step += 1
        all_a = data["action"].to(self.device)
        all_s = data["obs"].to(self.device)

        self.gmm_score_model.train()
        loss = gmm_nll_loss(self.gmm_score_model, all_a, all_s)
        self.loss = loss

        self.gmm_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.gmm_optimizer.step()
        if self.gmm_lr_scheduler is not None:
            self.gmm_lr_scheduler.step()

        return loss
        



def update_target(new, target, tau):
    # Update the frozen target models
    for param, target_param in zip(new.parameters(), target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)






class IQL_Critic(nn.Module):
    def __init__(self, adim, sdim, args) -> None:
        super().__init__()
        self.q0 = TwinQ(adim, sdim, layers=args.q_layer).to(args.device)
        self.q0_target = copy.deepcopy(self.q0).to(args.device)

        self.vf = ValueFunction(sdim).to(args.device)
        self.q_optimizer = torch.optim.Adam(self.q0.parameters(), lr=args.iql_critic_lr)   # 3e-4 有些任务爆炸，试试更小的lr
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=args.iql_critic_lr)

        self.discount = 0.99
        self.args = args
        self.tau = args.tau  # 改成args.tau
        # self.tau = 0.9 if "maze" in args.env_id else 0.8     # 默认0.7
        
        
        
    def update_q0(self, data):
        s = data["obs"]
        a = data["action"]
        r = data["rewards"]
        s_ = data["next_obs"]
        d = data["done"]
        with torch.no_grad():
            target_q = self.q0_target(a, s).detach()
            next_v = self.vf(s_).detach()

        # Update value function
        v = self.vf(s)
        adv = target_q - v
        self.adv = adv.mean()

        v_loss = asymmetric_l2_loss(adv, self.tau)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # 用于logger
        self.v = v.mean()

        # Update Q function
        # 这里做了修改，1-done是256维度，乘以后面的256，1维度会错误的变成256，256
        targets = r.unsqueeze(1) + (1. - d.float()).unsqueeze(1) * self.discount * next_v.detach()

        self.r = r.mean()
        self.d = d.sum()
        self.target = targets.mean()

        qs = self.q0.both(a, s)
        q_loss = sum(torch.nn.functional.mse_loss(q, targets) for q in qs) / len(qs)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        self.v_loss = v_loss
        self.q_loss = q_loss

        self.q = target_q.mean()
        self.q_std = target_q.std()
        self.q_max = target_q.max()
        self.q_min = target_q.min()

        self.v_next = next_v.mean()
        self.v_next_std = next_v.std()

        # Update target
        update_target(self.q0, self.q0_target, 0.005)


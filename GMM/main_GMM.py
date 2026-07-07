# Diffusion-QL Copyright 2022 Twitter, Inc and Zhendong Wang.
# Framework copyright. CFCQL and OMAR

# Algorithm: JAL_DQ, ind_DQ, ind_SRPO, JAL_SRPO, CTDE_SRPO
# ToDO Algo: QMIX_SRPO
import os, sys, tempfile
import json
import argparse
import numpy as np
import datetime
import random
from tqdm import tqdm

# import gym
from gym.spaces import Box, Discrete
import torch
from torch.autograd import Variable

# from tensorboard_logger import log_value, configure
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import DummyVecEnv

# # env check
# try:
#     from multiagent_mujoco.mujoco_multi import MujocoMulti
# except:
#     print ('MujocoMulti not installed')

# OMIGA
try:
    from envs.ma_mujoco.multiagent_mujoco.mujoco_multi import MujocoMulti
except:
    print ('OMIGA MujocoMulti not installed')

from bandit import ContinuousBanditEnv

# from algorithms.madiffQL import MADiff, MADiff_JAL  #MADiff_seq, MADiff_CTCE
# from algorithms.MASRPO import IND_SRPO, JAL_SRPO, CTDE_SRPO, OMSD     # VD_SRPO, CTDE_SRPO
from GMM.MASRPO_GMM import IND_SRPO_GMM, JAL_SRPO_GMM, CTDE_SRPO_GMM, OMSD_GMM

import wandb

# make parallel MA-Env
def make_parallel_env(env_id, seed, discrete_action):
    def get_env_fn(rank):
        env = make_env(env_id, discrete_action=discrete_action)
        env.seed(seed + rank * 1000)
        np.random.seed(seed + rank * 1000)
        return env
    return DummyVecEnv([get_env_fn(0)])

# evaluate policy in eval module with envs(seed+100) on cpu
"""这里要检查一下mpe的逻辑"""
"""新增收集eval policy数据"""
def eval_policy(agent, env_name, seed, eval_episodes, discrete_action, device='cpu', env_args=None):

    # 创建数据收集字典
    collected_data = {
        'obs': [],      # 观测
        'actions': [],  # 动作
        'rewards': [],  # 奖励
        'episode_lens': [], # 每个episode的长度
        'episode_returns': [] # 每个episode的累积奖励
    }

    if env_name in ['HalfCheetah-v2', 'Hopper-v2', 'Ant-v2']:
        env = MujocoMulti(env_args=env_args)
        env.seed(seed + 100)
        all_episodes_rewards = []
        for ep_i in range(eval_episodes):
            agent.prep_rollouts(device=device)  # 转成 eval 模式
            env.reset()
            done = False
            episode_reward = 0.

            # 新增：收集数据
            episode_obs = []
            episode_actions = []
            episode_rewards = []


            while not done:
                obs = env.get_obs()
                # torch_obs = [Variable(torch.Tensor(obs[i]).unsqueeze(0), requires_grad=False) for i in range(agent.nagents)] 
                torch_obs = [torch.Tensor(obs[i]).unsqueeze(0).to(device)  for i in range(agent.nagents)] 
                torch_agent_actions = agent.step(torch_obs, explore=False)
                # if torch.is_tensor(torch_agent_actions):
                if all(isinstance(item, torch.Tensor) for item in torch_agent_actions):
                    agent_actions = [ac.data.numpy() for ac in torch_agent_actions]  # 从 tensor([[a], [a], [a]]) 变为 list[np[], np[], np[]] 
                elif all(isinstance(item, np.ndarray) for item in torch_agent_actions):
                    agent_actions = torch_agent_actions
                actions = [ac.squeeze(0) for ac in agent_actions]  # 变为 list[np, np, np]


                # 收集数据
                episode_obs.append([o.cpu().numpy() for o in torch_obs])
                episode_actions.append(actions)


                # reward, done, info = env.step(actions)  
                # omiga wrapper 
                # 目前是 [array array array]
                new_obs, new_state, rewards, dones, info, avaliable_actions = env.step(actions)
                done = dones[0]
                episode_reward += rewards[0][0].item()

                episode_rewards.append(rewards[0][0].item())

                # episode_reward += reward
            all_episodes_rewards.append(episode_reward)  

            # 保存这个episode的数据
            collected_data['obs'].append(episode_obs)
            collected_data['actions'].append(episode_actions)
            collected_data['rewards'].append(episode_rewards)
            collected_data['episode_lens'].append(len(episode_obs))
            collected_data['episode_returns'].append(episode_reward)  

        mean_episode_reward = np.mean(np.array(all_episodes_rewards))
        return mean_episode_reward, collected_data
    elif env_name == 'bandit':
        env = ContinuousBanditEnv()
        
        agent.prep_rollouts(device=device)  # 转成 eval 模式
        obs = env.reset()
        done = False
        torch_obs = [torch.Tensor(obs).unsqueeze(0).to(device)  for i in range(agent.nagents)]
        torch_agent_actions = agent.step(torch_obs, explore=False)
        actions = [ac.squeeze(0) for ac in torch_agent_actions]  # 变为 list[np, np, np]
        _, reward, done, info = env.step(actions)  # 可以简单点，直接把action相乘

        mean_episode_reward = np.array(reward)
        return mean_episode_reward
    elif env_name == 'simple_spread':
        avg_predator_return = 0.
        env = make_parallel_env(env_name, seed + 100, discrete_action)
        for ep_i in range(0, eval_episodes):
            obs = env.reset()    # 修改过，返回的是list不再是array
            obs = np.array(obs)  # 处理返回的list形态reset数据
            agent.prep_rollouts(device=device)

            # 新增：收集数据
            episode_reward = 0
            episode_obs = []
            episode_actions = []
            episode_rewards = []


            for et_i in range(config.episode_length):
                obs_len = agent.nagents
                if env_name in ['simple_tag', 'simple_world']:  # if predator-prey
                    obs_len += agent.num_preys
                torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])), requires_grad=False) for i in range(obs_len)]
                # 把 obs_dim * n_agents 的tensor变成 [n * [obs_dim*1]] 的变量
                torch_agent_actions = agent.step(torch_obs, explore=False)
                if torch.is_tensor(torch_agent_actions):
                    # 从 tensor([[a], [a], [a]]) 变为 list[np[], np[], np[]]
                    agent_actions = [ac.data.numpy() for ac in torch_agent_actions]  
                elif isinstance(torch_agent_actions, list):
                    # 从 list[tensor, tensor, tensor] 变为 list[np[], np[], np[]]
                    agent_actions = []
                    for ac in torch_agent_actions:
                        if torch.is_tensor(ac):
                            agent_actions.append(ac.squeeze().data.numpy())
                            # 最内层action必须是 [2]，不可以是[1,2]，否则在env.step中会报维度错误
                            # agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
                        elif isinstance(ac, np.ndarray):
                            agent_actions.append(ac)
                        else:
                            raise TypeError(f"Unsupported type in list: {type(ac)}")
                else:
                    agent_actions = torch_agent_actions
                

                actions = [agent_actions]

                # 收集数据
                episode_obs.append([o.cpu().numpy() for o in torch_obs])
                episode_actions.append(agent_actions)


                next_obs, rewards, dones, infos = env.step(actions)
                # 这里修改了MPE的原函数 utils/env_wrappers，返回的next obs是一个list而不是array
                next_obs = np.array(next_obs)
                
                if env_name in ['simple_tag', 'simple_world']:
                    avg_predator_return += rewards[0][0]
                    episode_rewards.append(rewards[0][0])
                else:
                    avg_agent_reward = np.mean(rewards[0])
                    avg_predator_return += avg_agent_reward
                    episode_rewards.append(avg_agent_reward)

                obs = next_obs
            
            # 保存这个episode的数据
            collected_data['obs'].append(episode_obs)
            collected_data['actions'].append(episode_actions)
            collected_data['rewards'].append(episode_rewards)
            collected_data['episode_lens'].append(len(episode_obs))
            collected_data['episode_returns'].append(episode_reward)

        avg_predator_return /= eval_episodes
        return avg_predator_return, collected_data
    elif env_name == 'simple_tag' or env_name == 'simple_world':
        avg_predator_return = 0.
        env = make_parallel_env(env_name, seed + 100, discrete_action)
        for ep_i in range(0, eval_episodes):
            obs = env.reset()    # 修改过，返回的是list不再是array
            agent.prep_rollouts(device=device)

            # 新增：收集数据
            episode_reward = 0
            episode_obs = []
            episode_actions = []
            episode_rewards = []


            for et_i in range(config.episode_length):
                obs_len = agent.nagents
                if env_name in ['simple_tag', 'simple_world']:  # if predator-prey
                    obs_len += agent.num_preys
                
                # 新增代码适配simple tag/world环境
                torch_obs = [Variable(torch.Tensor(obs_i).unsqueeze(0), requires_grad=False) for obs_i in obs[0]]
                # 把 obs_dim * n_agents 的tensor变成 [n * [obs_dim*1]] 的变量
                torch_agent_actions = agent.step(torch_obs, explore=False)
                if torch.is_tensor(torch_agent_actions):
                    # 从 tensor([[a], [a], [a]]) 变为 list[np[], np[], np[]]
                    agent_actions = [ac.data.numpy() for ac in torch_agent_actions]  
                elif isinstance(torch_agent_actions, list):
                    # 从 list[tensor, tensor, tensor] 变为 list[np[], np[], np[]]
                    agent_actions = []
                    for ac in torch_agent_actions:
                        if torch.is_tensor(ac):
                            agent_actions.append(ac.squeeze().data.numpy())
                            # 最内层action必须是 [2]，不可以是[1,2]，否则在env.step中会报维度错误
                            # agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
                        elif isinstance(ac, np.ndarray):
                            agent_actions.append(ac)
                        else:
                            raise TypeError(f"Unsupported type in list: {type(ac)}")
                else:
                    agent_actions = torch_agent_actions
                

                actions = [agent_actions]

                # 只收集predator的数据
                episode_obs.append([o.cpu().numpy() for o in torch_obs[:agent.num_predators]])  # 只取predator的obs
                episode_actions.append(agent_actions[:agent.num_predators])  # 只取predator的actions
                


                next_obs, rewards, dones, infos = env.step(actions)
                
                if env_name in ['simple_tag', 'simple_world']:
                    avg_predator_return += rewards[0][0]
                    episode_reward += rewards[0][0]  # predator的reward
                    episode_rewards.append(rewards[0][0])
                else:
                    avg_agent_reward = np.mean(rewards[0])
                    avg_predator_return += avg_agent_reward
                    episode_reward += avg_agent_reward  # predator的reward
                    episode_rewards.append(avg_agent_reward)

                obs = next_obs

            # 保存这个episode的数据
            collected_data['obs'].append(episode_obs)
            collected_data['actions'].append(episode_actions)
            collected_data['rewards'].append(episode_rewards)
            collected_data['episode_lens'].append(len(episode_obs))
            collected_data['episode_returns'].append(episode_reward)

        avg_predator_return /= eval_episodes
        return avg_predator_return, collected_data


# log params to tensorboard
def log_and_print(key, value, t, writer, multi=False):
    if multi:
        print("t:", t, end=" | ")
        for i in range(len(key)):
            end = " | " if i < len(key) - 1 else "\n"
            print("{}: {:.3f}".format(key[i], value[i][0]), end=end)
            # log_value(key[i], value[i], t)    
            writer.add_scalar(key[i], value[i], t)
    else:
        print("t:{}, {}: {:.3f}".format(t, key, value))
        writer.add_scalar(key, value, t)
        # log_value(key, value, t)


def load_SRPO_critic(srpo_model, load_path, srpo_type, epoch_num):
    if load_path is not None:
        print("loading critic...")
        if srpo_type == 'IND':
            for agent_index, srpo_i in enumerate(srpo_model.agents):
                # SRPO_premodels/exp_seed/IND/best_critic_i
                if epoch_num == 'best':
                    load_path_i = os.path.join(load_path, 'IND', f'best_critic_{agent_index}.pth')
                else:
                    load_path_i = os.path.join(load_path, 'IND', f'critic_{agent_index}_epoch{epoch_num}.pth')
                ckpt = torch.load(load_path_i, map_location=srpo_model.device)
                srpo_i.q[0].load_state_dict(ckpt)
        elif srpo_type == 'JAL':
            # SRPO_models/exp_seed/JAL/best_critic.pth
            load_path = os.path.join(load_path, 'JAL', f'best_critic.pth')
            ckpt = torch.load(load_path, map_location=srpo_model.device)
            srpo_model.agents[0].q[0].load_state_dict(ckpt)
        elif srpo_type == 'CTDE' or srpo_type == 'SEQ':
            for srpo_i in srpo_model.agents:
                # SRPO_premodels/exp_seed/JAL/best_critic  每个agent都有一个central Q，共享
                if epoch_num == 'best':
                    load_path_i = os.path.join(load_path, 'critic', f'best_critic.pth')
                else:
                    # load_path_i = os.path.join(load_path, 'critic', f'critic_epoch{epoch_num}.pth')   # used for OMIGA
                    load_path_i = os.path.join(load_path, 'JAL', f'critic_epoch{epoch_num}.pth')  # used for MPE
                ckpt = torch.load(load_path_i, map_location=srpo_model.device)
                srpo_i.q[0].load_state_dict(ckpt)

    else:
        assert False

def load_SRPO_GMM(srpo_model, load_path, srpo_type, epoch_num):
    # IND & CTDE load ind diffusion score, JAL load joint diffusion score
    if load_path is not None:
        print("loading actor...")
        if srpo_type == 'IND' or srpo_type == 'CTDE':
            for agent_index, srpo_i in enumerate(srpo_model.agents):
                # SRPO_premodels/exp_seed/IND/best_diffusion_i.pth
                if epoch_num == 'best':
                    load_path_i = os.path.join(load_path, 'IND', f'best_gmm_{agent_index}.pth')
                else:
                    load_path_i = os.path.join(load_path, 'IND', f'gmm_{agent_index}_epoch{epoch_num}.pth')
                ckpt = torch.load(load_path_i, map_location=srpo_model.device)
                # for k,v in ckpt.items():
                #     print("{} ckpt: {}, srpo: {}".format(k, ckpt[k].shape, srpo_i.state_dict()[k].shape))
                srpo_i.load_state_dict({k:v for k,v in ckpt.items() if "gmm_score_model" in k}, strict=False)
        elif srpo_type == 'SEQ':
            for agent_index, srpo_i in enumerate(srpo_model.agents):
                # SRPO_premodels/exp_seed/diffusion/best_diffusion_i.pth
                if epoch_num == 'best':
                    load_path_i = os.path.join(load_path, f'best_gmm_agent{agent_index}.pth')
                else:
                    load_path_i = os.path.join(load_path, f'gmm_agent{agent_index}_epoch{epoch_num}.pth') 
                    # load_path_i = os.path.join(load_path, f'diffusion_agent{agent_index}_epoch{epoch_num}.pth')    # 用于 ordered
                ckpt = torch.load(load_path_i, map_location=srpo_model.device)
                srpo_i.load_state_dict({k:v for k,v in ckpt.items() if "gmm_score_model" in k}, strict=False)
        elif srpo_type == 'JAL':
            # SRPO_premodels/exp_seed/ + JAL/diffusion.pth
            srpo = srpo_model.agents[0]
            load_path = os.path.join(load_path, 'JAL', f'best_gmm.pth')
            ckpt = torch.load(load_path, map_location=srpo_model.device)

            # 这里的load可能有逻辑错误，期望的是net.0.weight不带外层前缀，可能需要对整个agent采取 spro_i.load_state_dict(ckpt, strict=False)
            # 暂时用不到所以先不管
            srpo.gmm_score_model.load_state_dict({k:v for k,v in ckpt.items() if "gmm_score_model" in k}, strict=False)
    else:
        assert False


def offline_train(config):
    unique_token = f"{config.marltype}_{config.data_type}_beta{config.beta}_critic{config.critic_epoch}_{config.n_gmm_components}GMM_seed{config.dataset_num}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}"

    # unique_token = config.marltype+"_"+unique_token
    # JAL/IND/VD/SEQ _ time _ seed 

    if not config.no_log:
        # if config.conditional_order is not None:
        #     outdir = os.path.join(config.dir, "omiga", config.env_id, config.conditional_order, unique_token)
        # else:
        outdir = os.path.join(config.dir, "gmm", config.env_id, unique_token)
        # outdir = os.path.join(config.dir, "omiga", config.env_id, config.conditional_order, unique_token)
        os.makedirs(outdir)
        print('\033[1;32mOutput files are saved in {} \033[1;0m'.format(outdir))
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    run = None   # 用来控制wandb的，已经移除

    
    if config.env_id in ['simple_spread', 'simple_tag', 'simple_world']:
        env = make_parallel_env(config.env_id, config.seed, config.discrete_action)
        env_args, env_info = None, None
    elif config.env_id == "bandit":
        env = ContinuousBanditEnv()
        env_args, env_info = None, None
    else:
        # env_args = {"scenario": config.env_id, "episode_limit": 1000, "agent_conf": '6x1', "agent_obsk": 1,}
        print(config.env_args)
        env = MujocoMulti(env_args=config.env_args)
        env.seed(config.seed)
        env_info = env.get_env_info()


    """参考pretrain，Jan.30 考虑新增"""
    # 创建score model和load buffer时用得到
    if config.env_id in ['HalfCheetah-v2', 'Hopper-v2', 'Ant-v2', 'bandit']:
        each_state_shape = [env_info['state_shape'] for _ in env.observation_space]
        # MaMujuco use obs as input
        each_obs_shape = [env_info['obs_shape'] for _ in env.observation_space]
        each_action_shape = [acsp.shape[0] for acsp in env.action_space]
        agent_num = len(each_action_shape) 
        state_dim = each_obs_shape[0]
        action_dim = each_action_shape[0]
    elif config.env_id in ['simple_spread']:
        each_state_shape = [obsp.shape[0] for obsp in env.observation_space]
        each_action_shape = [acsp.shape[0] for acsp in env.action_space]
        each_action_max = [acsp.high[0] for acsp in env.action_space]
        agent_num = len(each_action_shape) 
        state_dim = each_state_shape[0]
        action_dim = each_action_shape[0]
        action_max = each_action_max[0]
    elif config.env_id in ['simple_tag', 'simple_world']:
        adversary_indices = [i for i, agent_type in enumerate(env.agent_types) if agent_type == 'adversary']   # 这里agents区分prey和predators
        # 去除 agent 预训练的数据，不需要score model
        each_state_shape = [env.observation_space[i].shape[0] for i in adversary_indices]
        each_action_shape = [env.action_space[i].shape[0] for i in adversary_indices]
        each_action_max = [env.action_space[i].high[0] for i in adversary_indices]
        agent_num = len(adversary_indices) 
        state_dim = each_state_shape[0]
        action_dim = each_action_shape[0]
        action_max = each_action_max[0]


  
    kwargs={'logging_interval': config.logging_interval,
            'no_log': config.no_log,
            'train_num_steps': config.num_steps}



    # # 支持指定的扰动顺序
    # conditional_order = getattr(config, "conditional_order", None)
    # if conditional_order is None:
    #     conditional_order = list(range(config.agent_num))

    # if isinstance(conditional_order, str):
    #     conditional_order = [int(x) for x in conditional_order.split("-") if x != ""]

    # if len(conditional_order) != config.agent_num:
    #     raise ValueError(f"conditional_order length {len(conditional_order)} is not equal to agent_num {config.agent_num}")
    # if agent_num not in conditional_order:
    #     raise ValueError(f"agent {agent_num} is not in conditional_order {conditional_order}")


    # select algorithms from [JAL, ind, seq, VD] + [DiffusionQL, SRPO]
    # if config.difftype == "DQL":
    #     if config.marltype == "JAL":
    #         algo_name = "JAL_Diffusion-QL"
    #         ma_agent = MADiff_JAL.init_from_env(env, env_id = config.env_id, env_info=env_info,
    #                                             agent_alg="diffusion", adversary_alg="ddpg",
    #                                             gamma=config.gamma, tau=config.tau, lr=config.lr,
    #                                             hidden_dim = config.hidden_dim, denoise_steps = config.T,
    #                                             batch_size=config.batch_size, device = config.device, **kwargs)
    #         print("JAL-Diff-QL agents have been created")
    #     elif config.marltype == "IND":
    #         algo_name = "IND_Diffusion-QL"
    #         ma_agent = MADiff.init_from_env(env, env_id = config.env_id, env_info=env_info,
    #                                         agent_alg="diffusion", adversary_alg="ddpg",
    #                                         gamma=config.gamma, tau=config.tau, lr=config.lr,
    #                                         hidden_dim = config.hidden_dim, denoise_steps = config.T,
    #                                         batch_size=config.batch_size, device = config.device, **kwargs)
    #         print("Parallel update independent Diff-QL agents have been created")
    #     else:
    #         algo_name = "SEQ/VD_Diffusion-QL"
    #         # ma_agent = MADiff_seq.init_from_env()   # to be finished in algo/madiffQL
    #         print("Sequential update and VDN Diffusion-QL agents haven't been established")



    if config.difftype == "SRPO":
        # JAL, IND, CTDE, SEQ, QMIX
        if config.marltype == "JAL":
            algo_name = "JAL_SRPO"
            ma_agent = JAL_SRPO_GMM.init_from_env(env, env_id = config.env_id, env_info=env_info,
                                                agent_alg="diffusion", adversary_alg="ddpg",
                                                gamma=config.gamma, tau=config.tau, lr=config.lr,
                                                hidden_dim = config.hidden_dim, denoise_steps = config.T,
                                                batch_size=config.batch_size, device = config.device, config=config, **kwargs)
            print("JAL-SRPO agents have been created")
        elif config.marltype == "IND":
            algo_name = "IND_SRPO"
            ma_agent = IND_SRPO_GMM.init_from_env(env, env_id = config.env_id, env_info=env_info,
                                            agent_alg="diffusion", adversary_alg="ddpg",
                                            gamma=config.gamma, tau=config.tau, lr=config.lr,
                                            hidden_dim = config.hidden_dim, denoise_steps = config.T,
                                            batch_size=config.batch_size, device = config.device, config=config, **kwargs)
            print("Parallel update independent SRPO agents have been created")
            print(ma_agent.init_dict)
        elif config.marltype == 'CTDE':
            algo_name = "CTDE_SRPO"
            ma_agent = CTDE_SRPO_GMM.init_from_env(env, env_id = config.env_id, env_info=env_info,
                                            agent_alg="diffusion", adversary_alg="ddpg",
                                            gamma=config.gamma, tau=config.tau, lr=config.lr,
                                            hidden_dim = config.hidden_dim, denoise_steps = config.T,
                                            batch_size=config.batch_size, device = config.device, config=config, **kwargs)
            print("Naive CTDE SRPO agents have been established")
        elif config.marltype == 'SEQ':
            algo_name = "SEQ_SRPO"
            ma_agent = OMSD_GMM.init_from_env(env, env_id = config.env_id, env_info=env_info,
                                            agent_alg="diffusion", adversary_alg="ddpg",
                                            gamma=config.gamma, tau=config.tau, lr=config.lr,
                                            hidden_dim = config.hidden_dim, denoise_steps = config.T,
                                            batch_size=config.batch_size, device = config.device, config=config, **kwargs)
            print("Sequential update SRPO agents have been established")
    else:
        print("Neither SRPO nor Diffusion-QL have been selected. Choose valid diffusion model")
             
    """一些胡言乱语，暂时不知道有什么用"""
    # score 提取直接load joint diffusion，先生成联合动作然后denoise得到联合score，然后分别取两个分量给每个agent用作epsilon
    # sequential score 是独立的diffusion, 第一个人3维度action，6维度条件；第二个人3维度action，6+3维度condition，
    # 提取score的时候第一个人先denoise得到score，然后基于这个第一个人的采样动作结合state给到第二个人去denoise得到score


    # load pretrained critics and diffusion to SRPO
    if config.critic_load_path is not None:
        load_SRPO_critic(srpo_model=ma_agent, load_path=config.critic_load_path, srpo_type=config.marltype, epoch_num=config.critic_epoch)
        print('Critic models are loaded to SRPO')
    else:
        print('Critic models are not loaded to SRPO')

    if config.gmm_load_path is not None:
        load_SRPO_GMM(srpo_model=ma_agent, load_path=config.gmm_load_path, srpo_type=config.marltype, epoch_num=config.gmm_epoch)
        print('Diffusion models are loaded to SRPO')
    else:
        print('Diffusion models are not loaded to SRPO')

    # load pretrained preys model to DDPG
    if config.env_id in ['simple_tag', 'simple_world']:
        # pretrained_model_dir = './datasets/{}/pretrained_adv_model.pt'.format(config.env_id)
        pretrained_model_dir = str(_root / "datasets" / config.env_id / "pretrained_adv_model.pt")
        ma_agent.load_pretrained_preys(pretrained_model_dir)
    else:
        print('Prey DDPG are not loaded')

    # load full offline datasets to buffer
    if config.env_id in ['simple_spread', 'simple_tag', 'simple_world', 'bandit']:
        replay_buffer = ReplayBuffer(
            config.buffer_length, ma_agent.nagents,
            [obsp.shape[0] for obsp in env.observation_space],
            [acsp.shape[0] if isinstance(acsp, Box) else acsp.n for acsp in env.action_space], device = config.device
        )
    else:
        replay_buffer = ReplayBuffer(
            config.buffer_length, ma_agent.nagents,
            [env_info['obs_shape'] for _ in env.observation_space],
            [acsp.shape[0] for acsp in env.action_space],
            is_mamujoco=True,
            state_dims=[env_info['state_shape'] for _ in env.observation_space], device = config.device, store_on_gpu=True
        )

    
    if config.env_id in ['simple_spread', 'simple_tag', 'simple_world', 'bandit']:
        replay_buffer.load_batch_data(config.dataset_dir, rew_scale = config.rew_scale)
    elif config.env_id in ['HalfCheetah-v2', 'Hopper-v2', 'Ant-v2']:
        replay_buffer.load_batch_data_omiga(config.dataset_dir, rew_scale = config.rew_scale)

    if np.isinf(replay_buffer.ave_reward):   # 如果变量是 inf, 代表这条轨迹没有 done=True，要进行 scale; 应该是主要用于MPE环境
        replay_buffer.ave_reward = replay_buffer.sum_reward / (replay_buffer.filled_i/config.episode_length)
    print('Average_reward:', replay_buffer.ave_reward)

    # tensorboard log dir and save configurations
    if not config.no_log:
        # configure(outdir)
        writer = SummaryWriter(outdir)
        config_log_dict = {"env": config.env_id,
                           "dataset": "{}".format(config.data_type),
                           "dataset_ave_reward": float(replay_buffer.ave_reward),
                           "algo": algo_name,
                           "seed": config.seed,
                           "beta": float(config.beta),
                           "sequential_update_order": config.conditional_order,
                           "Diffusion Epoch":config.gmm_epoch,
                           "Critic Epoch":config.critic_epoch,
                           "Denoise_steps": config.T,
                           "batch size": config.batch_size,
                           "discount factor": float(config.gamma),
                           "soft update": float(config.tau),
                           "IDQN hidden MLP": config.resnet_hidden_dim,
                           "IDQN state-action embed": config.resnet_hidden_dim,
                           "IDQN actor block": config.actor_blocks,
                           }
        
        print(config_log_dict)

        writer.add_text('configurations', str(config_log_dict))

        param_dict = os.path.join(outdir, 'config.json')
        json_str = json.dumps(config_log_dict, separators=(',', ':'))
        json_str = json_str.replace(',', ',\n')

        with open(param_dict, 'w') as f:
            f.write(json_str)

        # with open(param_dict, 'w') as f:
        #     json.dump(config_log_dict, f, indent=None, separators=(',', ': '))
    
    


    # training process
    ma_agent.prep_training(device=config.device)

    progress_bar = tqdm(range(config.num_steps+1), desc = 'Training Process', leave=True)

    for t in range(config.num_steps + 1):
        # set as eval() when eval
        if t % config.eval_interval == 0 or t == config.num_steps:
            # eval_policy will set rollouts at start
            print('Start to {} times eval | Timestep:{}'.format(t % config.eval_interval, t))
            if config.env_id in ['simple_spread', 'simple_tag', 'simple_world']:
                eval_return, eval_data = eval_policy(ma_agent, config.env_id, config.seed, config.eval_episodes, config.discrete_action)
            elif config.env_id in ['HalfCheetah-v2', 'Hopper-v2', 'Ant-v2']:
                eval_return, eval_data = eval_policy(ma_agent, config.env_id, config.seed, config.eval_episodes, config.discrete_action, device='cpu', env_args=config.env_args)
            if not config.no_log:
                log_and_print('eval_return', eval_return, t, writer)
                log_and_print('normed_eval_return', eval_return/replay_buffer.ave_reward, t, writer)

                # 保存评估数据
                if t % config.save_buffer_interval == 0 and config.save_eval_buffer:
                    data_save_path = os.path.join(outdir, f'eval_data_step_{t}.npz')
                    np.savez(data_save_path, 
                            obs=np.array(eval_data['obs']),
                            actions=np.array(eval_data['actions']),
                            rewards=np.array(eval_data['rewards']),
                            episode_lens=np.array(eval_data['episode_lens']),
                            episode_returns=np.array(eval_data['episode_returns']))
                
            # when eval finished, switch to train()
            ma_agent.prep_training(device=config.device)
                
        # load joint datasets for JAL and indpendent trajectory for ind/seq training
        if config.marltype == "JAL":
            sample = replay_buffer.sample(config.batch_size, to_gpu=config.use_gpu)
            ma_agent.update(sample, t, writer, run)  
            # 这里一个可能的问题是，JAL需不需要区分pray的数据
            # Jan 30回答：需要区分，simple tag/world的数据给了3号agent作为pray的数据，要丢掉
        elif config.marltype == "IND":
            nagents = ma_agent.nagents if config.env_id in ['simple_spread', 'HalfCheetah-v2', 'Hopper-v2', 'Ant-v2', 'bandit'] else ma_agent.num_predators
            samples = replay_buffer.sample(config.batch_size, to_gpu=config.use_gpu)
            # 只拿agent i自己的buffer，并只更新a i策略
            for a_i in range(nagents):
                sample_i = samples[a_i]
                ma_agent.update(sample_i, a_i, t, writer, run)   # 只有base srpo有 a_i 指定
        elif config.marltype == "CTDE" or config.marltype == "SEQ":
            samples = replay_buffer.sample(config.batch_size, to_gpu=config.use_gpu)
            # 只拿agent i自己的buffer，并只更新a i策略; 但是计算Q值用的是total state，以及other policy actions
            # 一起输入给进去再分开，更新体现在ma agent内部

            # 原始的默认顺序更新
            # ma_agent.update(samples, t, writer, run)
            # 支持指定的扰动顺序
            if config.conditional_order is not None:
                ma_agent.update_ordered(samples, t, writer, run)
            else:
                ma_agent.update(samples, t, writer, run)
        else:  # QMIX_SRPO
            pass 
            
        progress_bar.update(1)

    if not config.no_log:
        writer.close()

    try:
        env.close()
    except Exception as e:
        print(f"An error occurred while closing the environment: {e}")
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    """   Changable params by users   """
    # Dataset selection  e.g. "simple spread_medium_0"
    parser.add_argument("--env_id", default='simple_tag', type=str, help="Name of environment")   # HalfCheetah-v2  bandit
    parser.add_argument("--data_type", default='expert', type=str)
    parser.add_argument("--dataset_num", default=0, type=int, help="Dataset seed number from 0-4")
    
    # Algo choice: Diffusion QL or SRPO
    parser.add_argument("--difftype", default='SRPO') # DQL for Diffusion-QL, SRPO for SRPO algo
    # JAL for joint action learning CTCE, IND for independent learning, VD for QMIX decomposition, SEQ for sequential update/regularization
    parser.add_argument("--marltype", default='SEQ') # JAL, IND, CTDE, SEQ
    # parser.add_argument("--diff_epoch", default=159)  # 49,99,149  used for diffusion model
    
    # params for GMM
    parser.add_argument("--gmm_epoch", default=19)  # 49,99,149  used for GMM model
    parser.add_argument("--n_gmm_components", default=4, type=int, help="GMM mixture size K (MASRPO_Behavior_GMM)")
    parser.add_argument("--gmm_hidden_dim", default=512, type=int, help="ConditionalGMM MLP hidden dim")
    parser.add_argument("--critic_epoch", default=179)  # 19,39,59,79,99,119,139,159,179,199


    # Set diffusion params
    parser.add_argument("--T", default=5, type=int, help="Denoising steps for DDPM")
    parser.add_argument("--beta_schedule", default='vp', type=str)
    parser.add_argument("--seed", default=37, type=int, help="Random seed")
    parser.add_argument("--use_gpu", default=True, type=bool, help='use cuda or not')
    parser.add_argument("--device", default=4, type=int, help='cuda number')


    """   Unchangeable Params   """
    # log and save dir
    parser.add_argument("--dir", type=str, default='/data/qiaodan/code/diffmarl/results', help="tensorboard log directory")
    parser.add_argument('--dataset_dir', default='/data/qiaodan/code/diffmarl/datasets', type=str)

    # params for MPE envs
    parser.add_argument("--discrete_action", action='store_true', default=False)
    
    # params for buffer and data
    parser.add_argument("--buffer_length", default=int(2e6), type=int)
    parser.add_argument("--episode_length", default=25, type=int, help='MPE epi_length is 25, MAMuJoCo epi_length is 1000')
    parser.add_argument("--steps_per_update", default=100, type=int)   # 似乎没用
    parser.add_argument("--hidden_dim", default=64, type=int)  # DDPG hidden dim
    # set_lr is unuseful
    parser.add_argument("--set_lr", action='store_true')   # 没用
    parser.add_argument("--lr", default=3e-4, type=float)    # 大部分实验用的1e-3，这似乎是DDPG的lr，而且DDPG并不更新
    parser.add_argument("--rew_scale", default=1.0, type=float)
    
    # params for RL
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--tau", default=0.01, type=float)

    # params for evaluation
    parser.add_argument('--eval_episodes', default=10, type=int)
    parser.add_argument('--eval_interval', default=10000, type=int)

    # training steps
    parser.add_argument('--num_steps', default=int(5e5), type=int)

    # params for logging
    parser.add_argument("--logging_interval", default=500, type=int)
    parser.add_argument("--no_log", action='store_true')

    ######### args for SRPO To be revise #########

    # regularization para
    parser.add_argument('--beta', type=float, default=0.01)  
    parser.add_argument('--pretrain_model_path', type=str, default='/data/qiaodan/code/diffmarl/SRPO_premodels')
    # parser.add_argument('--critic_load_path', type=str, default='/data/qiaodan/code/diffmarl/SRPO_premodels/')  # HalfCheetah-v2_expert
    # parser.add_argument('--gmm_load_path', type=str, default='/data/qiaodan/code/diffmarl/SRPO_premodels/HalfCheetah-v2') # HalfCheetah-v2_expert
    parser.add_argument('--WT', type=str, default="VDS")
    # twin Q MLP layers
    parser.add_argument('--q_layer', type=int, default=2)
    # params for IDQL Score Network
    parser.add_argument("--actor_blocks", default=2, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--t_GassProj_dims", default=32, type=int) # 默认的是 32 dims   
    parser.add_argument("--t_embed_dims", default=64, type=int)  # 默认  64
    parser.add_argument("--sa_embed_dims", default=32, type=int)
    parser.add_argument("--resnet_hidden_dim", default=512, type=int)   # ResNet MLP hidden dim 
    parser.add_argument("--learning_rates", default=3e-4, type=float)   # lr for diffusion model
    parser.add_argument("--dilac_lr", default=3e-4, type=float)   # lr for dilac policy model
    parser.add_argument('--n_policy_epochs', default=100, type=int)
    # Dilac Policy layers, maze = 4, else = 2
    parser.add_argument('--policy_layer', type=int, default=None) 
    parser.add_argument('--regq', type=int, default=0)
    parser.add_argument('--iql_critic_lr', type=float, default=3e-4)
    ##################################################
    parser.add_argument('--save_eval_buffer', action='store_true')
    parser.add_argument("--conditional_order", default=None, type=str)
    
    config = parser.parse_args()

    # temperature_coefficients = {"simple_spread": 0.08,
    #                         "HalfCheetah-v2": 0.02,
    #                         "bandit": 0.02}
    # if config.beta is None:
    #     config.beta = temperature_coefficients[config.env_id]

    if config.policy_layer is None:
        config.policy_layer=4 if "maze" in config.env_id else 2

    if "maze" in config.env_id:
        config.regq = 1

    if config.use_gpu:
        config.device = f"cuda:{config.device}"
    else:
        config.device = "cpu"
    
    # dataset premodel path
    # if config.env_id in ['HalfCheetah-v2', 'Hopper-v2', 'Ant-v2']:
    #     config.critic_load_path = config.pretrain_model_path + f"{config.env_id}_{config.data_type}"
    #     config.gmm_load_path = config.pretrain_model_path + f"{config.env_id}_{config.data_type}"
    if config.env_id in ['simple_spread', 'simple_tag', 'simple_world']:
        config.critic_load_path = os.path.join(config.pretrain_model_path, f"{config.env_id}_{config.data_type}_seed{config.dataset_num}")
        config.gmm_load_path = f"/data/qiaodan/code/diffmarl/pretrain/gmm/{config.n_gmm_components}_GMM/{config.env_id}_{config.data_type}_Seq"


    # make envs params
    if config.env_id in ['simple_spread', 'simple_tag', 'simple_world']:
        config.lr=0.005
        config.num_steps = 200000
        # config.n_policy_epochs = 20
        config.eval_interval = 500
        config.save_buffer_interval = 25000
        config.logging_interval = 500
        config.tau = 0.005
        config.gamma=0.99
        if config.env_id == 'simple_world':
            config.steps_per_update=20
    elif config.env_id == 'bandit':
        config.num_steps = int(3e3)
        config.steps_per_update = 10 # 也没用
        config.eval_interval = 100
        config.logging_interval = 100
        config.episode_length = 1
        config.gamma=0.99
        config.dilac_lr = 0.01
        config.tau = 0.005
    else:  # MaMujoco
        # config.num_steps = int(1e6)
        config.steps_per_update = 10 # 也没用
        config.eval_interval = 10000
        config.save_buffer_interval = 25000
        config.logging_interval = 10000
        config.episode_length = 1000
        config.gamma=0.99
        config.lr = 0.0003  # 并没有进入 SRPO，只在DDPG上
        config.tau = 0.005

    if config.marltype == 'JAL':
        config.T=20
        # control Diffusion-QL, dont control SRPO

    # if config.env_id == "bandit":
    #     config.dataset_dir = config.dataset_dir + '/' + config.env_id
    # else:        
    #     config.dataset_dir = config.dataset_dir + '/' + config.env_id + '/' + config.data_type + '/' + 'seed_{}_data'.format(config.dataset_num)

    # 只用于omiga
    # if config.env_id == "HalfCheetah-v2":      
    #     config.env_args = {"scenario": config.env_id, "episode_limit": 1000, "agent_conf": '6x1', "agent_obsk": 1,}
    # elif config.env_id == "Ant-v2":
    #     config.env_args = {"scenario": config.env_id, "episode_limit": 1000, "agent_conf": '2x4', "agent_obsk": 1,}  # 需要更新确认
    # elif config.env_id == "Hopper-v2":
    #     config.env_args = {"scenario": config.env_id, "episode_limit": 1000, "agent_conf": '3x1', "agent_obsk": 1,} 

    # # combine dir
    # if config.env_id in ['HalfCheetah-v2']:
    #     config.dataset_dir = f"{config.dataset_dir}/omiga/{config.env_id}-6x1-{config.data_type}.hdf5"
    # elif config.env_id in ['Ant-v2']:
    #     config.dataset_dir = f"{config.dataset_dir}/omiga/{config.env_id}-2x4-{config.data_type}.hdf5"
    # elif config.env_id in ['Hopper-v2']:
    #     config.dataset_dir = f"{config.dataset_dir}/omiga/{config.env_id}-3x1-{config.data_type}.hdf5"


    # combine dir for mpe datasets
    if config.env_id in ['simple_spread']:
        config.dataset_dir = f"{config.dataset_dir}/simple_spread/{config.data_type}/seed_{config.dataset_num}_data"
    elif config.env_id in ['simple_tag']:
        config.dataset_dir = f"{config.dataset_dir}/simple_tag/{config.data_type}/seed_{config.dataset_num}_data"
    elif config.env_id in ['simple_world']:
        config.dataset_dir = f"{config.dataset_dir}/simple_world/{config.data_type}/seed_{config.dataset_num}_data"



    print(f"Training GMM OMSD {config.env_id}: {config.data_type}: Dataset Num {config.dataset_num}: GMM Components {config.n_gmm_components}")


    offline_train(config) 


    


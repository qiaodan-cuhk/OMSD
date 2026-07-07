""" Pretrain Joint Policy Diffusion and Individual Diffusion to get scores"""

import functools
import os
import numpy as np
import torch
import tqdm
import argparse
from datetime import datetime
from gym.spaces import Box, Discrete

from utils.buffer import ReplayBuffer

from utils.make_env import make_env
from utils.env_wrappers import DummyVecEnv

import sys
from pathlib import Path
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


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

from algorithms.SRPO import MASRPO_Behavior
from torch.utils.tensorboard import SummaryWriter

# make parallel MA-Env
def make_parallel_env(env_id, seed, discrete_action):
    def get_env_fn(rank):
        env = make_env(env_id, discrete_action=discrete_action)
        env.seed(seed + rank * 1000)
        np.random.seed(seed + rank * 1000)
        return env
    return DummyVecEnv([get_env_fn(0)])


def marginal_prob_std(t, device="cuda",beta_1=20.0,beta_0=0.1):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$."""    
    # t = torch.tensor(t, device=device)
    t = t.clone().detach().to(device)
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    alpha_t = torch.exp(log_mean_coeff)
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return alpha_t, std


def ensure_conditional_order_list(args):
    """Parse args.conditional_order into args.conditional_order_list (list[int])."""
    raw = getattr(args, "conditional_order", None)
    if raw is None:
        order = list(range(args.agent_num))
    elif isinstance(raw, str):
        order = [int(x) for x in raw.split("-") if x != ""]
    else:
        order = list(raw)
    if len(order) != args.agent_num:
        raise ValueError(
            f"conditional_order length {len(order)} is not equal to agent_num {args.agent_num}"
        )
    args.conditional_order_list = order


def train_ind_behavior(args, score_model, data_loader, agent_num, writer, start_epoch=0):
    n_epochs = 200
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    evaluation_inerval = 1
    epoch_save_interval = args.save_interval
    # best_loss = 1e3

    save_dir = os.path.join("/data/qiaodan/code/diffmarl/SRPO_premodels", f"{args.env_id}_{args.data_type}_seed{args.dataset_num}", "IND")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for step_in_epoch in range(10000):
            data = data_loader.sample(args.batch_size)
            data_i = data[agent_num]
            loss2 = score_model.update_behavior(data_i)
            avg_loss += score_model.loss.detach().cpu().numpy()
            num_items += 1
            writer.add_scalar('agent {}/episode loss'.format(agent_num), loss2, step_in_epoch)
        tqdm_epoch.set_description('Average Loss of Agent {}: {:5f}'.format(agent_num, avg_loss / num_items))

        epoch_loss = score_model.loss.detach().cpu().numpy()
        writer.add_scalar("agent {}/lr".format(agent_num), score_model.diffusion_optimizer.state_dict()['param_groups'][0]['lr'], epoch+1)

        """ Log by tensorboard"""
        if (epoch % evaluation_inerval == (evaluation_inerval -1)) or epoch==0:
            writer.add_scalar('agent {}/epoch loss'.format(agent_num), epoch_loss, epoch+1)
            writer.add_scalar('agent {}/mean epoch loss'.format(agent_num), avg_loss / num_items, epoch+1)
            # args.run.log({"loss/diffusion": score_model.loss.detach().cpu().numpy()}, step=epoch+1)

        """ Save models """
        # if args.save_model and epoch_loss < best_loss:
        #     best_loss = epoch_loss
        #     print("New lowest loss in epoch {}, Save best models".format(epoch))
        #     torch.save(score_model.state_dict(), os.path.join("/data/qiaodan/code/diffmarl/SRPO_premodels", f"{args.env_id}_{args.data_type}_seed{args.dataset_num}", "IND", "best_diffusion_{}.pth".format(agent_num)))
        #     # SRPO_premodels/env_id_level_seed/IND/best_diffusion_i.pth
        
        if args.save_model and epoch % epoch_save_interval == (epoch_save_interval - 1): 
            
            print("Save models: Epoch {}".format(epoch))
            # torch.save(score_model.state_dict(), os.path.join("/data/qiaodan/code/diffmarl/SRPO_premodels", f"{args.env_id}_{args.data_type}_seed{args.dataset_num}", "IND", "diffusion_{}_epoch{}.pth".format(agent_num, epoch)))
            torch.save(score_model.state_dict(), os.path.join(save_dir, "diffusion_{}_epoch{}.pth".format(agent_num, epoch)))
            # SRPO_premodels/env_id_level_seed/IND/diffusion_i_epoch150.pth   
        
# MPE 的 JAL 需要修改
def train_joint_behavior(args, score_model, data_loader, writer, start_epoch=0):
    n_epochs = 200
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    evaluation_inerval = 1
    epoch_save_interval = args.save_interval
    best_loss = 1e3

    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for step_in_epoch in range(10000):
            data = data_loader.sample(args.batch_size)

            # we need to concate marl datasets with [{}, {}] into one dict
            d0 = data[0]
            d1 = data[1]
            data_concate = {}
            for item in ["obs", "action", "next_obs", "next_action"]:
                data_concate[item] = torch.cat((d0[item], d1[item]), dim=1).to(args.device)
                assert data_concate[item].size()[0] == args.batch_size
            data_concate["state"] = d0["state"].to(args.device)
            data_concate["next_state"] = d0["next_state"].to(args.device)
            data_concate["rewards"] = d0["rewards"].to(args.device)
            data_concate["done"] = d0["done"].to(args.device)

            loss2 = score_model.update_behavior(data_concate)
            avg_loss += score_model.loss.detach().cpu().numpy()
            num_items += 1
            writer.add_scalar('JAL/episode loss', loss2, step_in_epoch)
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))

        epoch_loss = score_model.loss.detach().cpu().numpy()
        writer.add_scalar("JAL/lr", score_model.diffusion_optimizer.state_dict()['param_groups'][0]['lr'], epoch+1)
        
        if (epoch % evaluation_inerval == (evaluation_inerval -1)) or epoch==0:
            writer.add_scalar('JAL/epoch loss', epoch_loss, epoch+1)
            writer.add_scalar('JAL/mean epoch loss', avg_loss / num_items, epoch+1)

        """ Save models """
        if args.save_model and epoch_loss < best_loss:
            best_loss = epoch_loss
            print("New lowest loss in epoch {}, Save best models".format(epoch))
            torch.save(score_model.state_dict(), os.path.join("./SRPO_premodels", f"{args.env_id}_{args.data_type}", "JAL", "best_diffusion.pth"))
                # SRPO_premodels/env_id_level/JAL/best_diffusion.pth
        
        if args.save_model and epoch % epoch_save_interval == (epoch_save_interval - 1): 
            print("Save models: Epoch {}".format(epoch))
            torch.save(score_model.state_dict(), os.path.join("./SRPO_premodels", f"{args.env_id}_{args.data_type}", "JAL", "diffusion_{}.pth".format(epoch)))
            # SRPO_premodels/env_id_level/JAL/diffusion_150.pth

# 默认012顺序
# def train_seq_behavior(args, score_model, data_loader, agent_num, writer, start_epoch=0):

#     n_epochs = 200
#     tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
#     evaluation_inerval = args.log_interval  # 1
#     epoch_save_interval = args.save_interval  # 50
#     best_loss = 1e3


#     if agent_num == 0:

#         for epoch in tqdm_epoch:
#             avg_loss = 0.
#             num_items = 0
#             for step_in_epoch in range(10000):
#                 data = data_loader.sample(args.batch_size)
#                 data_i = data[agent_num]
#                 loss2 = score_model.update_behavior(data_i)
#                 avg_loss += score_model.loss.detach().cpu().numpy()
#                 num_items += 1
#                 writer.add_scalar('agent {}/episode loss'.format(agent_num), loss2, step_in_epoch)
#             tqdm_epoch.set_description('Average Loss of Agent {}: {:5f}'.format(agent_num, avg_loss / num_items))

#             epoch_loss = score_model.loss.detach().cpu().numpy()
#             writer.add_scalar("agent {}/lr".format(agent_num), score_model.diffusion_optimizer.state_dict()['param_groups'][0]['lr'], epoch+1)

#             """ Log by tensorboard"""
#             if (epoch % evaluation_inerval == (evaluation_inerval -1)) or epoch==0:
#                 writer.add_scalar('agent {}/epoch loss'.format(agent_num), epoch_loss, epoch+1)
#                 writer.add_scalar('agent {}/mean epoch loss'.format(agent_num), avg_loss / num_items, epoch+1)
#                 # args.run.log({"loss/diffusion": score_model.loss.detach().cpu().numpy()}, step=epoch+1)
            
#             """ Save models """
#             # if args.save_model and epoch_loss < best_loss:
#             #     best_loss = epoch_loss
#             #     print("New lowest loss in epoch {}, Save best models".format(epoch))
#             #     torch.save(score_model.state_dict(), os.path.join("/data/qiaodan/code/diffmarl/pretrain/omiga", f"{args.env_id}_{args.data_type}_Seq", "best_diffusion_agent{}.pth".format(agent_num)))
#             #     # SRPO_premodels/env_id_level/Seq/best_diffusion_i.pth
            
#             if args.save_model and epoch % epoch_save_interval == (epoch_save_interval - 1): 
#                 print("Save models: Epoch {}".format(epoch))
#                 torch.save(score_model.state_dict(), os.path.join("/data/qiaodan/code/diffmarl/pretrain/omiga_appendix", f"{args.env_id}_{args.data_type}_Seq", "diffusion_agent{}_epoch{}.pth".format(agent_num, epoch)))
#                 # SRPO_premodels/env_id_level/Seq/diffusion_i_epoch150.pth   
#     elif agent_num > 0:
#         for epoch in tqdm_epoch:
#             avg_loss = 0.
#             num_items = 0
#             for step_in_epoch in range(10000):
#                 data = data_loader.sample(args.batch_size)
#                 data_i = data[agent_num]

#                 # 收集所有前序agents的action
#                 prev_actions = []
#                 prev_next_actions = []
#                 for prev_agent in range(agent_num):
#                     prev_actions.append(data[prev_agent]['action'])
#                     prev_next_actions.append(data[prev_agent]['next_action'])
                
#                 # 将前序agents的action连接到当前agent的观察中
#                 data_i['obs'] = torch.cat([data_i['obs']] + prev_actions, dim=1).to(args.device)
#                 data_i['next_obs'] = torch.cat([data_i['next_obs']] + prev_next_actions, dim=1).to(args.device)


#                 loss2 = score_model.update_behavior(data_i)
#                 avg_loss += score_model.loss.detach().cpu().numpy()
#                 num_items += 1
#                 writer.add_scalar('agent {}/episode loss'.format(agent_num), loss2, step_in_epoch)
#             tqdm_epoch.set_description('Average Loss of Agent {}: {:5f}'.format(agent_num, avg_loss / num_items))

#             epoch_loss = score_model.loss.detach().cpu().numpy()
#             writer.add_scalar("agent {}/lr".format(agent_num), score_model.diffusion_optimizer.state_dict()['param_groups'][0]['lr'], epoch+1)

#             """ Log by tensorboard"""
#             if (epoch % evaluation_inerval == (evaluation_inerval -1)) or epoch==0:
#                 writer.add_scalar('agent {}/epoch loss'.format(agent_num), epoch_loss, epoch+1)
#                 writer.add_scalar('agent {}/mean epoch loss'.format(agent_num), avg_loss / num_items, epoch+1)
#                 # args.run.log({"loss/diffusion": score_model.loss.detach().cpu().numpy()}, step=epoch+1)

#             """ Save models """
#             # if args.save_model and epoch_loss < best_loss:
#             #     best_loss = epoch_loss
#             #     print("New lowest loss in epoch {}, Save best models".format(epoch))
#             #     torch.save(score_model.state_dict(), os.path.join("/data/qiaodan/code/diffmarl/pretrain/omiga", f"{args.env_id}_{args.data_type}_Seq", "best_diffusion_agent{}.pth".format(agent_num)))
#             #     # SRPO_premodels/env_id_level/Seq/best_diffusion_i.pth
            
#             if args.save_model and epoch % epoch_save_interval == (epoch_save_interval - 1): 
#                 print("Save models: Epoch {}".format(epoch))
#                 torch.save(score_model.state_dict(), os.path.join("/data/qiaodan/code/diffmarl/pretrain/omiga_appendix", f"{args.env_id}_{args.data_type}_Seq", "diffusion_agent{}_epoch{}.pth".format(agent_num, epoch)))
#                 # SRPO_premodels/env_id_level/Seq/diffusion_i_epoch150.pth   



def train_seq_behavior(args, score_model, data_loader, agent_num, writer, start_epoch=0):

    # 支持指定的扰动顺序
    # conditional_order = getattr(args, "conditional_order", None)
    # if conditional_order is None:
    #     conditional_order = list(range(args.agent_num))

    # if isinstance(conditional_order, str):
    #     conditional_order = [int(x) for x in conditional_order.split("-") if x != ""]

    # if len(conditional_order) != args.agent_num:
    #     raise ValueError(f"conditional_order length {len(conditional_order)} is not equal to agent_num {args.agent_num}")

    conditional_order = args.conditional_order_list

    if agent_num not in conditional_order:
        raise ValueError(f"agent {agent_num} is not in conditional_order {conditional_order}")

    agent_idx = conditional_order.index(agent_num)
    prefix_agent_idx = conditional_order[:agent_idx]

    n_epochs = 200
    tqdm_epoch = tqdm.trange(start_epoch, n_epochs)
    evaluation_inerval = args.log_interval  # 1
    epoch_save_interval = args.save_interval  # 50
    best_loss = 1e3

 


    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for step_in_epoch in range(10000):
            data = data_loader.sample(args.batch_size)

            # 这里保证取出来要训练的agent的数据
            data_i = data[agent_num]

            # 收集所有前序agents的action进行拼接
            prev_actions = []
            prev_next_actions = []
            for prev_agent in prefix_agent_idx:
                prev_actions.append(data[prev_agent]['action'])
                prev_next_actions.append(data[prev_agent]['next_action'])
            
            # 将前序agents的action连接到当前agent的观察中
            data_i['obs'] = torch.cat([data_i['obs']] + prev_actions, dim=1).to(args.device)
            data_i['next_obs'] = torch.cat([data_i['next_obs']] + prev_next_actions, dim=1).to(args.device)


            loss2 = score_model.update_behavior(data_i)
            avg_loss += score_model.loss.detach().cpu().numpy()
            num_items += 1
            writer.add_scalar('agent {}/episode loss'.format(agent_num), loss2, step_in_epoch)
        tqdm_epoch.set_description('Average Loss of Agent {}: {:5f}'.format(agent_num, avg_loss / num_items))

        epoch_loss = score_model.loss.detach().cpu().numpy()
        writer.add_scalar("agent {}/lr".format(agent_num), score_model.diffusion_optimizer.state_dict()['param_groups'][0]['lr'], epoch+1)

        """ Log by tensorboard"""
        if (epoch % evaluation_inerval == (evaluation_inerval -1)) or epoch==0:
            writer.add_scalar('agent {}/epoch loss'.format(agent_num), epoch_loss, epoch+1)
            writer.add_scalar('agent {}/mean epoch loss'.format(agent_num), avg_loss / num_items, epoch+1)
            # args.run.log({"loss/diffusion": score_model.loss.detach().cpu().numpy()}, step=epoch+1)

        """ Save models """
        # if args.save_model and epoch_loss < best_loss:
        #     best_loss = epoch_loss
        #     print("New lowest loss in epoch {}, Save best models".format(epoch))
        #     torch.save(score_model.state_dict(), os.path.join("/data/qiaodan/code/diffmarl/pretrain/omiga", f"{args.env_id}_{args.data_type}_Seq", "best_diffusion_agent{}.pth".format(agent_num)))
        #     # SRPO_premodels/env_id_level/Seq/best_diffusion_i.pth
        
        if args.save_model and epoch % epoch_save_interval == (epoch_save_interval - 1): 
            print("Save models: Epoch {}".format(epoch))
            torch.save(score_model.state_dict(), os.path.join("/data/qiaodan/code/diffmarl/pretrain/omiga_appendix", f"{args.env_id}_{args.data_type}_Seq", args.conditional_order, "diffusion_agent{}_epoch{}.pth".format(agent_num, epoch)))
            # SRPO_premodels/env_id_level/Seq/diffusion_i_epoch150.pth   
 

        

def behavior(args):
    tmp_dir = "/data/qiaodan/code/diffmarl/pretrain/omiga_appendix"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    if args.env_id in ['HalfCheetah-v2', 'Ant-v2', 'Hopper-v2']:
        env = MujocoMulti(env_args=args.env_args)
        env.seed(args.seed + 100)
        env_info = env.get_env_info()
    elif args.env_id == 'bandit':
        env = ContinuousBanditEnv()
        env_info = env.env_info
    elif args.env_id in ['simple_spread', 'simple_tag', 'simple_world']:
        env = make_parallel_env(args.env_id, args.seed, args.discrete_action)
        env_args, env_info = None, None
    else:   
        print("Env not in MaMujoco, bandit, MPE")

    # === 这里要小心维度如何影响更新 ===
    if args.env_id in ['HalfCheetah-v2', 'Ant-v2', 'Hopper-v2', 'bandit']:
        each_state_shape = [env_info['state_shape'] for _ in env.observation_space]
        # MaMujuco use obs as input
        each_obs_shape = [env_info['obs_shape'] for _ in env.observation_space]
        each_action_shape = [acsp.shape[0] for acsp in env.action_space]
        agent_num = len(each_action_shape) 
        state_dim = each_obs_shape[0]
        action_dim = each_action_shape[0]
    elif args.env_id in ['simple_spread']:
        each_state_shape = [obsp.shape[0] for obsp in env.observation_space]
        each_action_shape = [acsp.shape[0] for acsp in env.action_space]
        each_action_max = [acsp.high[0] for acsp in env.action_space]
        agent_num = len(each_action_shape) 
        state_dim = each_state_shape[0]
        action_dim = each_action_shape[0]
        action_max = each_action_max[0]
    elif args.env_id in ['simple_tag', 'simple_world']:
        adversary_indices = [i for i, agent_type in enumerate(env.agent_types) if agent_type == 'adversary']
        # 去除 agent 预训练的数据，不需要score model
        each_state_shape = [env.observation_space[i].shape[0] for i in adversary_indices]
        each_action_shape = [env.action_space[i].shape[0] for i in adversary_indices]
        each_action_max = [env.action_space[i].high[0] for i in adversary_indices]
        agent_num = len(adversary_indices) 
        state_dim = each_state_shape[0]
        action_dim = each_action_shape[0]
        action_max = each_action_max[0]
        pass
        # 这里agents区分prey和predators

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.agent_num = agent_num

    marginal_prob_std_fn = functools.partial(marginal_prob_std, device=args.device,beta_1=20.0)
    args.marginal_prob_std_fn = marginal_prob_std_fn
    if args.srpo_mode == 'IND' or args.srpo_mode == 'CTDE':
        score_model= [MASRPO_Behavior(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device) for agent in range(agent_num)]
    elif args.srpo_mode == 'JAL':
        score_model= MASRPO_Behavior(input_dim=(state_dim+action_dim)*agent_num, output_dim=action_dim*agent_num, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
        # JAL model is p(all_a|all_obs)
    elif args.srpo_mode == 'Seq':
        """ 改成适配 n agents 的结构"""
        # score_model= [MASRPO_Behavior(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device),
        #               MASRPO_Behavior(input_dim=state_dim+action_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)]
        # 第一个srpo p(a1|s)， 第二个Srpo p(a2|s,a1)

        # 现在需要支持任意扰动顺序的score model

        # score_model = [
        #                 MASRPO_Behavior(
        #                     input_dim=state_dim + action_dim + (i * action_dim if i > 0 else 0),
        #                     output_dim=action_dim,
        #                     marginal_prob_std=marginal_prob_std_fn,
        #                     args=args
        #                 ).to(args.device)
        #                 for i in range(agent_num)
        #             ]

        # 支持指定的扰动顺序
        conditional_order = getattr(args, "conditional_order", None)
        if conditional_order is None:
            conditional_order = list(range(args.agent_num))
        if isinstance(conditional_order, str):
            conditional_order = [int(x) for x in conditional_order.split("-") if x != ""]
        if len(conditional_order) != args.agent_num:
            raise ValueError(f"conditional_order length {len(conditional_order)} is not equal to agent_num {args.agent_num}")
        if args.seq_agent_id not in conditional_order:
            raise ValueError(f"agent {args.seq_agent_id} is not in conditional_order {conditional_order}")

        agent_order_idx = conditional_order.index(args.seq_agent_id)

        score_model = MASRPO_Behavior(
                            input_dim=state_dim + action_dim + (agent_order_idx * action_dim if agent_order_idx > 0 else 0),
                            output_dim=action_dim,
                            marginal_prob_std=marginal_prob_std_fn,
                            args=args
                        ).to(args.device)
                    


    if args.env_id in ['HalfCheetah-v2', 'Hopper-v2', 'bandit']:
        replay_buffer = ReplayBuffer(args.buffer_length,
                                     agent_num,
                                     [env_info['obs_shape'] for _ in env.observation_space],
                                     [acsp.shape[0] for acsp in env.action_space],
                                     is_mamujoco=True,
                                     state_dims=[env_info['state_shape'] for _ in env.observation_space],
                                     device = args.device)
    elif args.env_id in ['Ant-v2']:
        replay_buffer = ReplayBuffer(args.buffer_length,
                                     agent_num,
                                     [env_info['obs_shape'] for _ in env.observation_space],
                                     [acsp.shape[0] for acsp in env.action_space],
                                     is_mamujoco=True,
                                     state_dims=[env_info['state_shape'] for _ in env.observation_space],
                                     device = args.device,
                                     store_on_gpu=True)
    elif args.env_id in ['simple_spread', 'simple_tag', 'simple_world']:  
        replay_buffer = ReplayBuffer(
            args.buffer_length, agent_num,
            each_state_shape,
            [acsp.shape[0] if isinstance(acsp, Box) else acsp.n for acsp in env.action_space], device = args.device
        )

    if args.env_id in ['simple_spread', 'simple_tag', 'simple_world', 'bandit']:
        replay_buffer.load_batch_data(args.dataset_dir, rew_scale = args.rew_scale)
    elif args.env_id in ['HalfCheetah-v2', 'Ant-v2', 'Hopper-v2']:
        replay_buffer.load_batch_data_omiga(args.dataset_dir, rew_scale=args.rew_scale)


    """ Train Log Dir """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_log_path = os.path.join("/data/qiaodan/code/diffmarl/Diffusion_Logs",
                               "{}".format(str(args.env_id)),
                               "{}".format(args.data_type),
                               "{}_agent{}_{}".format(args.srpo_mode, args.seq_agent_id, timestamp))
    # 创建日志目录
    os.makedirs(tb_log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_path)

    # tb_log_path = os.path.join("./logs_SRPO_diffusion_pretrain", "{}_{}_{}_seed{}_{}".format(str(args.env_id), args.data_type, args.srpo_mode, args.seed, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")) )
    # writer = SummaryWriter(log_dir=tb_log_path)

    """ Model Saving Dir """
    if not os.path.exists(os.path.join("/data/qiaodan/code/diffmarl/pretrain/omiga_appendix", f"{args.env_id}_{args.data_type}_{args.srpo_mode}", args.conditional_order)):
        os.makedirs(os.path.join("/data/qiaodan/code/diffmarl/pretrain/omiga_appendix", f"{args.env_id}_{args.data_type}_{args.srpo_mode}", args.conditional_order))


    print("training behavior")
    # if args.srpo_mode == 'CTDE' or args.srpo_mode == 'IND':
    #     for i in range(agent_num):
    #         train_ind_behavior(args, score_model[i], replay_buffer, i, writer, start_epoch=0)
    if args.srpo_mode == 'CTDE' or args.srpo_mode == 'IND':
        if not (0 <= args.seq_agent_id < agent_num):
            raise ValueError(f"seq_agent_id {args.seq_agent_id} out of range [0, {agent_num - 1}]")
        train_ind_behavior(
            args,
            score_model[args.seq_agent_id],
            replay_buffer,
            args.seq_agent_id,
            writer,
            start_epoch=0
        )
    elif args.srpo_mode == 'JAL':
        train_joint_behavior(args, score_model, replay_buffer, writer, start_epoch=0)
    elif args.srpo_mode == 'Seq':
        agent_id = args.seq_agent_id
        train_seq_behavior(args, score_model, replay_buffer, agent_id, writer, start_epoch=0)
    print("finished")





def pretrain_behavior_args():
    parser = argparse.ArgumentParser()

    """   Changable params by users   """
    # Dataset selection  e.g. "simple spread_medium_0"
    parser.add_argument("--env_id", default='simple_tag', type=str, help="Name of environment") # HalfCheetah-v2 bandit
    parser.add_argument("--data_type", default='expert', type=str)  # medium-replay
    parser.add_argument("--dataset_num", default=0, type=int, help="Dataset seed number from 0-4")  # for MPE
    parser.add_argument("--seq_agent_id", default=2, type=int)  # 加速训练，直接指定训练某一个agent 0 or 1
    # train mode
    parser.add_argument("--log_interval", default=1, type=int)
    parser.add_argument("--save_interval", default=20, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default=4, type=int, help='cuda number')
    parser.add_argument("--srpo_mode", default='Seq', type=str)
    # params for Score Network
    parser.add_argument("--actor_blocks", default=2, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--t_GassProj_dims", default=32, type=int)
    parser.add_argument("--t_embed_dims", default=64, type=int)
    parser.add_argument("--sa_embed_dims", default=32, type=int)
    parser.add_argument("--resnet_hidden_dim", default=512, type=int)   # ResNet MLP hidden dim 
    parser.add_argument("--learning_rates", default=3e-4, type=float)
    parser.add_argument("--lr_anneal", default=True, type=bool)

    
    """   UnChangable params by users   """
    parser.add_argument('--dataset_dir', default='/data/qiaodan/code/diffmarl/datasets', type=str)
    parser.add_argument("--use_gpu", default=True, type=bool, help='use cuda or not')
    # params for buffer and data
    parser.add_argument("--buffer_length", default=int(2e6), type=int)  # full marl dataset is 1e6 
    parser.add_argument("--rew_scale", default=1.0, type=float)
    parser.add_argument("--save_model", default=True, type=bool)
    # continuous MPE default False
    parser.add_argument("--discrete_action", action='store_true', default=False)

    parser.add_argument("--conditional_order", default="0-1-2", type=str)

    # mixed datasets
    # parser.add_argument("--mixed_data", action='store_true', default=False)
    # parser.add_argument("--eval_models", default=False, type=bool)

    config = parser.parse_args()

    # 只用于omiga
    if config.env_id == "HalfCheetah-v2":      
        config.env_args = {"scenario": config.env_id, "episode_limit": 1000, "agent_conf": '6x1', "agent_obsk": 1,}
    elif config.env_id == "Ant-v2":
        config.env_args = {"scenario": config.env_id, "episode_limit": 1000, "agent_conf": '2x4', "agent_obsk": 1,}  # 需要更新确认
    elif config.env_id == "Hopper-v2":
        config.env_args = {"scenario": config.env_id, "episode_limit": 1000, "agent_conf": '3x1', "agent_obsk": 1,} 



    # combine dir
    if config.env_id in ['simple_spread', 'simple_tag', 'simple_world']:
        config.dataset_dir = f"{config.dataset_dir}/{config.env_id}/{config.data_type}/seed_{config.dataset_num}_data"
    elif config.env_id in ['HalfCheetah-v2']:
        config.dataset_dir = f"{config.dataset_dir}/omiga/{config.env_id}-6x1-{config.data_type}.hdf5"
    elif config.env_id in ['Ant-v2']:
        config.dataset_dir = f"{config.dataset_dir}/omiga/{config.env_id}-2x4-{config.data_type}.hdf5"
    elif config.env_id in ['Hopper-v2']:
        config.dataset_dir = f"{config.dataset_dir}/omiga/{config.env_id}-3x1-{config.data_type}.hdf5"
    elif config.env_id in ['bandit']:
        config.dataset_dir = f"{config.dataset_dir}/{config.env_id}"



    if config.use_gpu:
        config.device = f"cuda:{config.device}"
    else:
        config.device = "cpu"



    # 支持指定的扰动顺序
    # config.conditional_order = str("1-2-0")

    return config


if __name__ == "__main__":
    args = pretrain_behavior_args()
    print(f"Training Score Models {args.env_id}: {args.data_type}: {args.srpo_mode}")
    behavior(args)

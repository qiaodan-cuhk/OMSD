import torch
import torch.nn.functional as F

from utils.misc import soft_update, average_gradients
from utils.agents import DDPGAgent
import numpy as np
from utils.noise import action_noise

import copy
import torch.nn as nn
from .SRPO import SRPO, SRPO_CTDE, SRPO_ssd

# from tensorboard_logger import log_value
from torch.utils.tensorboard import SummaryWriter


import functools



# SRPO marginal_prob_std\\\
def marginal_prob_std(t, device="cuda",beta_1=20.0,beta_0=0.1):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.
    """    
    # t = torch.tensor(t, device=device)
    t = t.clone().detach().to(device)
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    alpha_t = torch.exp(log_mean_coeff)
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return alpha_t, std

# log
def log_and_print(key, value, t, writer, multi=False):
    if multi:
        print("t:", t, end=" | ")
        for i in range(len(key)):
            end = " | " if i < len(key) - 1 else "\n"
            # print("{}: {:.3f}".format(key[i], value[i][0]), end=end)
            print("{}: {}".format(key[i], value[i]), end=end)
            # log_value(key[i], value[i], t)    
            writer.add_scalar(key[i], value[i], t)
    else:
        print("t:{}, {}: {:.3f}".format(t, key, value))
        # log_value(key, value, t)
        writer.add_scalar(key, value, t)



# MARL SRPO Base Independent Learning
class BASE_SRPO(object):
    def __init__(
        self, 
        agent_init_params, # state & action dim
        agent_max_actions, 
        alg_types, 
        denoise_steps=20,
        device = 'cpu',
        adv_init_params=None,
        gamma=0.99, # RL discount gamma
        tau=0.01,  # 这个 tau 是用来 target network soft update的
        lr=0.01, 
        hidden_dim=64, 
        discrete_action=False, 
        env_id=None,
        batch_size = 100,
        config = None,
        **kwargs
    ):
        self.env_id = env_id
        self.is_mamujoco = True if self.env_id in ['HalfCheetah-v2', 'Hopper-v2', 'Ant-v2', 'bandit'] else False

        assert (ma == agent_max_actions[0] for ma in agent_max_actions)
        self.max_action = agent_max_actions[0]
        self.min_action = -self.max_action
        self.hidden_dim = hidden_dim  # 64 for DDPG

        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.tau = tau
  
        self.agent_init_params = agent_init_params
        self.state_dim = self.agent_init_params[0]['state_dim']
        self.action_dim = self.agent_init_params[0]['action_dim']
        self.device = device
        self.config = config
        
        self.gamma = gamma
        self.lr = lr
        self.discrete_action = discrete_action

        self.T = denoise_steps
        self.batch_size = batch_size

        for k, v in kwargs.items():
            setattr(self, k, v)

        marginal_prob_std_fn = functools.partial(marginal_prob_std, device=self.device, beta_1=20.0)

        self.agents = [SRPO(input_dim = self.state_dim+self.action_dim, output_dim=self.action_dim, marginal_prob_std=marginal_prob_std_fn, args=config) for agent in alg_types]
        for age in self.agents:
            age.q[0].to(self.device)


        if self.env_id in ['simple_tag', 'simple_world']:
            self.num_predators = len(agent_init_params)
            self.num_preys = len(adv_init_params)

            self.preys = [DDPGAgent(lr=lr, discrete_action=self.discrete_action, hidden_dim=self.hidden_dim, **params) for params in adv_init_params]

        self.niter = 0
  

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        actions = []

        """在simple tag和world里，nagents只考虑了diffusion的agent，没加prey agent"""
        # nagents = agents + prey (if have)
        # for i, obs in zip(range(self.nagents), observations):
        for i, obs in enumerate(observations):   
            if self.env_id in ['simple_world', 'simple_tag']:
                if i < self.num_predators:
                    predator_action = self.agents[i].SRPO_policy.select_actions(obs)  # SRPO use Dilac sample action
                    actions.append(predator_action)
                else:
                    prey_action = self.preys[i - self.num_predators].step(obs, explore=False)
                    actions.append(prey_action)
            else:
                action = self.agents[i].SRPO_policy.select_actions(obs)
                actions.append(action)
        return actions

        # actions = [array([0.99983406, 0...e=float32), array([0.9950481 , 0...e=float32), array([ 0.44896033, ...e=float32)]
        # observations = [tensor([[ 0.0000,  0... 0.0000]]), tensor([[ 0.0000,  0... 0.0000]]), tensor([[ 0.0000,  0... 0.0000]])]
        # obs = tensor([1, 18])
    

    def update(self, sample, agent_i, t, writer, run):

        curr_agent = self.agents[agent_i]

        # data reconstruction
        # 参考OMAR，mamujoco提供了state，但是训练用的还是obs
        if self.is_mamujoco:
            sample_bridge = {"s": sample["obs"],
                             "a": sample["action"],
                             "r": sample["rewards"],
                             "s_": sample["next_obs"],
                             "d": sample["done"]
            }
        else:
            sample_bridge = {"s": sample["obs"],
                             "a": sample["action"],
                             "r": sample["rewards"],
                             "s_": sample["next_obs"],
                             "d": sample["done"]
            }

        # loss_tot, epsilon, guidance, error_a = curr_agent.update_SRPO_policy(sample_bridge)
        if self.config.env_id == 'bandit':
            loss_tot, epsilon, guidance, error_a, a_plt_i = curr_agent.update_SRPO_policy(sample_bridge)
        else:
            loss_tot, epsilon, guidance, error_a = curr_agent.update_SRPO_policy(sample_bridge)
        
        """ logging metric """
        if t % self.logging_interval == 0 and not self.no_log:
            dic = {}
            dic.update({"IND/SRPO loss"+str(agent_i): loss_tot})
            dic.update({"IND/action errors"+str(agent_i): error_a.item()})
            # dic.update({"diffusion loss"+str(agent_i): epsilon})
            # dic.update({"Q gradient"+str(agent_i): guidance})
            
            log_and_print(list(dic.keys()), list(dic.values()), t, writer, multi=True)

            if run is not None:
                run.log({"IND/SRPO loss"+str(agent_i): loss_tot,
                        "IND/action errors"+str(agent_i): error_a.item()})
            
        if self.config.env_id == 'bandit':
            return epsilon, guidance, a_plt_i

    # prepare train() or eval() 
    def prep_training(self, device='cpu'):

        for a in self.agents:
            a.diffusion_behavior.train()
            a.SRPO_policy.train()
            a.q[0].train()

        fn = lambda x: x.to(device)   
        for a in self.agents:
            a.diffusion_behavior = fn(a.diffusion_behavior)
            a.SRPO_policy = fn(a.SRPO_policy)
            a.q[0] = fn(a.q[0])

        if self.env_id in ['simple_tag', 'simple_world']:
            for p in self.preys:
                p.policy = fn(p.policy)
                p.target_policy = fn(p.target_policy)

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.diffusion_behavior.eval()
            a.SRPO_policy.eval()
            a.q[0].eval()

        fn = lambda x: x.to(device)

        for a in self.agents:
            a.diffusion_behavior = fn(a.diffusion_behavior)
            a.SRPO_policy = fn(a.SRPO_policy)

        if self.env_id in ['simple_tag', 'simple_world']:
            for p in self.preys:
                p.policy = fn(p.policy)


    @classmethod
    def init_from_env(cls, env, env_id, env_info=None, agent_alg="diffusion", adversary_alg="ddpg",
                       gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                       batch_size=None, denoise_steps=20, config=None, **kwargs):
        """
        Instantiate instance of this class from multi-agent environment
        """

        # create n trainable agents without prey, alg_types = ['diff', 'diff', 'diff']
        if env_id in ['simple_tag', 'simple_world']:
            alg_types = [agent_alg for atype in env.agent_types if atype == 'adversary']
        elif env_id in ['simple_spread']:
            alg_types = [agent_alg for atype in env.agent_types]
        elif env_id in ['HalfCheetah-v2', 'Hopper-v2', 'Ant-v2']:
            alg_types = [agent_alg for atype in range(env_info['n_agents'])]
        elif env_id in ['bandit']:
            alg_types = [agent_alg for atype in range(2)]

        agent_init_params = []
        all_n_actions = []
        agent_max_actions = []
        adv_init_params = []

        # make agent_init_params, adv_init_params, agent_max_actions, all_n_actions
        if env_id in ['HalfCheetah-v2', 'Hopper-v2', 'Ant-v2']:
            for agent_idx in range(len(alg_types)):
                acsp = env_info['action_spaces'][agent_idx]
                num_in_pol = env_info['obs_shape']
                num_out_pol = acsp.shape[0]

                agent_init_params.append({'state_dim': num_in_pol, 'action_dim': num_out_pol})
                
                agent_max_actions.append(acsp.high[0])
                all_n_actions.append(acsp.shape[0])
        elif env_id == "bandit":
            for acsp, obsp in zip(env.action_space, env.observation_space):
                num_in_pol = obsp.shape[0]
                num_out_pol = acsp.shape[0]
                num_in_critic = num_in_pol + num_out_pol

                
                agent_init_params.append({'state_dim': num_in_pol, 'action_dim': num_out_pol})
                agent_max_actions.append(acsp.high[0])
                
                all_n_actions.append(acsp.shape[0])

            for i in range(1, len(all_n_actions)):
                assert (all_n_actions[i] == all_n_actions[0])  
        else:  # MPE env
            for acsp, obsp, agent_type in zip(env.action_space, env.observation_space, env.agent_types):
                num_in_pol = obsp.shape[0]
                num_out_pol = acsp.shape[0]
                num_in_critic = num_in_pol + num_out_pol

                if env_id in ['simple_spread']:
                    agent_init_params.append({'state_dim': num_in_pol, 'action_dim': num_out_pol})
                    agent_max_actions.append(acsp.high[0])
                else:
                    if agent_type == 'adversary':  # adversary 是猎人，agent 是猎物
                        agent_init_params.append({'state_dim': num_in_pol, 'action_dim': num_out_pol})
                        agent_max_actions.append(acsp.high[0])
                    elif agent_type == 'agent':
                        adv_init_params.append({'num_in_pol': num_in_pol, 'num_out_pol': num_out_pol, 'num_in_critic': num_in_critic})

                all_n_actions.append(acsp.shape[0])

            for i in range(1, len(all_n_actions)):
                assert (all_n_actions[i] == all_n_actions[0])  # 同时包括了 predator 和 prey 的动作维度


        init_dict = {
            'agent_init_params': agent_init_params,
            'agent_max_actions': agent_max_actions,
            'alg_types': alg_types,
            'denoise_steps': denoise_steps,
            'device': 'cpu',
            'adv_init_params': adv_init_params,
            'gamma': gamma, 
            'tau': tau,
            'lr': lr,
            'hidden_dim': hidden_dim,
            'discrete_action': False,
            'env_id': env_id,
            'batch_size': batch_size,   
            'config': config,
        }

        # if have additional params in kwargs
        init_dict.update(kwargs)
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        
        return instance


    def load_pretrained_preys(self, filename):
        if not torch.cuda.is_available():
            save_dict = torch.load(filename, map_location=torch.device('cpu'))
        else:
            save_dict = torch.load(filename, map_location=self.device)  # 保证prey也加载到与predator相同的device上

        if self.env_id in ['simple_tag', 'simple_world']:
            prey_params = save_dict['agent_params'][self.num_predators:]

        for i, params in zip(range(self.num_preys), prey_params):
            self.preys[i].load_params_without_optims(params)

        for p in self.preys:
            p.policy.eval()
            p.target_policy.eval()
    


# MARL SRPO Independent Learning
class IND_SRPO(BASE_SRPO):
    def __init__(
        self, 
        agent_init_params, # state & action dim
        agent_max_actions, 
        alg_types, 
        denoise_steps=20,
        device = 'cpu',
        adv_init_params=None,
        gamma=0.99, # RL discount gamma
        tau=0.01,  # 这个 tau 是用来 target network soft update的
        lr=0.01, 
        hidden_dim=64, 
        discrete_action=False, 
        env_id=None,
        batch_size = 100,
        config = None,
        **kwargs
    ):
        super().__init__(agent_init_params, # state & action dim
        agent_max_actions, 
        alg_types, 
        denoise_steps=denoise_steps,
        device = device,
        adv_init_params=adv_init_params,
        gamma=gamma, # RL discount gamma
        tau=tau,  # 这个 tau 是用来 target network soft update的
        lr=lr, 
        hidden_dim=hidden_dim, 
        discrete_action=discrete_action, 
        env_id=env_id,
        batch_size = batch_size,
        config = config,
        **kwargs)
        self.class_mode = 'Independent MASRRPO'




"""
Joint action learning: central critic, joint policy diffusion
Independent learning: independent critic, indpendent policy diffusion
QMIX learning: central critic -> VD ind critic, ind policy diffusion
Seq: central critic, joint policy diffusion, partial diffusion to regularize ind policy
"""


# MARL SRPO Joint Action Learning
class JAL_SRPO(BASE_SRPO):
    def __init__(
        self, 
        agent_init_params, # state & action dim
        agent_max_actions, 
        alg_types, 
        denoise_steps=20,
        device = 'cpu',
        adv_init_params=None,
        gamma=0.99, # RL discount gamma
        tau=0.01,  # 这个 tau 是用来 target network soft update的
        lr=0.01, 
        hidden_dim=64, 
        discrete_action=False, 
        env_id=None,
        batch_size = 100,
        config = None,
        **kwargs
    ):
        super().__init__(agent_init_params, # state & action dim
        agent_max_actions, 
        alg_types, 
        denoise_steps=denoise_steps,
        device = device,
        adv_init_params=adv_init_params,
        gamma=gamma, # RL discount gamma
        tau=tau,  # 这个 tau 是用来 target network soft update的
        lr=lr, 
        hidden_dim=hidden_dim, 
        discrete_action=discrete_action, 
        env_id=env_id,
        batch_size = batch_size,
        config = config,
        **kwargs)

        # in JAL, the state and action dim are multiplex by agent numbers
        self.state_dim = self.agent_init_params[0]['state_dim'] * self.nagents  # use concate obs as input
        self.action_dim = self.agent_init_params[0]['action_dim'] * self.nagents

        marginal_prob_std_fn = functools.partial(marginal_prob_std, device=self.device, beta_1=20.0)

        self.agents = [SRPO(input_dim = self.state_dim+self.action_dim, output_dim=self.action_dim, marginal_prob_std=marginal_prob_std_fn, args=config)]
        self.agents[0].q[0].to(self.device)
        self.class_mode = 'JAL MASRRPO'


    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """

        # input torch_obs is [torch(1,18), torch(1,18), torch(1,18)]
        # output actions is [array(2), array(2), array(2)]
        # for JAL, one diff model generate all agents actions

        if self.env_id in ['simple_world', 'simple_tag']:
            obs_predator = observations[:self.num_predators]
            obs_predator = torch.cat(obs_predator, dim=1)
            obs_prey = observations[self.num_predators:]
            
            predator_actions = self.agents[0].SRPO_policy.select_actions(obs_predator) # input tensor[1,obs_dim * n] output array[1, act_dim * n]
            actions = np.split(predator_actions, self.num_predators, axis=0)

            for i, obs in zip(obs_prey):
                prey_action = self.preys[i].step(obs, explore=False)
                actions.append(prey_action)
        else:
            observations = torch.cat(observations, dim=1)
            joint_actions = self.agents[0].SRPO_policy.select_actions(observations)
            # joint_actions = joint_actions.squeeze(0)
            # actions = np.split(joint_actions, self.nagents, axis=0)
            actions = np.split(joint_actions, self.nagents, axis=1)

        assert actions[0].shape[-1] == self.agent_init_params[0]['action_dim']

        return actions


    def update(self, sample, t, writer, run):

        JAL_agent = self.agents[0]

        jal_obs = [sample_i['obs'] for sample_i in sample]
        jal_obs = torch.cat(jal_obs, dim=1)
        assert jal_obs.size()[0] == self.batch_size

        jal_act = [sample_i['action'] for sample_i in sample]  # nagents * batch_size * act_dim
        jal_act = torch.cat(jal_act, dim=1) # batch_size * (nagents*act_dim)

        jal_rew = sample[0]['rewards']  # [batch_size]
        jal_done = sample[0]['done'] # [batch_size]

        jal_next_obs = [sample_i['next_obs'] for sample_i in sample]
        jal_next_obs = torch.cat(jal_next_obs, dim=1)

        # data reconstruction
        # 参考OMAR，mamujoco提供了state，但是训练用的还是obs
        if self.is_mamujoco:
            sample_bridge = {"s": jal_obs,
                             "a": jal_act,
                             "r": jal_rew,
                             "s_": jal_next_obs,
                             "d": jal_done
            }
        else:
            sample_bridge = {"s": jal_obs,
                             "a": jal_act,
                             "r": jal_rew,
                             "s_": jal_next_obs,
                             "d": jal_done
            }

        if self.config.env_id == 'bandit':
            loss_tot, episilon, guidance, error_a, a_plt = JAL_agent.update_SRPO_policy(sample_bridge)
        else:
            loss_tot, episilon, guidance, error_a = JAL_agent.update_SRPO_policy(sample_bridge)

        # loss_tot, epsilon, guidance, error_a = JAL_agent.update_SRPO_policy(sample_bridge)
        
        """ logging metric """
        if t % self.logging_interval == 0 and not self.no_log:
            dic = {}
            dic.update({"JAL/SRPO loss": loss_tot})
            # dic.update({"diffusion loss of JAL": epsilon})
            # dic.update({"Q gradient of JAL": guidance})
            dic.update({"JAL/action errors": error_a.item()})
            
            log_and_print(list(dic.keys()), list(dic.values()), t, writer, multi=True)

            if run is not None:
                run.log({"JAL/SRPO loss": loss_tot,
                        "JAL/action errors": error_a.item()})
        
        if self.config.env_id == 'bandit':
            return  episilon, guidance, a_plt


"""
get other agents' gaussian policy,
calculate Q_total and Ai
diffusion score from single agent miu_i(a|s)

Final loss is E_pi^i_new, pi^ -i [A^i + log miu_i] d/ pi_i
"""


# naive CTDE SRPO Learning
class CTDE_SRPO(BASE_SRPO):
    def __init__(
        self, 
        agent_init_params, # state & action dim
        agent_max_actions, 
        alg_types, 
        denoise_steps=20,
        device = 'cpu',
        adv_init_params=None,
        gamma=0.99, # RL discount gamma
        tau=0.01,  # 这个 tau 是用来 target network soft update的
        lr=0.01, 
        hidden_dim=64, 
        discrete_action=False, 
        env_id=None,
        batch_size = 100,
        config = None,
        **kwargs
    ):
        super().__init__(agent_init_params, # state & action dim
        agent_max_actions, 
        alg_types, 
        denoise_steps=denoise_steps,
        device = device,
        adv_init_params=adv_init_params,
        gamma=gamma, # RL discount gamma
        tau=tau,  # 这个 tau 是用来 target network soft update的
        lr=lr, 
        hidden_dim=hidden_dim, 
        discrete_action=discrete_action, 
        env_id=env_id,
        batch_size = batch_size,
        config = config,
        **kwargs)
        
        marginal_prob_std_fn = functools.partial(marginal_prob_std, device=self.device, beta_1=20.0)

        config.alg_types = alg_types # used for SRPO_CTDE

        self.agents = [SRPO_CTDE(input_dim = self.state_dim+self.action_dim,
                                 output_dim=self.action_dim,
                                 marginal_prob_std=marginal_prob_std_fn,
                                 args=config) for agent in alg_types]
        for age in self.agents:
            age.q[0].to(self.device)


    def update(self, samples, t, writer, run):
        # Loss i = Q(s, a-, a, a+) + beta score i，这里所有人的action是由每个人的policy采样出来的，dilac policy所以是确定性的
        # 每个 agent 计算 Q value 都拿到别人policy进行sample，或者输入之前每个人都用当前policy sample构造当前的joint action给所有人一起使用
        # CTDE 不需要考虑 sequential 问题，给定s直接所有人take action

        joint_a = []
        joint_s = []
        for agent_id, current_agent in enumerate(self.agents):
                   
            s = samples[agent_id]['obs']
            current_agent.diffusion_behavior.eval()
            a_curr = self.agents[agent_id].SRPO_policy(s).detach()   #用作计算Q值的joint actions，detach gradients且不需要添加gradient
            joint_s.append(s)
            joint_a.append(a_curr)  
            
        joint_states = torch.cat(joint_s, dim=1)

        # joint_states = torch.cat((samples[0]["obs"], samples[1]["obs"]), axis=1)
        # joint a = [a1, a2], feed in for qs = q[0].target(joint a, joint s)
        if self.config.env_id == 'bandit':
            epi_all = []
            guide_all = []
            a_plt_all = []
        
        for agent_id, current_agent in enumerate(self.agents):
            # get data i with joint s+a
            if self.is_mamujoco:
                sample_bridge = {"s": samples[agent_id]["obs"],
                                "a": samples[agent_id]["action"],
                                "r": samples[agent_id]["rewards"],
                                "s_": samples[agent_id]["next_obs"],
                                "d": samples[agent_id]["done"],
                                "s_joint": joint_states,
                                "a_joint": joint_a,
                }
            else:
                sample_bridge = {"s": samples[agent_id]["obs"],
                                "a": samples[agent_id]["action"],
                                "r": samples[agent_id]["rewards"],
                                "s_": samples[agent_id]["next_obs"],
                                "d": samples[agent_id]["done"],
                                "s_joint": joint_states,
                                "a_joint": joint_a,
                }

            # Use Joint Q/A and individual score to update
            if self.config.env_id == 'bandit':
                loss_tot, episilon, guidance, error_a, a_plt_i = current_agent.update_SRPO_policy(sample_bridge, agent_id)
            else:
                loss_tot, episilon, guidance, error_a = current_agent.update_SRPO_policy(sample_bridge, agent_id)
                
            if self.config.env_id == 'bandit':
                epi_all.append(episilon)
                guide_all.append(guidance)
                a_plt_all.append(a_plt_i)
            
            """ logging metric """
            if t % self.logging_interval == 0 and not self.no_log:
                dic = {}
                dic.update({"CTDE/SRPO loss"+str(agent_id): loss_tot.item()})
                dic.update({"CTDE/action errors"+str(agent_id): error_a.item()})
                # dic.update({"diffusion loss"+str(agent_i): epsilon})
                # dic.update({"Q gradient"+str(agent_i): guidance})
                
                log_and_print(list(dic.keys()), list(dic.values()), t, writer, multi=True)

                if run is not None:
                    run.log({"CTDE/SRPO loss"+str(agent_id): loss_tot.item(),
                            "CTDE/action errors"+str(agent_id): error_a.item()})
                

        
        if self.config.env_id == 'bandit':
            return epi_all, guide_all, a_plt_all



class OMSD(CTDE_SRPO):
    def __init__(
        self, 
        agent_init_params, # state & action dim
        agent_max_actions, 
        alg_types, 
        denoise_steps=20,
        device = 'cpu',
        adv_init_params=None,
        gamma=0.99, # RL discount gamma
        tau=0.01,  # 这个 tau 是用来 target network soft update的
        lr=0.01, 
        hidden_dim=64, 
        discrete_action=False, 
        env_id=None,
        batch_size = 100,
        config = None,
        **kwargs
    ):
        super().__init__(agent_init_params, # state & action dim
        agent_max_actions, 
        alg_types, 
        denoise_steps=denoise_steps,
        device = device,
        adv_init_params=adv_init_params,
        gamma=gamma, # RL discount gamma
        tau=tau,  # 这个 tau 是用来 target network soft update的
        lr=lr, 
        hidden_dim=hidden_dim, 
        discrete_action=discrete_action, 
        env_id=env_id,
        batch_size = batch_size,
        config = config,
        **kwargs)


        marginal_prob_std_fn = functools.partial(marginal_prob_std, device=self.device, beta_1=20.0)

        config.alg_types = alg_types # used for SRPO_CTDE

        # 第一个agent不变，策略、score、critic都是跟CTDE一样，更新也是
        # 第二个agent仅改变score，critic和策略网络不变
        # self.agents = [
        #     SRPO_CTDE(
        #         input_dim=self.state_dim + self.action_dim,
        #         output_dim=self.action_dim,
        #         marginal_prob_std=marginal_prob_std_fn,
        #         args=config
        #     ) if i == 0 else SRPO_ssd(
        #         input_dim=self.state_dim + (i+1)*self.action_dim,
        #         output_dim=self.action_dim,
        #         marginal_prob_std=marginal_prob_std_fn,
        #         agent_idx=i,
        #         args=config
        #     )
        #     for i in range(self.nagents)
        # ]

        """这里需要支持conditional order来设置network的维度"""
        # 支持指定的扰动顺序
        conditional_order = getattr(config, "conditional_order", None)
        if conditional_order is None:
            conditional_order = list(range(self.nagents))
        if isinstance(conditional_order, str):
            conditional_order = [int(x) for x in conditional_order.split("-") if x != ""]
        if len(conditional_order) != self.nagents:
            raise ValueError(f"conditional_order length {len(conditional_order)} is not equal to nagents {self.nagents}")
        if set(conditional_order) != set(range(self.nagents)):
            raise ValueError(f"conditional_order {conditional_order} must contain all agents 0 to {self.nagents-1}")
        
        self.conditional_order = conditional_order
        # 创建映射：真实 agent_id -> 在顺序中的位置
        self.agent_to_stage = {agent_id: stage_idx for stage_idx, agent_id in enumerate(conditional_order)}

        self.agents = []
        for i in range(self.nagents):
            stage_idx = self.agent_to_stage[i]  # 当前 agent 在顺序中的位置
            if stage_idx == 0:
                # 第一个位置的 agent 使用 CTDE 结构
                self.agents.append(SRPO_CTDE(
                    input_dim=self.state_dim + self.action_dim,
                    output_dim=self.action_dim,
                    marginal_prob_std=marginal_prob_std_fn,
                    args=config
                ))
            else:
                # 后续位置的 agent 使用 ssd 结构，输入维度 = state + action + prefix_actions
                input_dim = self.state_dim + self.action_dim + stage_idx * self.action_dim
                self.agents.append(SRPO_ssd(
                    input_dim=input_dim,
                    output_dim=self.action_dim,
                    marginal_prob_std=marginal_prob_std_fn,
                    agent_idx=stage_idx,  # 传入在顺序中的位置
                    args=config
                ))


        for age in self.agents:
            age.q[0].to(self.device)


    def update(self, samples, t, writer, run):
        # Loss i = Q(s, a-, a, a+) + beta score i，这里所有人的action是由每个人的policy采样出来的，dilac policy所以是确定性的
        # 每个 agent 计算 Q value 都拿到别人policy进行sample，或者输入之前每个人都用当前policy sample构造当前的joint action给所有人一起使用
        # CTDE 不需要考虑 sequential 问题，给定s直接所有人take action


        """这里需要支持conditional order来设置joint a和joint s的维度"""


        
        joint_a = []
        joint_s = []
        for agent_id, current_agent in enumerate(self.agents):
            s = samples[agent_id]['obs']
            current_agent.diffusion_behavior.eval()
            """这里使用的是局部obs而不是全局state，mamujoco和mpe都是"""
            # 每个agent的dilac policy输出动作，用作计算Q值的joint actions，detach gradients且不需要添加gradient
            a_curr = self.agents[agent_id].SRPO_policy(s).detach()   
            joint_s.append(s)
            joint_a.append(a_curr)  
        # 最后得到的joint_s = [obs, obs, obs]  joint_a = [a0, a1, a2]
            
        joint_states = torch.cat(joint_s, dim=1)  # [3*obs, 1]

        # joint_states = torch.cat((samples[0]["obs"], samples[1]["obs"]), axis=1)
        # joint a = [a1, a2], feed in for qs = q[0].target(joint a, joint s)
        if self.config.env_id == 'bandit':
            epi_all = []
            guide_all = []
            a_plt_all = []
        
        """这里要修改，因为这个是为2 agent的第二个agent设计的，要修改joint a和joints，加入 agent idx 判断"""
        for agent_id, current_agent in enumerate(self.agents):
            # get data i with joint s+a
            if self.is_mamujoco:
                sample_bridge = {"s": samples[agent_id]["obs"],
                                "a": samples[agent_id]["action"],
                                "r": samples[agent_id]["rewards"],
                                "s_": samples[agent_id]["next_obs"],
                                "d": samples[agent_id]["done"],
                                "s_joint": joint_states,
                                "a_joint": joint_a,
                }
            else:
                sample_bridge = {"s": samples[agent_id]["obs"],  # MPE 也是 obs 不是 state
                                "a": samples[agent_id]["action"],
                                "r": samples[agent_id]["rewards"],
                                "s_": samples[agent_id]["next_obs"],  # MPE 也是 obs 不是 state
                                "d": samples[agent_id]["done"],
                                "s_joint": joint_states,
                                "a_joint": joint_a,
                }

            # Use Joint Q/A and individual score to update
            if self.config.env_id == 'bandit':
                loss_tot, episilon, guidance, error_a, a_plt_i = current_agent.update_SRPO_policy(sample_bridge, agent_id)
            else:
                loss_tot, episilon, guidance, error_a = current_agent.update_SRPO_policy(sample_bridge, agent_id)
                
            if self.config.env_id == 'bandit':
                epi_all.append(episilon)
                guide_all.append(guidance)
                a_plt_all.append(a_plt_i)
            
            """ logging metric """
            if t % self.logging_interval == 0 and not self.no_log:
                dic = {}
                dic.update({"SEQ/SRPO loss"+str(agent_id): loss_tot.item()})
                dic.update({"SEQ/action errors"+str(agent_id): error_a.item()})
                # dic.update({"diffusion loss"+str(agent_i): epsilon})
                # dic.update({"Q gradient"+str(agent_i): guidance})
                
                log_and_print(list(dic.keys()), list(dic.values()), t, writer, multi=True)

                # if run is not None:
                #     run.log({"SEQ/SRPO loss"+str(agent_id): loss_tot.item(),
                #             "SEQ/action errors"+str(agent_id): error_a.item()})
                
        
        if self.config.env_id == 'bandit':
            return epi_all, guide_all, a_plt_all


    def update_ordered(self, samples, t, writer, run):
        # Loss i = Q(s, a-, a, a+) + beta score i，这里所有人的action是由每个人的policy采样出来的，dilac policy所以是确定性的
        # 每个 agent 计算 Q value 都拿到别人policy进行sample，或者输入之前每个人都用当前policy sample构造当前的joint action给所有人一起使用
        # CTDE 不需要考虑 sequential 问题，给定s直接所有人take action

        
        # 这里只调用dilac policy生成联合动作获取梯度，所以不需要按照condition order
        joint_a = []
        joint_s = []
        for agent_id, current_agent in enumerate(self.agents):
            s = samples[agent_id]['obs']
            current_agent.diffusion_behavior.eval()
            """这里使用的是局部obs而不是全局state，mamujoco和mpe都是"""
            # 每个agent的dilac policy输出动作，用作计算Q值的joint actions，detach gradients且不需要添加gradient
            a_curr = self.agents[agent_id].SRPO_policy(s).detach()   
            joint_s.append(s)
            joint_a.append(a_curr)  
        # 最后得到的joint_s = [obs, obs, obs]  joint_a = [a0, a1, a2]
            
        joint_states = torch.cat(joint_s, dim=1)  # [3*obs, 1]

        # joint_states = torch.cat((samples[0]["obs"], samples[1]["obs"]), axis=1)
        # joint a = [a1, a2], feed in for qs = q[0].target(joint a, joint s)
        if self.config.env_id == 'bandit':
            epi_all = []
            guide_all = []
            a_plt_all = []
        
        # 额外配置一个按照order顺序的joint action
        joint_a_ordered = []
        for agent_id in self.conditional_order:
            joint_a_ordered.append(joint_a[agent_id])
        # joint_a_ordered = torch.cat(joint_a_ordered, dim=1)


        """这里的顺序更新，需要考虑conditional order来采样prefix agent的动作"""
        # 这里需要按照order顺序更新
        # for agent_id, current_agent in enumerate(self.agents):
        # for order_agent_id in self.conditional_order:
        for update_idx, agent_id in enumerate(self.conditional_order):
            # agent_id = self.conditional_order.index(order_agent_id)
            current_agent = self.agents[agent_id]
            # get data i with joint s+a
            if self.is_mamujoco:
                sample_bridge = {"s": samples[agent_id]["obs"],
                                "a": samples[agent_id]["action"],
                                "r": samples[agent_id]["rewards"],
                                "s_": samples[agent_id]["next_obs"],
                                "d": samples[agent_id]["done"],
                                "s_joint": joint_states,
                                "a_joint": joint_a,
                                "a_joint_ordered": joint_a_ordered,
                }
            else:
                sample_bridge = {"s": samples[agent_id]["obs"],  # MPE 也是 obs 不是 state
                                "a": samples[agent_id]["action"],
                                "r": samples[agent_id]["rewards"],
                                "s_": samples[agent_id]["next_obs"],  # MPE 也是 obs 不是 state
                                "d": samples[agent_id]["done"],
                                "s_joint": joint_states,
                                "a_joint": joint_a,
                                "a_joint_ordered": joint_a_ordered,
                }

            
            # Use Joint Q/A and individual score to update
            if self.config.env_id == 'bandit':
                loss_tot, episilon, guidance, error_a, a_plt_i = current_agent.update_SRPO_policy(sample_bridge, agent_id)
            else:
                # loss_tot, episilon, guidance, error_a = current_agent.update_SRPO_policy(sample_bridge, agent_id)
                # 按照条件顺序更新
                loss_tot, episilon, guidance, error_a = current_agent.update_SRPO_policy_ordered(sample_bridge, agent_id, update_idx)
                
            if self.config.env_id == 'bandit':
                epi_all.append(episilon)
                guide_all.append(guidance)
                a_plt_all.append(a_plt_i)
            
            """ logging metric """
            if t % self.logging_interval == 0 and not self.no_log:
                dic = {}
                dic.update({"SEQ/SRPO loss"+str(agent_id): loss_tot.item()})
                dic.update({"SEQ/action errors"+str(agent_id): error_a.item()})
                # dic.update({"diffusion loss"+str(agent_i): epsilon})
                # dic.update({"Q gradient"+str(agent_i): guidance})
                
                log_and_print(list(dic.keys()), list(dic.values()), t, writer, multi=True)

                # if run is not None:
                #     run.log({"SEQ/SRPO loss"+str(agent_id): loss_tot.item(),
                #             "SEQ/action errors"+str(agent_id): error_a.item()})
                
        
        if self.config.env_id == 'bandit':
            return epi_all, guide_all, a_plt_all
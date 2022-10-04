# ===== add python path ===== #
import glob
import sys
import os
PATH = os.getcwd()
for dir_idx, dir_name in enumerate(PATH.split('/')):
    dir_path = '/'.join(PATH.split('/')[:(dir_idx+1)])
    file_list = [os.path.basename(sub_dir) for sub_dir in glob.glob(f"{dir_path}/.*")]
    if '.git_package' in file_list:
        PATH = dir_path
        break
if not PATH in sys.path:
    sys.path.append(PATH)
# =========================== #

# Isaac Gym
from isaacgym import gymapi, gymutil

# Isaac Gym Envs
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from utils.jackal_env.task2 import Jackal
from agent import Agent

# Others
from collections import deque
import numpy as np
import torch
import wandb
import time
import yaml
import sys
import os

''' existing parameters
    'sim_device': 'cuda:0',
    'pipeline': 'gpu',
    'graphics_device_id': 0,
    'flex': False,
    'physx': False,
    'num_threads': 0,
    'subscenes': 0,
    'slices': 0,
'''
custom_parameters = [
    {"name": "--headless", "action": "store_true", "help": "Force display off at all times."},
    {"name": "--test", "action": "store_true", "help": "For test."},
    {"name": "--cfg_env", "type": str, "default": "env.yaml", "help": "Configuration file for environment."},
    {"name": "--cfg_train", "type": str, "default": "train.yaml", "help": "Configuration file for training."},
    {"name": "--save_freq", "type": int, "default": int(2e5), "help": "Agent save frequency."},
    {"name": "--total_steps", "type": int, "default": int(1e7), "help": "Total number of environmental steps."},
    {"name": "--update_steps", "type": int, "default": int(1e4), "help": "Number of environmental steps for updates."},
    {"name": "--seed", "type": int, "default": 1, "help": "Seed."},
]

def train(args):
    # default
    seed = args.seed
    total_steps = args.total_steps
    save_freq = args.save_freq
    set_seed(seed)

    # define environment
    env = Jackal(args, force_render=(not args.headless))

    # define agent
    with open(args.cfg_train, 'r') as f:
        agent_args = yaml.load(f, Loader=yaml.SafeLoader)
    agent_args['device'] = env.device
    agent_args['state_dim'] = env.observation_space.shape[0]
    agent_args['action_dim'] = env.action_space.shape[0]
    agent_args['action_bound_min'] = env.action_space.low
    agent_args['action_bound_max'] = env.action_space.high
    agent_args['num_envs'] = env.num_envs
    agent_args['save_dir'] = f"results/{agent_args['name']}_s{seed}"
    num_transitions_per_env = int(args.update_steps/env.num_envs)
    agent = Agent(agent_args)

    # for wandb
    wandb.init(project='[Isaac-Gym-Jackal] PPO', config=agent_args)

    buffer_len = 100
    score_buffer = deque(maxlen=buffer_len)
    cv_buffer = deque(maxlen=buffer_len)
    len_buffer = deque(maxlen=buffer_len)
    cur_score = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    cur_cv = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    cur_steps = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    env.reset(torch.linspace(0, env.num_envs - 1, env.num_envs, device=env.device, dtype=torch.long))
    states = env.compute_observations().clone().detach()
    step = 0
    while step < total_steps:
        states_list = []
        actions_list = []
        rewards_list = []
        dones_list = []
        fails_list = []
        next_states_list = []
        for _ in range(num_transitions_per_env):
            step += env.num_envs
            actions = agent.getAction(states, is_train=True)
            _, rewards, dones, infos = env.step(actions)
            if not args.headless: env.render()
            next_states = env.next_obs_buf.clone().detach()
            costs = env.cost_buf.clone().detach()
            cv = costs >= 0.5

            cur_score += rewards
            cur_cv += cv
            cur_steps += 1

            new_ids = (dones > 0).nonzero(as_tuple=False)
            score_buffer.extend(cur_score[new_ids][:, 0].cpu().numpy().tolist())
            len_buffer.extend(cur_steps[new_ids][:, 0].cpu().numpy().tolist())
            cv_buffer.extend(cur_cv[new_ids][:, 0].cpu().numpy().tolist())
            fails = torch.logical_and(cur_steps < env.max_episode_length, dones)
            cur_score[new_ids] = 0
            cur_steps[new_ids] = 0
            cur_cv[new_ids] = 0

            states_list.append(states)
            actions_list.append(actions)
            rewards_list.append(rewards)
            dones_list.append(dones)
            fails_list.append(fails)
            next_states_list.append(next_states)
            states = next_states

        score = np.mean(score_buffer)
        cv = np.mean(cv_buffer)
        ep_len = np.mean(len_buffer)

        states_tensor = torch.cat(states_list, dim=0).detach()
        actions_tensor = torch.cat(actions_list, dim=0).detach()
        rewards_tensor = torch.cat(rewards_list, dim=0).detach()
        dones_tensor = torch.cat(dones_list, dim=0).detach()
        fails_tensor = torch.cat(fails_list, dim=0).detach()
        next_states_tensor = torch.cat(next_states_list, dim=0).detach()

        v_loss, p_loss, kl, entropy = agent.train(states_tensor, actions_tensor, rewards_tensor, dones_tensor, fails_tensor, next_states_tensor)
        log_data = {"score":score, "episode length":ep_len, 'cv':cv, "value loss":v_loss, "policy loss":p_loss, "kl":kl, "entropy":entropy}
        print(log_data)
        wandb.log(log_data)

        if step%save_freq == 0:
            agent.save()


def test(args):
    # default
    seed = args.seed
    total_steps = args.total_steps
    set_seed(seed)

    # define environment
    env = Jackal(args, virtual_screen_capture=args.headless, force_render=(not args.headless))

    # define agent
    with open(args.cfg_train, 'r') as f:
        agent_args = yaml.load(f, Loader=yaml.SafeLoader)
    agent_args['device'] = env.device
    agent_args['state_dim'] = env.observation_space.shape[0]
    agent_args['action_dim'] = env.action_space.shape[0]
    agent_args['action_bound_min'] = env.action_space.low
    agent_args['action_bound_max'] = env.action_space.high
    agent_args['num_envs'] = env.num_envs
    agent_args['save_dir'] = f"results/{agent_args['name']}_s{seed}"
    agent = Agent(agent_args)

    env.reset(torch.linspace(0, env.num_envs - 1, env.num_envs, device=env.device, dtype=torch.long))
    states = env.compute_observations().clone().detach()
    step = 0
    # states_list = []
    while step < total_steps:
    # while step < int(10000):
        step += env.num_envs
        # actions = agent.getAction(states, is_train=False)
        actions = agent.getAction(states, is_train=True)
        _, rewards, dones, infos = env.step(actions)
        if not args.headless: env.render()
        next_states = env.next_obs_buf.clone().detach()
        # states_list.append(states)
        states = next_states
    # states = torch.cat(states_list, dim=0)
    # print(torch.mean(states, dim=0))
    # print(torch.std(states, dim=0))



if __name__ == "__main__":
    set_np_formatting()
    args = gymutil.parse_arguments(
        description="IsaacGym",
        custom_parameters=custom_parameters,
    )
    if args.test:
        test(args)
    else:
        train(args)

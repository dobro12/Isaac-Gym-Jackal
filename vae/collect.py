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
from utils.jackal_env.task import DummyJackal

# Others
import numpy as np
import torch
import time
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
    {"name": "--headless", "action": "store_true", "help": "Force display off at all times"},
    {"name": "--cfg_env", "type": str, "default": "env.yaml", "help": "Configuration file for environment"},
    {"name": "--cfg_train", "type": str, "default": "train.yaml", "help": "Configuration file for training"},
]

def main(args):
    task = DummyJackal(args, force_render=True)
    device = task.device
    num_envs = task.num_envs
    action_dim = task.num_actions

    env_ids = torch.linspace(0, num_envs-1, num_envs, device=device, dtype=torch.long)
    task.reset(env_ids)
    for i in range(1000):
        actions = torch.rand((num_envs, action_dim), dtype=torch.float32, device=args.sim_device)
        task.step(actions)
        task.render(mode="human")
        task.save_depth_img()
        

if __name__ == "__main__":
    set_np_formatting()
    args = gymutil.parse_arguments(
        description="IsaacGym",
        custom_parameters=custom_parameters,
    )
    main(args)

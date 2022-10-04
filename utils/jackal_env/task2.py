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
from isaacgym import gymtorch
from isaacgym import gymapi

# Isaac Gym Envs
from isaacgymenvs.tasks.base.vec_task import VecTask
from utils.jackal_env.task import DummyJackal
from utils.jackal_env.task import quat_rot
from vae.decoder import Decoder
from vae.encoder import Encoder

# Others
import numpy as np
import pickle
import torch
import yaml
import time
import cv2

class Jackal(DummyJackal):
    def __init__(self, args, virtual_screen_capture=False, force_render=False):
        super().__init__(args, virtual_screen_capture, force_render)

        # define VAE encoder
        self.encoder = Encoder().to(device=self.device)
        self.decoder = Decoder().to(device=self.device)
        checkpoint = torch.load(f"{PATH}/vae/model/checkpoint")
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])


    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # ==== start to retrieve image tensor ==== #
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        total_camera_tensor = torch.stack(self.camera_tensors, dim=0).to(device=self.device)
        self.gym.end_access_image_tensors(self.sim)
        # ======================================== #

        total_camera_tensor = total_camera_tensor.view(-1, 1, self.camera_height, self.camera_width)
        total_camera_tensor[total_camera_tensor == -np.inf] = 0
        total_camera_tensor = torch.clamp(total_camera_tensor, -self.max_depth, 0.0)
        total_camera_tensor = (1.0 - total_camera_tensor/(torch.min(total_camera_tensor) + 1e-8))
        with torch.no_grad():
            encoded_tensor, _, _ = self.encoder(total_camera_tensor)
            encoded_tensor = encoded_tensor.detach()

        inv_robot_quat = torch.tensor([-1.0, -1.0, -1.0, 1.0], device=self.device, dtype=torch.float)
        inv_robot_quat = self.robot_quat[env_ids]*inv_robot_quat
        inv_robot_quat = inv_robot_quat[:, [3,0,1,2]]
        rel_goal_pos = self.goal_pos[env_ids] - self.robot_pos[env_ids]
        rel_goal_pos = quat_rot(inv_robot_quat, rel_goal_pos)[:, :2]

        goal_dist = torch.sqrt(torch.sum(torch.square(rel_goal_pos), dim=-1, keepdim=True))
        goal_dir = rel_goal_pos/goal_dist
        goal_dist = torch.clamp(goal_dist, 0.0, self.max_goal_dist).flatten()
        lin_vel = torch.sqrt(torch.sum(torch.square(self.robot_lin_vel[env_ids, :]), dim=-1))
        ang_vel = torch.sqrt(torch.sum(torch.square(self.robot_ang_vel[env_ids, :]), dim=-1))
        lin_acc = (lin_vel - self.pre_robot_lin_vel[env_ids])*self.sim_params.dt
        self.pre_robot_lin_vel[env_ids] = lin_vel.clone().detach()

        self.obs_buf[env_ids, :2] = goal_dir
        self.obs_buf[env_ids, 2] = goal_dist
        self.obs_buf[env_ids, 3] = lin_vel
        self.obs_buf[env_ids, 4] = ang_vel
        self.obs_buf[env_ids, 5] = lin_acc/8.0
        self.obs_buf[env_ids, 6:38] = encoded_tensor[env_ids, :]
        return self.obs_buf

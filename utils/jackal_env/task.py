# Isaac Gym
from isaacgym import gymtorch
from isaacgym import gymapi

# Isaac Gym Envs
from isaacgymenvs.tasks.base.vec_task import VecTask

# Others
import numpy as np
import pickle
import torch
import yaml
import time
import cv2
import os

ABS_PATH = os.path.dirname(__file__)

class DummyJackal(VecTask):
    def __init__(self, args, virtual_screen_capture=False, force_render=False):
        # load environmental configurations
        with open(args.cfg_env, 'r') as f:
            self.cfg = yaml.load(f, Loader=yaml.SafeLoader)

        # ==== should be defined for BaseTask ==== #
        self.cfg["env"]["numObservations"] = 6 + 32
        self.cfg["env"]["numActions"] = 2
        self.cfg["sim"]["use_gpu_pipeline"] = (args.pipeline.lower() == "gpu")
        self.cfg["sim"]["physx"]["num_threads"] = args.num_threads
        # ======================================== #

        # environmental parameters
        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.max_goal_dist = self.cfg["env"]["maxGoalDist"]
        self.goal_dist_threshold = self.cfg["env"]["goalDistThreshold"]
        # set PD coefficient for Jackal velocity controller
        self.Kp = 100.0
        self.Kd = 100.0
        # Kinematics
        self.num_obstacles = 8
        self.num_goal_seq = 3
        self.camera_width = 96
        self.camera_height = 64
        self.camera_fov = 87.0
        self.num_saved_img = 0
        self.max_depth = 3.0
        self.obstacle_size = 0.25*np.sqrt(2.0)
        self.robot_size = 0.5
        self.h_coeff = 10.0
        self.wheel_radius = 0.098
        self.chasis_width = 0.187795*2.0
        self.action_scale = 1.5

        # call parent's __init__
        super().__init__(
            config=self.cfg, rl_device=args.sim_device, sim_device=args.sim_device, 
            graphics_device_id=args.graphics_device_id, headless=args.headless,
            virtual_screen_capture=virtual_screen_capture, force_render=force_render,
        )

        # for buffer
        self.cost_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.next_obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.hazard_step_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int)
        self.goal_idx_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)

        # reset camera pose
        if self.viewer != None:
            cam_pos = gymapi.Vec3(*[10.0, 10.0, 10.0])
            cam_target = gymapi.Vec3(*[-1.0, -1.0, -1.0])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # for observation
        self.num_actors = 1 + 1 + self.num_obstacles + 4 
        # robot & goal & obstacles & walls
        self.actor_jackal_idx = 0
        self.actor_goal_idx = self.actor_jackal_idx + 1
        self.actor_obstacle_idx = self.actor_goal_idx + 1
        self.actor_wall_idx = self.actor_obstacle_idx + self.num_obstacles
        root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(root_state).view(self.num_envs, self.num_actors, 13)
        # last dimension: 13 = pos (3) + quat (4) + lin_vel (3) + ang_vel (3)
        self.robot_state_tensor = self.root_state_tensor[:, self.actor_jackal_idx, :]
        self.goal_state_tensor = self.root_state_tensor[:, self.actor_goal_idx, :]
        self.obstacle_states_tensor = self.root_state_tensor[:, self.actor_obstacle_idx:self.actor_wall_idx, :]
        self.wall_states_tensor = self.root_state_tensor[:, self.actor_wall_idx:, :]
        self.robot_pos =self.robot_state_tensor[:, :3] 
        self.robot_quat = self.robot_state_tensor[:, 3:7]
        self.robot_lin_vel = self.robot_state_tensor[:, 7:10]
        self.robot_ang_vel = self.robot_state_tensor[:, 10:13]
        self.goal_pos = self.goal_state_tensor[:, :3]
        self.pre_robot_lin_vel = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.pre_goal_dist = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        # for action
        self.action_transfrom = torch.tensor([
            [1.0/self.wheel_radius,                     1.0/self.wheel_radius], 
            [-0.5*self.chasis_width/self.wheel_radius,  0.5*self.chasis_width/self.wheel_radius]
        ], device=self.device, dtype=torch.float)

        # for contact sensor
        # net contact force index:
        # 0~10: jackal, 11: goal, 12~19: obstacle, 20~23: wall
        # print(self.gym.get_actor_rigid_body_names(self.envs[0], self.jackal_handles[0])) =>
        # ['chassis_link', 'front_ball', 'front_fender_link', 'front_left_wheel_link', 
        # 'front_right_wheel_link', 'left_wheel_link', 'rear_ball', 'rear_fender_link', 
        # 'rear_left_wheel_link', 'rear_right_wheel_link', 'right_wheel_link']
        net_contact_force = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.net_contact_force_tensor = gymtorch.wrap_tensor(net_contact_force)
        self.net_contact_force_tensor = self.net_contact_force_tensor.view(self.num_envs, -1, 3)
        self.chassis_contact_force_tensor = self.net_contact_force_tensor[:, 0, :]

        # for randomization
        #  ========== To randomly spawn obstacle & goal ========== #
        with open(f"{ABS_PATH}/candidate.pkl", "rb") as f:
            obstacles_pose_list, goal_pose_seq_list = pickle.load(f)
        # for obstacle pose
        residual_pose_list = [0.25, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        residual_pose_list = [residual_pose_list for i in range(self.num_obstacles)]
        residual_pose_list = np.array([residual_pose_list for i in range(100)])
        obstacles_pose_list = np.concatenate([obstacles_pose_list, residual_pose_list], axis=2)
        # for goal pose
        residual_pose_list = [1.25, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        residual_pose_list = [residual_pose_list for i in range(self.num_goal_seq)]
        residual_pose_list = np.array([residual_pose_list for i in range(100)])
        goal_pose_seq_list = np.concatenate([goal_pose_seq_list, residual_pose_list], axis=2)
        # ======================================================== #
        self.obstacles_pose_torch = torch.tensor(obstacles_pose_list, device=self.device, dtype=torch.float)
        self.goal_pose_seq_torch = torch.tensor(goal_pose_seq_list, device=self.device, dtype=torch.float)
        self.init_robot_pose = torch.tensor(
            [0.0, 0.0, 0.06344, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
            device=self.device, dtype=torch.float)
        self.all_actor_indices = torch.arange(self.num_actors * self.num_envs, dtype=torch.long, 
                                                device=self.device).view(self.num_envs, self.num_actors)
        self.goal_pose_seq = torch.zeros((self.num_envs, self.num_goal_seq, 13), device=self.device, dtype=torch.float)

        # reset at first
        env_ids = torch.linspace(0, self.num_envs-1, self.num_envs, device=self.device, dtype=torch.long)
        self.reset(env_ids)


    def reset(self, env_ids):
        self.goal_idx_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.hazard_step_buf[env_ids] = 0
        self.pre_robot_lin_vel[env_ids] = 0.0

        sampled_indices = torch.randint(0, len(self.obstacles_pose_torch), (len(env_ids),), device=self.device) 
        sampled_obstacles_pose = self.obstacles_pose_torch[sampled_indices, ...]
        sampled_goal_pose_seq = self.goal_pose_seq_torch[sampled_indices, ...]
        sampled_goal_pose = sampled_goal_pose_seq[:, 0, :]
        self.goal_pose_seq[env_ids, :, :] = sampled_goal_pose_seq.clone().detach()

        self.robot_state_tensor[env_ids, :] = self.init_robot_pose
        self.goal_state_tensor[env_ids, :] = sampled_goal_pose
        self.obstacle_states_tensor[env_ids, :, :] = sampled_obstacles_pose

        actor_indices = self.all_actor_indices[env_ids, :10].flatten()
        actor_indices_int32 = actor_indices.to(device=self.device, dtype=torch.int32)

        result = self.gym.set_actor_root_state_tensor_indexed(self.sim, 
            gymtorch.unwrap_tensor(self.root_state_tensor), 
            gymtorch.unwrap_tensor(actor_indices_int32), 
            len(actor_indices_int32))
        if not result:
            print("reset fail.")

        # update pre_goal_dist
        rel_goal_pos = self.goal_pos[env_ids, :2] - self.robot_pos[env_ids, :2]
        self.pre_goal_dist[env_ids] = torch.sqrt(torch.sum(torch.square(rel_goal_pos), dim=-1))

    def create_sim(self):
        '''called from super().__init__()
        '''
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def compute_reward(self):
        rel_goal_pos = self.goal_pos[:, :2] - self.robot_pos[:, :2]
        goal_dist = torch.sqrt(torch.sum(torch.square(rel_goal_pos), dim=-1))
        contacts = torch.norm(self.chassis_contact_force_tensor, dim=-1)

        self.rew_buf[:], self.reset_buf[:], self.hazard_step_buf[:] = compute_reward_done(
            goal_dist, self.pre_goal_dist, self.cost_buf, torch.tensor(self.goal_dist_threshold, device=self.device),
            self.reset_buf, self.progress_buf, self.hazard_step_buf, torch.tensor(self.max_episode_length, device=self.device)
        )

        env_ids = (goal_dist < self.goal_dist_threshold).nonzero(as_tuple=False).squeeze(-1)
        self.pre_goal_dist = goal_dist.clone().detach()

        # update goal pos
        if len(env_ids) > 0:
            self.goal_idx_buf[env_ids] = torch.remainder(self.goal_idx_buf[env_ids] + 1, self.num_goal_seq)

            self.goal_state_tensor[env_ids, :] = self.goal_pose_seq[env_ids, self.goal_idx_buf[env_ids], :]
            actor_indices = self.all_actor_indices[env_ids, self.actor_goal_idx].flatten()
            actor_indices_int32 = actor_indices.to(device=self.device, dtype=torch.int32)
            result = self.gym.set_actor_root_state_tensor_indexed(
                self.sim, 
                gymtorch.unwrap_tensor(self.root_state_tensor), 
                gymtorch.unwrap_tensor(actor_indices_int32), 
                len(actor_indices_int32))
            if not result:
                print("fail to respawn goal.")

            # update pre_goal_dist
            rel_goal_pos = self.goal_pos[env_ids, :2] - self.robot_pos[env_ids, :2]
            self.pre_goal_dist[env_ids] = torch.sqrt(torch.sum(torch.square(rel_goal_pos), dim=-1))

    def compute_cost(self):
        robot_pos = self.robot_pos[:, :2]
        obstacle_dists = []
        # ==== distance from boxes ==== #
        for i in range(8):
            obstacle_pos = self.obstacle_states_tensor[:, i, :2]
            rel_pos = obstacle_pos - robot_pos
            obstacle_dist = torch.sqrt(torch.sum(torch.square(rel_pos), dim=-1)) - self.obstacle_size
            obstacle_dists.append(obstacle_dist)
        # ============================= #

        # ==== distance from walls ==== #
        obstacle_dist = 4.0 - robot_pos[:, 0] - 0.1
        obstacle_dists.append(obstacle_dist)
        obstacle_dist = 4.0 - robot_pos[:, 1] - 0.1
        obstacle_dists.append(obstacle_dist)
        obstacle_dist = robot_pos[:, 0] + 4.0 - 0.1
        obstacle_dists.append(obstacle_dist)
        obstacle_dist = robot_pos[:, 1] + 4.0 - 0.1
        obstacle_dists.append(obstacle_dist)
        # ============================= #

        obstacle_dists = torch.stack(obstacle_dists)
        min_dists, _ = torch.min(obstacle_dists, dim=0)
        costs = self.robot_size - min_dists
        self.cost_buf[:] = 1.0/(1.0 + torch.exp(-costs*self.h_coeff))

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # todo
        return self.obs_buf
        
    def pre_physics_step(self, actions):
        actions_tensor = actions.clone().to(self.device)
        actions_tensor[:, 0] = torch.clamp(actions_tensor[:, 0], 0.0, 1.0)
        actions_tensor[:, 1] = torch.clamp(actions_tensor[:, 1], -1.0, 1.0)
        actions_tensor = self.action_scale*torch.matmul(actions_tensor, self.action_transfrom)
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(actions_tensor))

    def post_physics_step(self):
        self.progress_buf += 1

        self.compute_observations()
        self.compute_cost()
        self.compute_reward()

        self.next_obs_buf[:] = self.obs_buf
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset(env_ids)
            self.compute_observations(env_ids)

    def save_depth_img(self):
        if not os.path.exists("img"):
            os.mkdir("img")
        for env_idx in range(self.num_envs):
            img_dir = f"img/env{env_idx}"
            if not os.path.exists(img_dir):
                os.mkdir(img_dir)

        self.gym.render_all_camera_sensors(self.sim)

        # ==== start to retrieve image tensor ==== #
        self.gym.start_access_image_tensors(self.sim)
        for env_idx in range(self.num_envs):
            cam_img = self.camera_tensors[env_idx].cpu().numpy()
            cam_img[cam_img == -np.inf] = -self.max_depth
            cam_img[cam_img < -self.max_depth] = -self.max_depth
            cam_img = 255.0*(1.0 - cam_img/np.min(cam_img))
            cam_img = cam_img.astype(np.uint8)
            img_dir = f"img/env{env_idx}"
            cv2.imwrite(f"{img_dir}/depth_{self.num_saved_img:04d}.png", cam_img)
        self.gym.end_access_image_tensors(self.sim)
        # ======================================== #
        self.num_saved_img += 1


    def _create_ground_plane(self):
        ''' called from create_sim()
        '''
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        ''' called from create_sim()
        '''
        # define plane on which environments are initialized
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # get asset info from configuration file
        asset_root = f"{ABS_PATH}/urdf"
        asset_file = "jackal.urdf"
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        # jackal asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.use_mesh_materials = True
        jackal_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(jackal_asset)

        # change friction property
        shape_names = self.gym.get_asset_rigid_body_names(jackal_asset)
        shape_indices = self.gym.get_asset_rigid_body_shape_indices(jackal_asset)
        front_ball_idx = 0
        rear_ball_idx = 0
        for i, idx in enumerate(shape_indices):
            if shape_names[i] == "front_ball":
                front_ball_idx = idx.start
            elif shape_names[i] == "rear_ball":
                rear_ball_idx = idx.start
        shape_props = self.gym.get_asset_rigid_shape_properties(jackal_asset)
        shape_props[front_ball_idx].friction = 0.0
        shape_props[front_ball_idx].rolling_friction = 0.0
        shape_props[front_ball_idx].torsion_friction = 0.0
        shape_props[rear_ball_idx].friction = 0.0
        shape_props[rear_ball_idx].rolling_friction = 0.0
        shape_props[rear_ball_idx].torsion_friction = 0.0
        self.gym.set_asset_rigid_shape_properties(jackal_asset, shape_props)

        # goal asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.density = 0.1
        radius = 0.25
        goal_asset = self.gym.create_sphere(self.sim, radius, asset_options)

        # obstacle asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.density = 1.0
        obstacle_size = [0.5, 0.5, 0.5]
        obstacle_asset = self.gym.create_box(self.sim, *obstacle_size, asset_options)

        # wall asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        wall_size = [7.8, 0.2, 1.0]
        wall_asset = self.gym.create_box(self.sim, *wall_size, asset_options)

        # jackal spawn pose
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.06344)
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # goal spawn pose
        goal_pose = gymapi.Transform()
        goal_pose.p = gymapi.Vec3(-1.0, -1.0, 0.25)
        goal_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        goal_color = gymapi.Vec3(1.0, 0.0, 0.0)

        # obstacle spawn pose
        obstacle_pos_list = [[2.0, 2.0, 0.25], [2.0, 1.0, 0.25], [2.0, 0.0, 0.25], [2.0, -1.0, 0.25], 
                            [2.0, -2.0, 0.25], [1.0, 1.0, 0.25], [1.0, 0.0, 0.25], [1.0, -1.0, 0.25]]
        obstacle_pose = gymapi.Transform()
        obstacle_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        obstacle_color = gymapi.Vec3(0.0, 0.0, 1.0)

        # set collision filter 
        # (if two objects have same values, not collide)
        collision_filter_jackal = 1
        collision_filter_wall = 2

        # camera transform
        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(0.3, 0.0, 0.2)
        local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.pi/6.0)

        self.jackal_handles = []
        self.jackal_rigid_handles = []
        self.camera_handles = []
        self.camera_tensors = []
        self.envs = []
        for i in range(num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            collision_group = i

            # get jackal actor handler
            jackal_handle = self.gym.create_actor(env_ptr, jackal_asset, pose, "jackal", collision_group, collision_filter_jackal)
            jackal_body_handle = self.gym.get_actor_root_rigid_body_handle(env_ptr, jackal_handle)
            self.jackal_rigid_handles.append(jackal_body_handle)

            # change jackal actor property
            dof_props = self.gym.get_actor_dof_properties(env_ptr, jackal_handle)
            dof_props['driveMode'][:] = gymapi.DOF_MODE_VEL
            dof_props['stiffness'][:] = self.Kp
            dof_props['damping'][:] = self.Kd
            dof_props['hasLimits'][:] = False
            self.gym.set_actor_dof_properties(env_ptr, jackal_handle, dof_props)

            # add camera sensor
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.camera_width
            camera_props.height = self.camera_height
            camera_props.horizontal_fov = self.camera_fov
            # ======= important to use tensor !!! ======= #
            camera_props.enable_tensors = True
            # =========================================== #
            camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
            self.gym.attach_camera_to_body(camera_handle, env_ptr, jackal_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
            self.camera_handles.append(camera_handle)

            # obtain camera tensor
            cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.ImageType.IMAGE_DEPTH)
            torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
            self.camera_tensors.append(torch_cam_tensor)

            # add goal
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_pose, "goal", collision_group, collision_filter_jackal)
            self.gym.set_rigid_body_color(env_ptr, goal_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, goal_color)

            # add obstacles
            for obs_idx in range(len(obstacle_pos_list)):
                obstacle_pos = obstacle_pos_list[obs_idx]
                obstacle_pose.p = gymapi.Vec3(*obstacle_pos)
                obs_handle = self.gym.create_actor(env_ptr, obstacle_asset, obstacle_pose, f"obstacle{obs_idx}", collision_group, collision_filter_wall)
                self.gym.set_rigid_body_color(env_ptr, obs_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, obstacle_color)

            # add wall
            for wall_idx in range(4):
                wall_pose = gymapi.Transform()
                if wall_idx % 2 == 0:
                    if wall_idx == 0:
                        wall_pose.p = gymapi.Vec3(0.0, 4.0, 0.5)
                    else:
                        wall_pose.p = gymapi.Vec3(0.0, -4.0, 0.5)
                    wall_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
                else:
                    if wall_idx == 1:
                        wall_pose.p = gymapi.Vec3(4.0, 0.0, 0.5)
                    else:
                        wall_pose.p = gymapi.Vec3(-4.0, 0.0, 0.5)
                    wall_pose.r = gymapi.Quat(0.0, 0.0, 0.7071, 0.7071)
                wall_handle = self.gym.create_actor(env_ptr, wall_asset, wall_pose, "wall", collision_group, collision_filter_wall)

            self.envs.append(env_ptr)
            self.jackal_handles.append(jackal_handle)


#####################################################################
### ======================= jit functions ======================= ###
#####################################################################

@torch.jit.script
def compute_reward_done(goal_dist, pre_goal_dist, costs, goal_threshold, reset_buf, progress_buf, hazard_step_buf, max_episode_length):
    contact = torch.logical_and(progress_buf > 1, costs >= 0.5)
    reward = pre_goal_dist - goal_dist
    reward = torch.where(goal_dist < goal_threshold, torch.ones_like(reward) + reward, reward)
    reward = reward - contact*0.01
    hazard_step = torch.where(contact, contact + hazard_step_buf, torch.zeros_like(hazard_step_buf))
    reset = torch.where(progress_buf >= max_episode_length, torch.ones_like(reset_buf), hazard_step >= 100)
    return reward, reset, hazard_step

@torch.jit.script
def quat_mul(q, r):
    # q: (w, x, y, z)
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape
    
    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((x, y, z, w), dim=1).view(original_shape)

@torch.jit.script
def quat_rot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)
    
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)
# 27(except for the right arm's 2 dof) dof
# sparse reward FT


# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import math

from isaacgym import gymtorch
from isaacgym import gymapi
### !!!!!!!!!!!!!!!! ###
# from isaacgymenvs.utils.torch_jit_utils import scale, unscale, quat_mul, quat_conjugate, quat_to_angle_axis, \
#     to_torch, get_axis_params, torch_rand_float, tensor_clamp, compute_heading_and_up, compute_rot, normalize_angle
from isaacgymenvs.utils.torch_jit_utils import *

from isaacgymenvs.tasks.base.vec_task import VecTask
# from isaacgymenvs.tasks.amp.utils_amp.motion_lib import MotionLib
from isaacgymenvs.tasks.interos.motion_lib import MotionLib
from isaacgymenvs.tasks.interos.poselib.skeleton.skeleton3d import SkeletonTree
import math
from isaacgym import gymutil
from isaacgym.terrain_utils import *
from tensorboardX import SummaryWriter
import cv2

import torch
import random

from typing import Tuple


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


## since the reward need to be calculated separately, we need to deine a new class for the task
# only a humanoid, a table, adn a ball are used in this task
class HumanoidPingpongTiltNESSparse27DOF(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        # 这个参数目前在代码中没有用到
        self.angular_velocity_scale = self.cfg["env"].get("angularVelocityScale", 0.1)
        self.contact_force_scale = self.cfg["env"]["contactForceScale"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.heading_weight = self.cfg["env"]["headingWeight"]
        self.up_weight = self.cfg["env"]["upWeight"]
        self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        self.energy_cost_scale = self.cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self.cfg["env"]["deathCost"]
        self.termination_height = self.cfg["env"]["terminationHeight"]
        self.key_body_names = self.cfg["env"]["keyBodyNames"]

        self.debug_viz = False # self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        #######################################
        self.future_dt = self.cfg["env"]["futureDt"]
        ###
        self.terrain = None

        self.is_train = not self.cfg["env"]["test"]
        self.is_g1 = self.cfg["env"].get("is_g1", False)

        self.character_idx = 0

        self.targets = None
        ###################################
        
        self.cfg["env"]["numObservations"] = 121+192 # 30+30+27+27+3+3 + 1 + 192

        # num_dofs
        self.cfg["env"]["numActions"] = 27

        self.alpha = self.cfg["env"]["alphaVelocityReward"]
        self.power_coefficient = self.cfg["env"]["powerCoefficient"]
        # self.penalty = self.cfg["env"]["penalty"]
        self.hit_table_reward = self.cfg["env"]["hitTableReward"]
        self.not_hit_table_penalty = self.cfg["env"]["nothitTablePenalty"]
        self.cross_net_reward_float = self.cfg["env"]["crossNetRewardFloat"]
        self.die_penalty_float = self.cfg["env"]["diePenaltyFloat"]
        self.hit_paddle_reward = self.cfg["env"]["hitPaddleReward"]
        self.miss_paddle_penalty_coefficient = self.cfg["env"]["missPaddlePenaltyCoefficient"]

        # speed_scale: 控制仿真中关节运动或动画的速度缩放比例
        self.speed_scale = 1.0

        # pingpong_ball initial speed and tilt angle range
        # self.initial_speed_range = (8.8, 9.6)  # 初速度范围 (单位: 米/秒)
        # self.tilt_angle_range = (-5.0, 5.0)  # 倾斜角范围 (单位: 度)
        # self.tilt_z_angle_range = (22.0, 28.0) # z方向倾斜角范围（单位：度）
        # 2/24 19:00之后的参数设计
        self.initial_speed_range = (5.0, 5.4)  # 初速度范围 (单位: 米/秒)
        self.tilt_angle_range = (-8.0, 3.0)  # 倾斜角范围 (单位: 度)
        self.tilt_z_angle_range = (14.0, 24.0) # z方向倾斜角范围（单位：度）
        # 2/27-3/1 增加 初位置范围 (单位: 米)
        self.initial_pos_y_range = (-0.5, 0.1)  
        self.initial_pos_z_range = (0.96, 1.05)

        # 3/6 更加diverse
        
        # # 2/24 19:00之后的参数设计
        # self.initial_speed_range = (5.2, 5.6)  # 初速度范围 (单位: 米/秒)
        # self.tilt_angle_range = (-10.0, 10.0)  # 倾斜角范围 (单位: 度)
        # self.tilt_z_angle_range = (8.0, 22.0) # z方向倾斜角范围（单位：度）
        # # 2/27-3/1 增加 初位置范围 (单位: 米)
        # self.initial_pos_y_range = (-0.5, 0.1)  
        # self.initial_pos_z_range = (0.95, 1.05)

        # # 3/7 增加角速度的尝试
        # self.angular_speed_range = (0.0, 10.0) # 角速度范围 (单位: 弧度/秒)

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        if not self.headless:
            cam_pos = gymapi.Vec3(50.0, 25.0, 2.4)
            cam_target = gymapi.Vec3(45.0, 25.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.actors_per_env = 3 # humanoid1 + table + ball2
        self.dofs_per_env = 27 # huamnoid1 dof
        self.rigid_bodies_per_env = 40 + 1 + 1 # humanoid1 + table + ball2
        # sensors_per_env = 3

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)

        # sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        # self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, 2 * 6)
        ################ !!!!!!!!!!!!!!!!!!!!!!! #################
        # self.all_actor_indices = torch.arange(5 * self.num_envs, dtype=torch.int32).view(self.num_envs, 5).cuda()
        
        self.all_actor_indices = to_torch(self.actor_handles, dtype=torch.long, device=self.device)
        # print("self.all_actor_indices.shape", self.all_actor_indices.shape)
        # print("self.all_actor_indices.dtype", self.all_actor_indices.dtype)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # dof force
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.dofs_per_env)

        ### !!!!!!!!!!!!!!!!!!!!! ###
        body_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.body_states_id = self.cfg["env"].get("bodyStatesIdPingpong", None)
        self.body_states_id = to_torch(self.body_states_id, dtype=torch.long, device=self.device)

        self.body_balance_states_id = self.cfg["env"].get("bodyStatesIdBalance", None)
        self.body_balance_states_id = to_torch(self.body_balance_states_id, dtype=torch.long, device=self.device)

        if self.body_states_id is not None:
            self.body_states = gymtorch.wrap_tensor(body_states).view(self.num_envs, self.num_humanoid_bodies+2, 13)
            self.body_states[:, self.body_balance_states_id, 7:10] = 0.0
        else:
            self.body_states = gymtorch.wrap_tensor(body_states).view(self.num_envs, self.num_humanoid_bodies+2, 13)

        self.initial_body_states = self.body_states.clone()

        # rigid body state tensors
        self.rb_states = gymtorch.wrap_tensor(_rb_states)
        self.vec_rb_states = self.rb_states.view(self.num_envs, -1, 13) # torch.Size([16, 42, 13]) 这个应该和self.body_states是一样的
        self.humanoid1_rb_states = self.vec_rb_states[:, 0:40, :] # torch.Size([16, 40, 13])
        self.humanoid1_pelvis_rb_states = self.vec_rb_states[:, 0, :] # torch.Size([16, 13])
        self.humanoid1_paddle_rb_states = self.vec_rb_states[:, 39, :] # torch.Size([16, 13])
        # self.humanoid2_rb_states = self.vec_rb_states[:, 40:80, :] # torch.Size([16, 40, 13])
        # self.humanoid2_paddle_rb_states = self.vec_rb_states[:, 79, :] # torch.Size([16, 13])

        # root state tensors
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.vec_root_states = self.root_states.view(self.num_envs, self.actors_per_env, 13)
        self.initial_root_states = self.root_states.clone()
        self.initial_vec_root_states = self.vec_root_states.clone()
        # print(self.initial_vec_root_states[:, 2, 10:13])

        # humanoid and ball state tensors
        self.humanoid1_root_states = self.vec_root_states[:, 0, :] # shape: (num_envs, 13)
        # self.humanoid2_root_states = self.vec_root_states[:, 1, :]
        self.table_root_states = self.vec_root_states[:, 1, :]
        # self.ball1_root_states = self.vec_root_states[:, 2, :]
        self.ball2_root_states = self.vec_root_states[:, 2, :] # shape: (num_envs, 13)
        
        # initial root states
        self.initial_vec_root_states = self.vec_root_states.clone() # shape: (num_envs, 3, 13)
        self.initial_pos = self.initial_vec_root_states[:, :, 0:3]
        # print("self.initial_pos", self.initial_pos)
        self.initial_rot = self.initial_vec_root_states[:, :, 3:7]
        self.humanoid1_initial_pos = self.initial_vec_root_states[:, 0, 0:3]
        self.humanoid1_initial_rot = self.initial_vec_root_states[:, 0, 3:7]
        # self.humanoid2_initial_pos = self.initial_vec_root_states[:, 1, 0:3]
        # self.humanoid2_initial_rot = self.initial_vec_root_states[:, 1, 3:7]
        self.table_initial_pos = self.initial_vec_root_states[:, 1, 0:3]
        self.table_initial_rot = self.initial_vec_root_states[:, 1, 3:7]
        # self.ball1_initial_pos = self.initial_vec_root_states[:, 3, 0:3]
        # self.ball1_initial_rot = self.initial_vec_root_states[:, 3, 3:7]
        self.ball2_initial_pos = self.initial_vec_root_states[:, 2, 0:3]
        self.ball2_initial_rot = self.initial_vec_root_states[:, 2, 3:7]

        ### !!! ###
        self.dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        self.vec_dof_states = self.dof_states.view(self.num_envs, self.dofs_per_env, 2) # shape: (num_envs, dof, 2)
        self.dof_pos = self.vec_dof_states[..., 0]
        self.dof_vel = self.vec_dof_states[..., 1]

        # self.initial_dof_states = self.vec_dof_states.clone() # shape: (num_envs, dof, 2)
        # print("self.initial_dof_states", self.initial_dof_states)
        self.initial_dof_states = torch.zeros_like(self.vec_dof_states, device=self.device, dtype=torch.float)
        self.initial_dof_pos = self.initial_dof_states[..., 0]
        self.initial_dof_vel = self.initial_dof_states[..., 1]
        # print("self.initial_dof_states", self.initial_dof_states)

        # zero_tensor = torch.tensor([0.0], device=self.device)
        # self.initial_dof_pos = torch.where(self.dof_limits_lower > zero_tensor, self.dof_limits_lower,
        #                                    torch.where(self.dof_limits_upper < zero_tensor, self.dof_limits_upper, self.initial_dof_pos))
        # self.initial_dof_vel = torch.zeros_like(self.dof_vel, device=self.device, dtype=torch.float)

        # initialize some data used later on
        # self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        # self.heading_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        # self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        # self.basis_vec0 = self.heading_vec.clone()
        # self.basis_vec1 = self.up_vec.clone()

        self.dt = self.cfg["sim"]["dt"] # self.cfg["sim"]["substeps"]
        # self.potentials = to_torch([-1000./self.dt], device=self.device).repeat(self.num_envs)
        # self.prev_potentials = self.potentials.clone()


        motion_file = cfg['env']['motion_file']
        motion_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), motion_file)
        # self._load_motion(motion_file_path)
        # self.init_obs_demo()

        self.num_steps = 0
        # self.reward_calculated = torch.zeros(self.num_envs, dtype=torch.bool, device = self.device)  # 标志张量，记录每个环境是否已计算桌面击球的奖励/惩罚
        self.paddle_condition_calculated = torch.zeros(self.num_envs, dtype=torch.bool, device = self.device)  # 标志张量，记录每个环境是否已计算paddle击球的奖励
        # self.missed_ball_calculated = torch.zeros(self.num_envs, dtype=torch.bool, device = self.device)  # 标志张量，记录每个环境是否已计算未接触球的奖励
        # self.net_condition_calculated = torch.zeros(self.num_envs, dtype=torch.bool, device = self.device)  # 标志张量，记录每个环境是否已计算球过网的奖励
        # self.no_bounce_before_half_mask = torch.ones(self.num_envs, dtype=torch.bool, device = self.device)  # 标志张量，记录每个环境是否在半程之前没有发生反弹

        self.die_penalty_calculated = torch.zeros(self.num_envs, dtype=torch.bool, device = self.device)  # 标志张量，记录每个环境是否已计算乒乓球掉落惩罚
        self.humanoid_die_calculated = torch.zeros(self.num_envs, dtype=torch.bool, device = self.device)  # 标志张量，记录每个环境是否已计算人物掉落惩罚
        self.hit_table_calculated = torch.zeros(self.num_envs, dtype=torch.bool, device = self.device)  # 标志张量，只记录乒乓球初次落点位置的reward or penalty

        # 计数器
        self.closer_to_paddle_count = torch.zeros(self.num_envs, dtype=torch.bool, device = self.device)  # 记录每个环境的接近paddle次数
        self.hit_paddle_count = torch.zeros(self.num_envs, dtype=torch.bool, device = self.device)  # 记录每个环境的paddle击球次数
        self.cross_net_count = torch.zeros(self.num_envs, dtype=torch.bool, device = self.device)  # 记录每个环境的球击中次数
        self.hit_table_count = torch.zeros(self.num_envs, dtype=torch.bool, device = self.device)  # 记录每个环境的乒乓球落点次数
        self.fall_down_count = torch.zeros(self.num_envs, dtype=torch.bool, device = self.device)  # 记录每个环境的Humanoid摔倒次数
        
        if not self.headless:
            self._init_camera()

    def clamp(self, x, min_value, max_value):
        return max(min(x, max_value), min_value)
    
    def initialize_dof_states(self, 
                              num_dofs, dof_types, dof_positions, 
                              lower_limits, upper_limits, has_limits):
        # initialize default positions, limits, and speeds (make sure they are in reasonable ranges)
        defaults = np.zeros(num_dofs)
        speeds = np.zeros(num_dofs)
        for i in range(num_dofs):
            # dof_name = dof_names[i]
            # if dof_name not in arm_dof_names:
            #     stiffnesses[i] = 1e6
            #     dampings[i] = 1e6
            #     speeds[i] = 0.0
                
            if has_limits[i]:
                if dof_types[i] == gymapi.DOF_ROTATION:
                    lower_limits[i] = self.clamp(lower_limits[i], -math.pi, math.pi)
                    upper_limits[i] = self.clamp(upper_limits[i], -math.pi, math.pi)
                # make sure our default position is in range
                if lower_limits[i] > 0.0:
                    defaults[i] = lower_limits[i]
                elif upper_limits[i] < 0.0:
                    defaults[i] = upper_limits[i]
            else:
                # set reasonable animation limits for unlimited joints
                if dof_types[i] == gymapi.DOF_ROTATION:
                    # unlimited revolute joint
                    lower_limits[i] = -math.pi
                    upper_limits[i] = math.pi
                elif dof_types[i] == gymapi.DOF_TRANSLATION:
                    # unlimited prismatic joint
                    lower_limits[i] = -1.0
                    upper_limits[i] = 1.0
            
            # set DOF position to default
            dof_positions[i] = defaults[i]

            # set speed depending on DOF type and range of motion
            if dof_types[i] == gymapi.DOF_ROTATION:
                speeds[i] = self.speed_scale * self.clamp(2 * (upper_limits[i] - lower_limits[i]), 0.25 * math.pi, 3.0 * math.pi)
            else:
                speeds[i] = self.speed_scale * self.clamp(2 * (upper_limits[i] - lower_limits[i]), 0.1, 7.0)

        # for i in range(num_dofs):
        #     print(defaults[i], speeds[i])
    
    def generate_random_speed_for_ball(self, initial_speed_range, tilt_angle_range, tilt_z_angle_range):
        # ball_velocities = torch.zeros((num_envs, 2, 3))  # 速度 (vx, vy, vz)
        # # 初始化球的位置，给定初速度和倾斜角
        # speed_1 = random.uniform(*initial_speed_range)
        # # print("i=", i, "speed=", speed_1)
        # tilt_angle_1 = random.uniform(*tilt_angle_range)

        # velocity_1 = gymapi.Vec3(speed_1 * math.cos(math.radians(tilt_angle_1)), speed_1 * math.sin(math.radians(tilt_angle_1)), 0.0)
        # # ball_velocities_1 = torch.tensor([velocity_1.x, velocity_1.y, velocity_1.z  # 速度 (vx, vy, vz)
        # #                                     ])
        
        speed_2 = random.uniform(*initial_speed_range)
        tilt_angle_2 = random.uniform(*tilt_angle_range)
        # velocity_2 = gymapi.Vec3(speed_2 * math.cos(math.radians(tilt_angle_2)), speed_2 * math.sin(math.radians(tilt_angle_2)), 0.0)

        # newv
        tilt_angle_z_2 = random.uniform(*tilt_z_angle_range)

        # 将速度分解到 x, y, z 方向 newv
        velocity_2 = gymapi.Vec3(-speed_2 * math.cos(math.radians(tilt_angle_2)) * math.cos(math.radians(tilt_angle_z_2)), 
                             speed_2 * math.sin(math.radians(tilt_angle_2)) * math.cos(math.radians(tilt_angle_z_2)), 
                             speed_2 * math.sin(math.radians(tilt_angle_z_2)))
        
        # ball_velocities_2 = torch.tensor([velocity_2.x, velocity_2.y, velocity_2.z  # 速度 (vx, vy, vz)
        #                                         ])
        # angular_speed_x = random.uniform(*angular_speed_range)
        # angular_speed_y = random.uniform(*angular_speed_range)
        # angular_speed_z = random.uniform(*angular_speed_range)

        # angular_velocity = gymapi.Vec3(angular_speed_x, angular_speed_y, angular_speed_z)

        return velocity_2# , angular_velocity
    
    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2

        # 配置文件中有(task)
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.8

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        if self.terrain is None:
            self._create_ground_plane()
        else:
            self._create_trimesh()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        #TODO Randomization
        # if self.randomize:
        #     self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = -0.21  # 将地面高度设置为-0.21 3/11
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_trimesh(self):
        horizontal_scale = self.grid_size = self.cfg['env']['plane']['horizontal_scale'] # [m]
        if self.is_g1:
            vertical_scale = 0.75
        else:
            vertical_scale = 1.0  # [m]
        heightfield = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.terrain)).T
        self.heightfield = to_torch(heightfield, device=self.device, dtype=torch.float32)
        vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale,
                                                             vertical_scale=vertical_scale, slope_threshold=0.5)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = vertices.shape[0]
        tm_params.nb_triangles = triangles.shape[0]

        tm_params.transform.p.x = self.trans_x = self.cfg['env']['plane']['transform_x']
        tm_params.transform.p.y = self.trans_y = self.cfg['env']['plane']['transform_y']
        tm_params.transform.p.z = 0.0
        self.gym.add_triangle_mesh(self.sim, vertices.flatten(), triangles.flatten(), tm_params)
        self._get_meshgrid()

    def _get_meshgrid(self):
        if self.is_g1:
            x_range, y_range = 0.6, 0.6
        else:
            x_range, y_range = 0.8, 0.8
        x_split, y_split = 15, 15
        xs = np.linspace(-x_range, x_range, x_split)
        ys = np.linspace(-y_range, y_range, y_split)
        x, y = np.meshgrid(xs, ys, indexing='xy')
        p = np.stack([x.flatten(), y.flatten(), np.zeros_like(x.flatten())], axis=1)
        p = np.stack([p] * self.num_envs, axis=0)
        self.meshgrid = to_torch(p, device=self.device, dtype=torch.float32)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        # asset_file = "mjcf/lafan_humanoid.xml"

        # humanoid_asset_file = self.cfg["env"]["asset"]["assetFileName"]

        # asset_path = os.path.join(asset_root, humanoid_asset_file)
        # asset_root = os.path.dirname(asset_path)
        # asset_file = os.path.basename(asset_path)

        # if 'g1' in asset_file.lower():
        #     self.is_g1 = True
        # else:
        #     self.is_g1 = False

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.use_mesh_materials = True

        ### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ###
        # asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        humanoid_asset = self.gym.load_asset(self.sim, '/home/mj/app/IsaacGymEnvs-main/assets/mjcf', 'g1_27dof.urdf', asset_options)
        # print("humanoid_asset=", humanoid_asset)
        # humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
       
        # no use _load_motion
        # self.dof_body_ids = self.cfg["env"]["asset"]["dofBodyIds"]
        # self.dof_offsets = self.cfg["env"]["asset"]["dofOffsets"]

        # rigid_body_names = [
        #     self.gym.get_asset_rigid_body_name(humanoid_asset, i) for i in range(40)
        # ]

        self.right_arm_dof_names = [
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
        ]

        #TODO: check if this is necessary
        # self.humanoid_assets = [humanoid_asset] * num_envs
        # sk_tree = SkeletonTree.from_mjcf(asset_file)
        # self.skeleton_trees = [sk_tree] * num_envs

        # # create force sensors at the feet
        # right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, self.cfg["env"]["footBodyNames"][1])
        # left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, self.cfg["env"]["footBodyNames"][0])
        # sensor_pose = gymapi.Transform()
        # self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        # self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)


        # self.body_states_id = to_torch(self.cfg["env"]["bodyStatesId"], dtype=torch.long, device=self.device)
        # self.feet_mask = torch.zeros_like(self.body_states_id, dtype=torch.bool, device=self.device)
        # if self.is_g1:
        #     self.feet_mask[[6, 12, 16, 22]] = True
        #     # self.feet_mask[[6, 12]] = True
        #     # self.feet_mask[[16, 22]] = True
        # else:
        #     self.feet_mask[[3, 7, 14, 18]] = True

        # self.torso_index = 0
        self.num_humanoid_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_humanoid_dofs = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_humanoid_joints = self.gym.get_asset_joint_count(humanoid_asset)

        # 这些是不与环境交互的不变量
        # get array of DOF properties
        dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
        dof_prop['driveMode'] = gymapi.DOF_MODE_POS
        if self.is_g1:
            self._set_pd_control(humanoid_asset)
            dof_prop['stiffness'] = self.p_gains
            dof_prop['damping'] = self.d_gains
            
        self.humanoid_dof_props = dof_prop

        self.humanoid_dof_states = np.zeros(self.num_humanoid_dofs, dtype=gymapi.DofState.dtype)

        # get list of DOF types
        self.humanoid_dof_types = [self.gym.get_asset_dof_type(humanoid_asset, i) for i in range(self.num_humanoid_dofs)]

        # get the position slice of the DOF state array
        self.humanoid_dof_positions = self.humanoid_dof_states['pos']

        # get the limit-related slices of the DOF properties array
        self.stiffnesses = self.humanoid_dof_props['stiffness'] # 刚度
        self.dampings = self.humanoid_dof_props['damping'] # 阻尼
        self.armatures = self.humanoid_dof_props['armature']
        self.has_limits = self.humanoid_dof_props['hasLimits']
        self.lower_limits = self.humanoid_dof_props['lower']
        self.upper_limits = self.humanoid_dof_props['upper']
        
        # start_pose = gymapi.Transform()
        # start_pose.p = gymapi.Vec3(*get_axis_params(1.14, self.up_axis_idx))
        # start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)
        # pingpong table asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.use_mesh_materials = True
        pingpong_table_asset = self.gym.load_asset(self.sim, '/home/mj/app/IsaacGymEnvs-main/assets/urdf', 'pingpong_table.urdf', asset_options)
        
        # pingpong ball asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False  
        asset_options.use_mesh_materials = True
        pingpong_ball_asset = self.gym.load_asset(self.sim, '/home/mj/app/IsaacGymEnvs-main/assets/urdf', 'small_ball.urdf', asset_options)

        self.actor_handles = []
        self.actor_indices = []
        self.dof_indices = []
        
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            # create env instance
            # the order is: humanoid1, table, ball2
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            self.envs.append(env_ptr)
            
            # add actor 
            # humanoid1_handle
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            # pose.r = gymapi.Quat(0.0, 0.0, -0.2588, 0.9659) # -30

            # param6: bit collision
            self.humanoid1_handle = self.gym.create_actor(env_ptr, humanoid_asset, pose, "humanoid1", i, 0)
            humanoid1_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, self.humanoid1_handle)
            # print("len(humanoid1_shape_props)=", len(humanoid1_shape_props)) # len=7
            for j in range(len(humanoid1_shape_props)): # len=7, only collision rigid
                humanoid1_shape_props[j].restitution = 0.6
                humanoid1_shape_props[j].friction = 0.5
                humanoid1_shape_props[j].rolling_friction = 0.5
            self.gym.set_actor_rigid_shape_properties(env_ptr, self.humanoid1_handle, humanoid1_shape_props)
            # 跟我原先的代码比新增的
            self.gym.enable_actor_dof_force_sensors(env_ptr, self.humanoid1_handle)
            
            self.actor_handles.append(self.humanoid1_handle)
            humanoid1_idx = self.gym.get_actor_index(env_ptr, self.humanoid1_handle, gymapi.DOMAIN_SIM)
            # print("humanoid1_idx=", humanoid1_idx)
            self.actor_indices.append(humanoid1_idx)
            self.dof_indices.append(humanoid1_idx)

            # self.initialize_dof_states(self.num_humanoid_dofs, 
            #                            self.humanoid_dof_types, self.humanoid_dof_positions, 
            #                            self.lower_limits, self.upper_limits, self.has_limits)
            self.gym.set_actor_dof_states(env_ptr, self.humanoid1_handle, self.humanoid_dof_states, gymapi.STATE_ALL)
            self.gym.set_actor_dof_properties(env_ptr, self.humanoid1_handle, dof_prop)

            # # humanoid2_handle
            # pose = gymapi.Transform()
            # pose.p = gymapi.Vec3(3.5, 0.0, 1.0)
            # pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

            # self.humanoid2_handle = self.gym.create_actor(env_ptr, humanoid_asset, pose, "humanoid2", i, 0)
            # humanoid2_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, self.humanoid2_handle)
            # for j in range(len(humanoid2_shape_props)):
            #     humanoid2_shape_props[j].restitution = 0.6
            #     humanoid2_shape_props[j].friction = 0.5
            #     humanoid2_shape_props[j].rolling_friction = 0.5
            # self.gym.set_actor_rigid_shape_properties(env_ptr, self.humanoid2_handle, humanoid2_shape_props)
            # self.gym.enable_actor_dof_force_sensors(env_ptr, self.humanoid2_handle)

            # self.actor_handles.append(self.humanoid2_handle)
            # self.gym.set_actor_dof_states(env_ptr, self.humanoid2_handle, self.humanoid_dof_states, gymapi.STATE_ALL)
            # self.gym.set_actor_dof_properties(env_ptr, self.humanoid2_handle, dof_prop)

            # humanoid2_idx = self.gym.get_actor_index(env_ptr, self.humanoid2_handle, gymapi.DOMAIN_SIM)
            # # print("humanoid2_idx=", humanoid2_idx)
            # self.actor_indices.append(humanoid2_idx)
            # self.dof_indices.append(humanoid2_idx)

            # table_handle
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(1.75, 0.0, 0.0)
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            self.table_handle = self.gym.create_actor(env_ptr, pingpong_table_asset, pose, "pingpong_table", i, 0)
            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, self.table_handle)
            table_shape_props[0].restitution = 1.5
            table_shape_props[0].friction = 0.2
            table_shape_props[0].rolling_friction = 0.2
            self.gym.set_actor_rigid_shape_properties(env_ptr, self.table_handle, table_shape_props)
            self.actor_handles.append(self.table_handle)

            table_idx = self.gym.get_actor_index(env_ptr, self.table_handle, gymapi.DOMAIN_SIM)
            self.actor_indices.append(table_idx)

            # generate random speed and tilt angle for each ball
            # velocity_1, velocity_2 = self.generate_random_speed_for_ball(self.initial_speed_range, self.tilt_angle_range)
            velocity_2 = self.generate_random_speed_for_ball(self.initial_speed_range, self.tilt_angle_range, self.tilt_z_angle_range)

            # # ball1_handle
            # name = 'pingpong_ball_1'.format(i)
            # pose = gymapi.Transform()
            # pose.p = gymapi.Vec3(0.4, 0.28, 1.3)
            # pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0) # 没有旋转

            # self.ball1_handle = self.gym.create_actor(env_ptr, pingpong_ball_asset, pose, name, i, 0)   
            # # set restitution coefficient
            # ball1_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, self.ball1_handle)
            # ball1_shape_props[0].restitution = 0.9
            # ball1_shape_props[0].friction = 0.2
            # ball1_shape_props[0].rolling_friction = 0.2
            # self.gym.set_actor_rigid_shape_properties(env_ptr, self.ball1_handle, ball1_shape_props)

            # self.actor_handles.append(self.ball1_handle)
            # ball1_idx = self.gym.get_actor_index(env_ptr, self.ball1_handle, gymapi.DOMAIN_SIM)
            # # print("ball1_idx=", ball1_idx)
            # self.actor_indices.append(ball1_idx)

            # # apply linear velocity to ball 1
            # self.gym.set_rigid_linear_velocity(env_ptr, 
            #                             self.gym.get_rigid_handle(env_ptr, 
            #                                                       name, 
            #                                                       self.gym.get_actor_rigid_body_names(env_ptr, self.ball1_handle)[0]), 
            #                             velocity_1)

            # ball2_handle
            name = 'pingpong_ball_2'.format(i)
            pose = gymapi.Transform()
            pose_p_y= random.uniform(*self.initial_pos_y_range)
            pose_p_z = random.uniform(*self.initial_pos_z_range)
            pose.p = gymapi.Vec3(2.9, pose_p_y, pose_p_z)
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0) # 没有旋转

            self.ball2_handle = self.gym.create_actor(env_ptr, pingpong_ball_asset, pose, name, i, 0)
            # set restitution coefficient
            ball2_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, self.ball2_handle)
            ball2_shape_props[0].restitution = 1.5
            ball2_shape_props[0].friction = 0.2
            ball2_shape_props[0].rolling_friction = 0.2
            self.gym.set_actor_rigid_shape_properties(env_ptr, self.ball2_handle, ball2_shape_props)
            self.actor_handles.append(self.ball2_handle)

            ball2_idx = self.gym.get_actor_index(env_ptr, self.ball2_handle, gymapi.DOMAIN_SIM)
            # print("ball2_idx=", ball2_idx)
            self.actor_indices.append(ball2_idx)

            # apply linear velocity to ball 2
            self.gym.set_rigid_linear_velocity(env_ptr, 
                                        self.gym.get_rigid_handle(env_ptr, 
                                                                  name, 
                                                                  self.gym.get_actor_rigid_body_names(env_ptr, self.ball2_handle)[0]), 
                                        velocity_2)
            # self.gym.set_rigid_angular_velocity(env_ptr, 
            #                             self.gym.get_rigid_handle(env_ptr, 
            #                                                       name, 
            #                                                       self.gym.get_actor_rigid_body_names(env_ptr, self.ball2_handle)[0]), 
            #                             angular_velocity_2)
        
        self.actor_indices = to_torch(self.actor_indices, dtype=torch.long, device=self.device)
        self.dof_indices = to_torch(self.dof_indices, dtype=torch.long, device=self.device)

        # dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)
        for j in range(self.num_humanoid_dofs):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        # if self.is_g1:
        #     self._pd_action_offset = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.175, 0., 1.5, 0., 0., 0., 0., -0.175, 0.0, 1.5, 0., 0., 0.])
        #     self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        # self._pd_action_offset_2_humanoid = torch.cat((self._pd_action_offset, self._pd_action_offset), dim=0)
        
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)
        # self._pd_action_scale_2_humanoid = torch.cat((self._pd_action_scale, self._pd_action_scale), dim=0)

        self._key_body_ids = self._build_key_body_ids_tensor(env_ptr, self.humanoid1_handle)
        self.extremities = to_torch([5, 8], device=self.device, dtype=torch.long)

        # 现在self.debug_viz是false
        if self.debug_viz:
            sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
            sphere_pose = gymapi.Transform(r=sphere_rot)
            self.sphere_geom = gymutil.WireframeSphereGeometry(0.03, 3, 3, sphere_pose, color=(1, 0, 0))

            self.marker_trans = gymapi.Transform()
            self.marker_trans.p = gymapi.Vec3(0., 0., 0.)
            self.marker_trans.r = gymapi.Quat(0., 0., 0., 1.)

        # self._dof_body_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        # self._dof_offsets = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63]
        # self._dof_obs_size = 13 + 126 + 63 + 12
        # self._num_actions = 63

    # 参数别动！
    def _set_pd_control(self, humanoid_asset):
        self.p_gains = np.array([
                # Pelvis is not controlled
                80.0, 80.0, 80.0,  # Left hip (pitch, roll, yaw)
                160.0,  # Left knee
                20.0, 20.0,  # Left ankle (pitch, roll)
                80.0, 80.0, 80.0,  # Right hip (pitch, roll, yaw)
                160.0,  # Right knee
                80.0, 20.0,  # Right ankle (pitch, roll)
                80.0, 80.0, 80.0, # Torso
                20.0, 20.0, 20.0,  # Left shoulder (pitch, roll, yaw)
                20.0,  # Left elbow pitch
                20.0, 5.0, 5.0,  # Left elbow roll
                20.0, 20.0, # 20.0,  # Left shoulder (pitch, roll, yaw)
                # 20.0,  # Left elbow pitch
                20.0, 5.0, 5.0  # Left elbow roll
            ])

        self.d_gains = self.p_gains / 40.

    # def _load_motion(self, motion_file):
    #     self._motion_lib = MotionLib(motion_file=motion_file,
    #                                  num_dofs=self.num_dof,
    #                                  key_body_ids=self._key_body_ids.cpu().numpy(),
    #                                  dof_body_ids=self.dof_body_ids,
    #                                  dof_offsets=self.dof_offsets,
    #                                  is_train=self.is_train,
    #                                  # is_train=False,
    #                                  device=self.device)

    def _build_key_body_ids_tensor(self, env_ptr, actor_handle):
        if 'all' in self.key_body_names:
            return torch.arange(self.num_humanoid_bodies, device=self.device, dtype=torch.long)
        # 现在cfg中是no
        if 'no' in self.key_body_names:
            return torch.tensor([0], device=self.device, dtype=torch.long)

        body_ids = []
        for body_name in self.key_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def compute_reward(self, actions):
        # self.rew_buf[:], self.reset_buf[:] = compute_pingpong_reward_only_paddle(
        #     self.humanoid1_root_states,
        #     self.humanoid1_paddle_rb_states,
        #     self.pre_ball2_root_states,
        #     self.ball2_root_states,
        #     self.dof_force_tensor,
        #     self.dof_vel,
        #     self.reset_buf,
        #     self.progress_buf,
        #     self.max_episode_length,
        #     self.alpha,
        #     self.power_coefficient,
        #     self.penalty,
        #     self.paddle_condition_calculated,
        #     self.missed_ball_calculated
        # )

        self.rew_buf[:], self.reset_buf[:], self.paddle_condition_calculated[:], self.hit_table_calculated[:], self.die_penalty_calculated[:], self.humanoid_die_calculated[:], self.closer_to_paddle_count[:], self.hit_paddle_count[:], self.cross_net_count[:], self.hit_table_count[:], self.fall_down_count[:] = compute_pingpong_reward_nv(
            self.humanoid1_root_states,
            self.humanoid1_pelvis_rb_states,
            self.humanoid1_paddle_rb_states,
            self.pre_ball2_root_states,
            self.ball2_root_states,
            self.dof_force_tensor,
            self.dof_vel,
            self.reset_buf,
            self.progress_buf,
            self.max_episode_length,
            self.alpha,
            self.power_coefficient,
            self.paddle_condition_calculated,
            self.hit_paddle_reward,
            self.miss_paddle_penalty_coefficient,
            self.cross_net_reward_float,
            self.hit_table_reward,
            self.not_hit_table_penalty,
            self.hit_table_calculated,
            self.die_penalty_float,
            self.die_penalty_calculated,
            self.humanoid_die_calculated,
            self.closer_to_paddle_count,
            self.hit_paddle_count,
            self.cross_net_count,
            self.hit_table_count,
            self.fall_down_count,
            self.initial_dof_pos,
            self.dof_pos,
            self.initial_dof_vel,
            self.body_states,
            self.initial_body_states,
            self.body_balance_states_id,
            self.is_g1,
            self.is_train
        )

        # self.reset_buf = torch.logical_or(self.reset_buf, self.reset_buf_force)

        if self.num_steps % 32 == 0: # 现在episode_length和horizen_length是32*5=160
            reward_mean = torch.mean(self.rew_buf).item()
            progress_mean = torch.mean(self.progress_buf.float()).item()
            print(f"{reward_mean:.4f}    {progress_mean:.4f}", flush=True)
            # self.extras['reward_mean'] = reward_mean
            # self.extras['progress_mean'] = progress_mean

    def compute_observations(self):
        # self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)
        # self.gym.refresh_dof_force_tensor(self.sim)

        # self_obs = compute_humanoid_observations(self.body_states, self.dof_pos, self.dof_vel, self.body_states_id)
        # # dir_obs = compute_direction_observations(self.body_states, self.dof_pos, self.dof_vel, self.ref_body_states_fut, self.ref_dof_pos, self.ref_dof_vel, self.body_states_id, targets=self.targets)
        # height_obs = compute_heightmap_observations(self.body_states, self.body_states_id, self.meshgrid, self.heightfield, self.trans_x, self.trans_y, self.grid_size)
        # # self.obs_buf[:] = torch.cat([self_obs, dir_obs, height_obs], dim=-1)

        # imi_obs = compute_imitation_observations(self.body_states, self.dof_pos, self.dof_vel, self.ref_body_states, self.ref_dof_pos, self.ref_dof_vel, self.body_states_id)
        # imi_obs_fut = compute_imitation_observations(self.body_states, self.dof_pos, self.dof_vel, self.ref_body_states_fut, self.ref_dof_pos_fut, self.ref_dof_vel_fut, self.body_states_id)
        # # self_obs = compute_humanoid_observations(self.body_states, self.dof_pos, self.dof_vel, self.body_states_id)
        # self.obs_buf[:] = torch.cat([imi_obs, imi_obs_fut, height_obs, self_obs], dim=-1)

        pingpong_obs = compute_pingpong_observations(
            self.body_states,
            self.body_states_id,
            self.ball2_root_states
        )

        humanoid_obs = compute_humanoid_observations(
            self.body_states, 
            self.dof_pos, 
            self.dof_vel, 
            self.body_states_id
        )

        imi_obs = compute_imitation_observations(
            self.body_states, 
            self.initial_body_states, 
            self.initial_dof_pos, 
            self.initial_dof_vel, 
            self.body_balance_states_id)
        
        self.obs_buf[:] = torch.cat([humanoid_obs, pingpong_obs, imi_obs], dim=-1)

    def refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def reset_idx(self, env_ids):
        # self._reset_actors(env_ids)
        # print("env_ids which need reset: ", env_ids)

        # # 输出计数器结果
        # print("the sum of the envs which are closer to the paddle:", torch.sum(self.closer_to_paddle_count[:]).item())
        # print("the sum of the envs which hit the paddle:", torch.sum(self.hit_paddle_count[:]).item())
        # print("the sum of the envs which cross the net:", torch.sum(self.cross_net_count[:]).item())
        # print("the sum of the envs which hit the table:", torch.sum(self.hit_table_count[:]).item())

        self._reset_idx(env_ids)

        # # 重置计数器
        # self.closer_to_paddle_count.fill_(0)
        # self.hit_paddle_count.fill_(0)
        # self.cross_net_count.fill_(0)
        # self.hit_table_count.fill_(0)

        return

    # no use
    def _reset_actors(self, env_ids):
        # self.dof_pos[env_ids] = self.initial_dof_pos[env_ids]
        # self.dof_vel[env_ids] = self.initial_dof_vel[env_ids]

        self.dof_pos[env_ids] = self.ref_dof_pos[env_ids]
        self.dof_vel[env_ids] = self.ref_dof_vel[env_ids]

        # self.ref_body_states[env_ids][:] = 0.

        # self.ref_root_states[env_ids, 2] = self.ref_root_states[env_ids, 2] + 0.1
        # self.ref_root_states[env_ids, 7:] = 0.
        # self.ref_root_states[env_ids, 6] = 1.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                              gymtorch.unwrap_tensor(self.initial_root_states),
        #                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.ref_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.progress_buf[env_ids] = 0
        # self.reset_buf[env_ids] = 0
        # self._terminate_buf[env_ids] = 0
        return
    
    def _reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # (self.num_envs, actors_per_env, 13)
        self.vec_root_states[env_ids, :, 0:3] = self.initial_pos[env_ids, :]
        self.vec_root_states[env_ids, :, 3:7] = self.initial_rot[env_ids, :]
        self.vec_root_states[env_ids, :, 7:13] = torch.zeros_like(self.vec_root_states[env_ids, :, 7:13])

        for index, env_id in enumerate(env_ids):
            pose_p_y= random.uniform(*self.initial_pos_y_range)
            pose_p_z = random.uniform(*self.initial_pos_z_range)
            self.vec_root_states[env_id, 2, 1] = pose_p_y
            self.vec_root_states[env_id, 2, 2] = pose_p_z
            
            # 随机生成球的速度
            velocity_2 = self.generate_random_speed_for_ball(self.initial_speed_range, self.tilt_angle_range, self.tilt_z_angle_range)
            ball_velocities_2 = torch.tensor([velocity_2.x, velocity_2.y, velocity_2.z  # 速度 (vx, vy, vz)
                                                ], device=self.device)
            # print("ball_velocities_2.x: ", ball_velocities_2)
            # ball_angular_velocity_2 = torch.tensor([angular_velocity_2.x, angular_velocity_2.y, angular_velocity_2.z], device=self.device)
            self.vec_root_states[env_id, 2, 7:10] = ball_velocities_2
            # self.vec_root_states[env_id, 2, 10:13] = ball_angular_velocity_2

        self.vec_dof_states[env_ids, :, :] = self.initial_dof_states[env_ids, :, :]

        # env_ids_int32 = env_ids.to(dtype=torch.int32)
        actor_indices = self.actor_indices.view(self.num_envs, self.actors_per_env)[env_ids].flatten().to(torch.int32)
        dof_indices = self.dof_indices.view(self.num_envs, 1)[env_ids].flatten().to(torch.int32)
        # root_state_tensor, the actor which need to reset index, the num
        reset1 = self.gym.set_actor_root_state_tensor_indexed(self.sim, 
                                                     gymtorch.unwrap_tensor(self.root_states), 
                                                     gymtorch.unwrap_tensor(actor_indices), len(env_ids)*3)
        
        # set dof to the original position and velocity
        reset2 = self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_states),
                                              gymtorch.unwrap_tensor(dof_indices), len(env_ids)*1)
        
        ### !!!!!!!!!!!!!!!!!! ###
        # to_target = self.targets[env_ids] - self.initial_root_states[env_ids, 0:3]
        # to_target[:, self.up_axis_idx] = 0

        # self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        # self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        # self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        # self.reward_calculated[env_ids] = False
        self.paddle_condition_calculated[env_ids] = False
        self.die_penalty_calculated[env_ids] = False
        self.humanoid_die_calculated[env_ids] = False
        self.hit_table_calculated[env_ids] = False

        # # 计数器清零
        # self.hit_paddle_count[env_ids] = 0
        # self.cross_net_count[env_ids] = 0
        # self.hit_table_count[env_ids] = 0

        # self.missed_ball_calculated[env_ids] = False
        # self.no_bounce_before_half_mask[env_ids] = True
        # self.net_condition_calculated[env_ids] = False
        return

    # def init_obs_demo(self):
    #     self.target_motion_ids = self._motion_lib.sample_motions(self.num_envs)
    #     self.target_motion_times = self._motion_lib.sample_time(self.target_motion_ids)
    #     self.target_motion_times_fut = self.target_motion_times + self.dt * self.future_dt

    #     #TODO Fix motion time
    #     # self.target_motion_times[:] = 80.

    #     motion_res = self._motion_lib.get_motion_state(self.target_motion_ids, self.target_motion_times)
    #     ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, ref_root_ang_vel, ref_dof_vel, ref_rb_pos, ref_rb_rot, \
    #     ref_body_vel, ref_body_ang_vel = \
    #         motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], \
    #         motion_res["root_ang_vel"], motion_res["dof_vel"], motion_res["rg_pos"], motion_res["rb_rot"], \
    #         motion_res["body_vel"], motion_res["body_ang_vel"]

    #     # ref_body_vel[:] = 0.
    #     # ref_body_ang_vel[:] = 0.
    #     # ref_dof_vel[:] = 0.

    #     self.ref_body_states = torch.cat([ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel], dim=-1)
    #     self.ref_root_states = self.ref_body_states[:, 0, :].clone()
    #     self.ref_dof_pos = ref_dof_pos.clone()
    #     self.ref_dof_vel = ref_dof_vel.clone()

    #     motion_res = self._motion_lib.get_motion_state(self.target_motion_ids, self.target_motion_times_fut)
    #     ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, ref_root_ang_vel, ref_dof_vel, ref_rb_pos, ref_rb_rot, \
    #     ref_body_vel, ref_body_ang_vel = \
    #         motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], \
    #         motion_res["root_ang_vel"], motion_res["dof_vel"], motion_res["rg_pos"], motion_res["rb_rot"], \
    #         motion_res["body_vel"], motion_res["body_ang_vel"]

    #     self.ref_body_states_fut = torch.cat([ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel], dim=-1)
    #     self.ref_root_states_fut = self.ref_body_states[:, 0, :].clone()
    #     self.ref_dof_pos_fut = ref_dof_pos.clone()
    #     self.ref_dof_vel_fut = ref_dof_vel.clone()


    #     self.compute_observations()

    #     # root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
    #     #     = self._motion_lib.get_motion_state(self.target_motion_ids, self.target_motion_times)
    #     # self.target_states = key_pos
    
    # 
    # def update_obs_demo(self):
    #     # reset_buf_np = self.reset_buf.cpu().numpy().astype(np.bool)
    #     # self.target_motion_times[reset_buf_np] = self.target_motion_times[reset_buf_np] // 2. * 2.
    #     # self.target_motion_ids = self._motion_lib.sample_motions(self.num_envs)

    #     # if np.random.rand() < 0.01:
    #     #     print('STEP FORWARD')
    #     #     self.target_motion_times = self.target_motion_times + 0.1

    #     self.target_motion_times = self.target_motion_times + self.dt
    #     # # TODO Random time step
    #     if self.num_steps // 32 > 3000 and self.is_train:
    #         randnum = np.random.rand(self.num_envs) < 0.02
    #         if (randnum).sum() > 0:
    #             randsize = 0.2
    #             self.target_motion_times[randnum] = self.target_motion_times[randnum] + np.random.uniform(-randsize, randsize, (randnum).sum())

    #     # self.target_motion_times[:] = 20.
    #     finished = self.target_motion_times > self._motion_lib._motion_lengths[self.target_motion_ids]
    #     self.reset_buf[finished] = 1
    #     # time_new = self._motion_lib.sample_time(self.target_motion_ids)
    #     # self.target_motion_times[finished] = time_new[finished]
    #     self.target_motion_times[finished] = np.random.rand() * self.dt

    #     motion_res = self._motion_lib.get_motion_state(self.target_motion_ids, self.target_motion_times)
    #     ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, ref_root_ang_vel, ref_dof_vel, ref_rb_pos, ref_rb_rot, \
    #     ref_body_vel, ref_body_ang_vel = \
    #         motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], \
    #         motion_res["root_ang_vel"], motion_res["dof_vel"], motion_res["rg_pos"], motion_res["rb_rot"], \
    #         motion_res["body_vel"], motion_res["body_ang_vel"]

    #     self.ref_body_states = torch.cat([ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel], dim=-1)
    #     self.ref_root_states = self.ref_body_states[:, 0, :].clone()
    #     self.ref_dof_pos = ref_dof_pos.clone()
    #     self.ref_dof_vel = ref_dof_vel.clone()

    #     self.target_motion_times_fut = self.target_motion_times + self.dt * self.future_dt
    #     motion_res = self._motion_lib.get_motion_state(self.target_motion_ids, self.target_motion_times_fut)
    #     ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, ref_root_ang_vel, ref_dof_vel, ref_rb_pos, ref_rb_rot, \
    #     ref_body_vel, ref_body_ang_vel = \
    #         motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], \
    #         motion_res["root_ang_vel"], motion_res["dof_vel"], motion_res["rg_pos"], motion_res["rb_rot"], \
    #         motion_res["body_vel"], motion_res["body_ang_vel"]

    #     self.ref_body_states_fut = torch.cat([ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel], dim=-1)
    #     self.ref_root_states_fut = self.ref_body_states[:, 0, :].clone()
    #     self.ref_dof_pos_fut = ref_dof_pos.clone()
    #     self.ref_dof_vel_fut = ref_dof_vel.clone()


    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        # print("actions:", self.actions)
        #TODO
        # ZERO OUT THE ACTIONS
        # self.actions[:] = 0.

        pd_tar = self._pd_action_offset + self._pd_action_scale * self.actions
        # pd_tar = self._pd_action_offset_2_humanoid + self._pd_action_scale_2_humanoid * self.actions

        # pd_tar = self.ref_dof_pos_fut + self._pd_action_scale * self.actions

        pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
        self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)

        # forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
        # force_tensor = gymtorch.unwrap_tensor(forces)
        # self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

        self.pre_ball2_root_states = self.ball2_root_states.clone()
        # print(self.dof_vel)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print("humnaoid1_pelvis height:", self.humanoid1_pelvis_rb_states[:, 2].item())
        

    def post_physics_step(self):
        self.progress_buf += 1
        # print("progress_buf: ", self.progress_buf)
        self.randomize_buf += 1

        self.refresh_sim_tensors()

        # self.post_ball2_root_states = self.ball2_root_states.clone()

        self.compute_reward(self.actions)  # R t

        # self.update_obs_demo()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            # 输出计数器结果
            print("the sum of the huamnoid which fall down:", torch.sum(self.fall_down_count[:]).item())
            print("the sum of the envs which are closer to the paddle:", torch.sum(self.closer_to_paddle_count[:]).item())
            print("the sum of the envs which hit the paddle:", torch.sum(self.hit_paddle_count[:]).item())
            print("the sum of the envs which cross the net:", torch.sum(self.cross_net_count[:]).item())
            print("the sum of the envs which hit the table:", torch.sum(self.hit_table_count[:]).item())
            self.reset_idx(env_ids)
            # 重置计数器
            self.closer_to_paddle_count.fill_(0)
            self.hit_paddle_count.fill_(0)
            self.cross_net_count.fill_(0)
            self.hit_table_count.fill_(0)
            self.fall_down_count.fill_(0)

        self.reset_buf_force[:] = 0

        self.compute_observations()  # S t+1

        # self.reset_buf[self.progress_buf >= self.max_episode_length] = 1

        if not self.headless:
            if self.enable_viewer_sync:
                self._update_camera()

                # if self.debug_viz:
                #     self._debug_visualization()

        self.num_steps += 1

        torch.cuda.empty_cache()

        # self.reset_buf[:] = 1

    # def _debug_visualization(self):
    #     self.gym.clear_lines(self.viewer)

    #     i = self.character_idx
    #     for j in range(len(self.body_states_id)):
    #         marker_trans = gymapi.Transform()
    #         marker_trans.p = gymapi.Vec3(self.ref_body_states[i, j, 0].item(),
    #                                             self.ref_body_states[i, j, 1].item(),
    #                                             self.ref_body_states[i, j, 2].item())
    #         marker_trans.r = gymapi.Quat(0., 0., 0., 1.)
    #         gymutil.draw_lines(self.sphere_geom, self.gym, self.viewer, self.envs[i], marker_trans)

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self.root_states[0, 0:3].cpu().numpy()

        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0],
                              self._cam_prev_char_pos[1] - 3.0,
                              1.0)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0],
                                 self._cam_prev_char_pos[1],
                                 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self.root_states[self.character_idx, 0:3].cpu().numpy()

        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0],
                                  char_root_pos[1] + cam_delta[1],
                                  cam_pos[2])

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return



#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_gradient_penalty(ball_positions, ball_velocity_x, hit_table_reward, not_hit_table_penalty, hit_table_calculated, hit_table_count, humanoid_die_calculated):
    # type: (torch.Tensor, torch.Tensor, float, float, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    计算有梯度的 penalty。

    参数:
    - ball_positions: 球的 x, y, z位置, 形状为 [num_envs, 3]。

    返回:
    - penalty: 每个球的 penalty，形状为 [num_envs]。
    """
    # 提取 x 和 y 坐标
    x = ball_positions[..., 0]  # x 坐标
    y = ball_positions[..., 1]  # y 坐标
    z = ball_positions[..., 2]  # z 坐标

    # 高度范围
    z_min, z_max = 0.82, 0.83  # z 范围
    z_in_range = (z >= z_min) & (z <= z_max) & (ball_velocity_x > 0)  # 仅当 z 坐标在 [0.79, 0.83] 之间且 x 方向速度大于 0 时

    # 定义正方形的范围
    x_min, x_max = 1.9, 3.1  # x 范围
    y_min, y_max = -0.6, 0.6  # y 范围

    # 定义中心点
    center_x, center_y = 2.5, 0.0

    # 计算每个球到中心点的欧几里得距离
    distance = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # 判断球是否在正方形范围内
    in_range = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)

    # 更新 hit_table_count
    hit_table_count = torch.where(
        z_in_range & in_range,  # 条件：满足 hit_table 条件
        torch.ones_like(hit_table_count),  # 满足条件时，设置为 1
        hit_table_count  # 不满足条件时，保持原值
    )

    # 设置 penalty
    penalty_and_reward = torch.zeros_like(distance)
    penalty_and_reward = torch.where(
        z_in_range & (~hit_table_calculated) & (~humanoid_die_calculated),  # 仅当 z 坐标在z in range之间且没有计算过时候计算 penalty
        torch.where(
            in_range,  # 如果在正方形范围内
            hit_table_reward,  # 给常数reward
            not_hit_table_penalty * distance  # 如果超出范围，penalty 与距离成正比
        ),
        penalty_and_reward  # 如果 z 坐标不在范围内，penalty = 0
    )
    hit_table_calculated |= z_in_range # 更新标志张量
    # if(penalty < 0).any(): 
    #     print("not the corret position to hit the table", (penalty<0).sum())

    return penalty_and_reward, hit_table_calculated, hit_table_count

DOF_Names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 
             'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 
             'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 
             'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 
             'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint', 'left_shoulder_pitch_joint', 
             'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 
             'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 
             'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 
             'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'] # 27

def compute_imitation_reward(dof_pos, rest_dof_pos, dof_vel, rest_dof_vel, body_states, initial_body_states, body_balance_states_id=None, is_g1=False, is_train=True):
    """
    计算分层次的 reference motion reward。
    - 前 20 个 DOF 的奖励系数较大，鼓励完全按照 reference motion 不动。
    - 后 5 个 DOF 的奖励系数较小，允许有一定的灵活性。

    参数:
    - dof_pos: 当前 DOF 位置，形状为 [B, num_dofs]。
    - rest_dof_pos: 参考 DOF 位置，形状为 [B, num_dofs]。
    - w_dof_pos: 奖励权重。
    - k_dof_pos: 奖励系数。
    - body_states: 身体状态张量，形状为 [B, J, _]。
    - is_g1: 是否为 G1 环境（可选）。

    返回:
    - dof_pos_reward: 分层次的 DOF 位置奖励，形状为 [B]。
    """
    if is_g1:
        k_pos, k_vel, k_dof_pos, k_dof_vel = 50., 4.0, 5.0, 0.05
        w_pos, w_vel, w_dof_pos, w_dof_vel = 0.4, 0.2, 0.2, 0.2
    else:
        k_pos, k_vel, k_dof_pos, k_dof_vel = 30., 1.0, 2.5, 0.5
        w_pos, w_vel, w_dof_pos, w_dof_vel = 0.4, 0.2, 0.2, 0.2

    B, J, _ = body_states.shape

    # body position reward
    if body_balance_states_id is not None:
        body_pos = body_states[:, body_balance_states_id, 0:3] 
        body_vel = body_states[:, body_balance_states_id, 7:10]
        ref_body_pos = initial_body_states[:, body_balance_states_id, 0:3]
        ref_body_vel = initial_body_states[:, body_balance_states_id, 7:10]
    # print("init body states:", initial_body_states.shape, initial_body_states) # (envs, 42, 13)
    # print("body_pos:", initial_body_states)
    # print("!")
    diff_global_body_pos = ref_body_pos - body_pos
    diff_body_pos_dist = (diff_global_body_pos ** 2).mean(dim=-1).mean(dim=-1)
    r_body_pos = torch.exp(-k_pos * diff_body_pos_dist)

    # body linear velocity reward
    diff_global_vel = ref_body_vel - body_vel
    diff_global_vel_dist = (diff_global_vel**2).mean(dim=-1).mean(dim=-1)
    r_body_vel = torch.exp(-k_vel * diff_global_vel_dist)

    # print("dof pos:", dof_pos)

    # 计算 DOF 位置的差异
    if is_g1:
        diff_dof_pos = (rest_dof_pos - dof_pos) ** 2  # 形状为 [B, num_dofs]
    else:
        dof_pos_quat = exp_map_to_quat(dof_pos.view(-1, 3))  # 将 DOF 位置转换为四元数
        ref_dof_pos_quat = exp_map_to_quat(rest_dof_pos.view(-1, 3))  # 将参考 DOF 位置转换为四元数
        diff_dof_pos_quat = quat_mul(ref_dof_pos_quat, quat_conjugate(dof_pos_quat))  # 计算四元数差异
        diff_dof_rot_angle = quat_to_angle_axis(diff_dof_pos_quat)[0].view(B, -1)  # 将四元数差异转换为旋转角度
        diff_dof_pos = diff_dof_rot_angle ** 2  # 形状为 [B, num_dofs]

    # 分层次计算pos奖励
    # 前 22 个 DOF 的系数较大（非右胳膊）
    w_dof_pos_first_22 = w_dof_pos * 50.0  # 前 22 个 DOF 的奖励系数较大
    k_dof_pos_first_22 = k_dof_pos * 500.0  # 前 22 个 DOF 的奖励系数较大

    # 计算前 22 个 DOF 的奖励
    diff_dof_pos_first_22 = diff_dof_pos[..., :22]  # 前 22 个 DOF 的差异
    r_dof_pos_first_22 = torch.exp(-k_dof_pos_first_22 * diff_dof_pos_first_22.mean(dim=-1)) # 前 22 个 DOF 的奖励
    dof_pos_reward_first_22 = w_dof_pos_first_22 * r_dof_pos_first_22 # 前 22 个 DOF 的奖励

    # 计算后 5 个 DOF 的奖励
    diff_dof_pos_last_5 = diff_dof_pos[..., 22:]  # 后 5 个 DOF 的差异
    r_dof_pos_last_5 = torch.exp(-k_dof_pos * diff_dof_pos_last_5.mean(dim=-1))  # 后 5 个 DOF 的奖励
    dof_pos_reward_last_5 = w_dof_pos * r_dof_pos_last_5  # 后 5 个 DOF 的奖励

    ### ----------------------------------------------------- ###
    # 计算dof速度的差异
    rest_dof_vel_no_right_arm = rest_dof_vel[..., :22]  # 参考 DOF 速度，不包括右臂
    dof_vel_no_right_arm = dof_vel[..., :22]  # 当前 DOF 速度，不包括右臂
    # diff_dof_vel = ((ref_dof_vel - dof_vel)**2).mean(dim=-1)
    if is_g1:
        diff_dof_vel = ((rest_dof_vel_no_right_arm - dof_vel_no_right_arm)**2).mean(dim=-1)
    else:
        dof_vel_quat = exp_map_to_quat(dof_vel_no_right_arm.view(-1, 3))
        ref_dof_vel_quat = exp_map_to_quat(rest_dof_vel_no_right_arm.view(-1, 3))
        diff_dof_vel_quat = quat_mul(ref_dof_vel_quat, quat_conjugate(dof_vel_quat))
        diff_dof_vel_angle = quat_to_angle_axis(diff_dof_vel_quat)[0].view(B, -1)
        diff_dof_vel = (diff_dof_vel_angle ** 2).mean(dim=-1)

    r_dof_vel = torch.exp(-k_dof_vel * diff_dof_vel) # .mean(dim=-1)

    # 综合奖励
    ref_reward = dof_pos_reward_first_22 + dof_pos_reward_last_5 + w_dof_vel * r_dof_vel + w_pos * r_body_pos + w_vel * r_body_vel # 形状为 [B]
    
    # 判断是否fallen
    if is_train:
        if is_g1:
            termination_distance = 0.32 # 0.3
        else:
            termination_distance = 0.4 
    else:
        termination_distance = 1e6
    # diff_body_pos = torch.norm(body_pos - ref_body_pos, dim=-1)  # 计算身体位置与参考位置的欧几里得距离
    has_fallen = torch.any(torch.norm(body_pos - ref_body_pos, dim=-1).mean(dim=-1, keepdim=True) > termination_distance, dim=-1)
    # has_fallen = torch.any(diff_body_pos > termination_distance, dim=-1)  # 判断是否 fallen

    death_cost = -50.0
    ref_reward = torch.where(has_fallen, torch.ones_like(ref_reward) * death_cost, ref_reward)

    return ref_reward, has_fallen

# def compute_rest_pose_reward(dof_pos, rest_dof_pos, w_dof_pos, k_dof_pos, body_states, is_g1=False):
#     print(dof_pos)
#     B, J, _ = body_states.shape
#     if is_g1:
#         diff_dof_pos = ((rest_dof_pos - dof_pos) ** 2).mean(dim=-1)
#     else:
#         dof_pos_quat = exp_map_to_quat(dof_pos.view(-1, 3))
#         ref_dof_pos_quat = exp_map_to_quat(rest_dof_pos.view(-1, 3))
#         diff_dof_pos_quat = quat_mul(ref_dof_pos_quat, quat_conjugate(dof_pos_quat))
#         diff_dof_rot_angle = quat_to_angle_axis(diff_dof_pos_quat)[0].view(B, -1)
#         diff_dof_pos = (diff_dof_rot_angle ** 2).mean(dim=-1)
#         # dof_pos_quat = quat_to_tan_norm(exp_map_to_quat(dof_pos.view(-1, 3))).reshape(B, J - 1, -1)
#         # ref_dof_pos_quat = quat_to_tan_norm(exp_map_to_quat(ref_dof_pos.view(-1, 3))).reshape(B, J - 1, -1)
#         # diff_dof_pos = ((ref_dof_pos_quat - dof_pos_quat) ** 2).mean(dim=-1).mean(dim=-1)
    
#     r_dof_pos = torch.exp(-k_dof_pos * diff_dof_pos)
#     dof_pos_reward = w_dof_pos * r_dof_pos

#     return dof_pos_reward

# @torch.jit.script
def compute_pingpong_reward_nv(
    humanoid1_root_states, #T
    humanoid1_pelvis_rb_states, #T
    humanoid1_paddle_rb_states, #T
    pre_ball2_root_states, #T
    ball2_root_states, #T
    dof_force_tensor, #T
    dof_vel, #T
    reset_buf, #T
    progress_buf, #T
    max_episode_length, #F
    alpha, #F
    power_coefficient, #F
    paddle_condition_calculated, #T
    hit_paddle_reward, #F
    miss_paddle_penalty_coefficient, #F
    cross_net_reward_float, #F
    hit_table_reward, #F
    not_hit_table_penalty, #F
    hit_table_calculated, #T
    die_penalty_float, #F
    die_penalty_calculated,  #T
    humanoid_die_calculated,  #T
    closer_to_paddle_count, #T
    hit_paddle_count, #T
    cross_net_count, #T
    hit_table_count,  #T
    fall_down_count, #T
    init_dof_pos, #T
    dof_pos, #T
    init_dof_vel, #T
    body_states, #T
    initial_body_states, #T
    body_balance_states_id=None, #T
    is_g1=False,
    is_train=True
    ):

    threshold = 0.78  # 乒乓球掉落高度阈值

    # angular_velocity = humanoid1_paddle_rb_states[..., 10:13]
    # angular_velocit_norm = torch.norm(angular_velocity, dim=-1)

    # compute the distance between the humanoid's right hand and the pingpong ball
    humanoid1_paddle_position = humanoid1_paddle_rb_states[..., 0:3] 

    ball2_root_state_position = ball2_root_states[..., 0:3]

    # ball velocity change
    pre_ball2_velocity_x = pre_ball2_root_states[..., 7]
    ball2_velocity_x = ball2_root_states[..., 7]

    velocity_reward = torch.zeros_like(ball2_velocity_x)
    pos_reward = torch.zeros_like(ball2_velocity_x)

    reward = torch.zeros_like(ball2_velocity_x)

    # ----------------------------------------------------------------------------------- #

    # 3/3 设置一个接近reset_pos的reward
    # 类似imitation reward 3/11继续完善
    ref_reward, has_fallen = compute_imitation_reward(dof_pos, init_dof_pos, dof_vel, init_dof_vel, body_states, initial_body_states, body_balance_states_id, is_g1, is_train)
    # print("has fallen: ", has_fallen)
    # height_penalty = torch.zeros_like(ball2_velocity_x)  # 初始化直立奖励

    # # 1. 高度奖励：如果 pelvis 高度低于阈值，则给予惩罚
    pelvis_height = humanoid1_pelvis_rb_states[..., 2]  # 根节点的 z 坐标（高度）
    # # print("pelvis_height: ", pelvis_height)
    height_threshold = 0.97  # 高度阈值（单位：米）
    # sparse
    # height_penalty = torch.where(
    #     (pelvis_height < height_threshold) & ~humanoid_die_calculated,
    #     die_penalty_float*100,  # 如果高度低于阈值，给予惩罚（这里的惩罚目前用的是乒乓球掉落的惩罚的100倍）
    #     torch.zeros_like(pelvis_height)  # 否则奖励为 0
    # )
    # humanoid_die_calculated |= pelvis_height < height_threshold  # 记录是否humanoid已经掉落

    # # dense
    # height_penalty = torch.where(
    #     (pelvis_height < height_threshold),
    #     -100 * (height_threshold - pelvis_height),  # 如果高度低于阈值，给予惩罚（dense）
    #     torch.zeros_like(ball2_velocity_x)  # 否则奖励为 0
    # )

    fall_down_count = torch.where(
        has_fallen > 0,
        torch.ones_like(pelvis_height),
        fall_down_count
    )
    # print("height_penalty: ", height_penalty)

    # ------------------------------------------------------------------- #

    # 提取 x, y, z 坐标
    ball_x = ball2_root_state_position[..., 0]  # 球的 x 坐标
    ball_y = ball2_root_state_position[..., 1]  # 球的 y 坐标
    ball_z = ball2_root_state_position[..., 2]  # 球的 z 坐标

    paddle_x = humanoid1_paddle_position[..., 0]  # 球拍的 x 坐标
    paddle_y = humanoid1_paddle_position[..., 1]  # 球拍的 y 坐标
    paddle_z = humanoid1_paddle_position[..., 2]  # 球拍的 z 坐标

    # 判断 ball 和 paddle 的 x 坐标是否第一次相差很小（0.2 之内）
    x_close = torch.abs(ball_x - paddle_x) < 0.2  # x 坐标相差小于 0.2
    first_time_close = x_close & ~paddle_condition_calculated  # 第一次满足条件

    # 计算 ball 和 paddle 在 y-z 平面上的距离
    yz_distance = torch.sqrt((ball_y - paddle_y) ** 2 + (ball_z - paddle_z) ** 2)

    # 判断 ball 是否在以 paddle 为中心的圆内（半径 0.15）
    in_circle = yz_distance < 0.15

    # 设置 reward
    pos_reward = torch.zeros_like(ball_x)  # 初始化 reward
    pos_reward = torch.where(
        first_time_close & ~humanoid_die_calculated,  # 仅当第一次 x 坐标相差很小
        torch.where(
            in_circle,  # 如果在圆内
            hit_paddle_reward,  # 给常数 reward
            miss_paddle_penalty_coefficient * yz_distance  # 如果不在圆内，penalty 与距离成正比
        ),
        pos_reward  # 如果不满足条件，reward = 0
    )

    closer_to_paddle_count = torch.where(
        first_time_close & in_circle,  # 仅当第一次 x 坐标相差很小且在圆内
        torch.ones_like(closer_to_paddle_count),  # 满足条件时，设置为 1
        closer_to_paddle_count  # 不满足条件时，保持原值
    )

    # if (pos_reward > 0).any():
    #     # print("paddle condition calculated: ", paddle_condition_calculated.sum())
    #     print("the ball is close to the paddle: ", (pos_reward > 0).sum())

    # ---------------------------------------------------------------------------------#

    hit_the_paddle = (pre_ball2_velocity_x < 0) & (ball2_velocity_x > 1.5)
    
    # 更新 hit_paddle_count
    hit_paddle_count = torch.where(
        hit_the_paddle,  # 条件：满足 hit_paddle 条件
        torch.ones_like(hit_paddle_count),  # 满足条件时，设置为 1
        hit_paddle_count  # 不满足条件时，保持原值
    )

    # if(hit_the_paddle.any()):
    #     print("hit the paddle: ", hit_the_paddle.sum())
    
    # 如果还没有记录过速度变化，则增加奖励
    # 只记录一次，防止多次打击paddle
    velocity_reward = torch.where(
        hit_the_paddle & ~paddle_condition_calculated & ~humanoid_die_calculated,  # 条件：速度变化且未计算过奖励
        alpha * torch.abs(ball2_velocity_x),  # 满足条件时，给予奖励
        velocity_reward  # 不满足条件时，保持原值
    )

    # 更新 paddle_condition_calculated 标志张量
    paddle_condition_calculated |= x_close  # 如果 x 坐标相差很小，则标记为已计算

    # 3/2 02：06增加 #
    # 时间惩罚项（仅在未击中时计算）
    humanoid_x = humanoid1_root_states[..., 0]
    time_penalty = torch.where(
        (ball_x > humanoid_x) & (ball2_velocity_x < 0),  # 条件：球在人前面 3.1 19：53改 # 20:50 发现bug，加上了后面的&
        -0.01 * progress_buf.float(), # -0.0003 * progress_buf.float() * progress_buf.float(),  # 满足条件时，计算时间惩罚 -0.015 * progress_buf.float(),
        torch.zeros_like(progress_buf)  # 不满足条件时，惩罚为 0
    )

    # -------------------------------------------------------------------------------#

    # 如果球击中对面球桌，给予奖励
    # hit_reward_and_penalty = torch.zeros_like(ball2_velocity_x)
    hit_reward_and_penalty, hit_table_calculated, hit_table_count = compute_gradient_penalty(ball2_root_state_position, ball2_velocity_x, hit_table_reward, not_hit_table_penalty, hit_table_calculated, hit_table_count, humanoid_die_calculated) # tensor, tensor, float, float

    # --------------------------------------------------------------------------------#

    # 过网的reward
    # 检查条件
    when_over_net = (
        (ball2_root_state_position[:, 0] > 1.72) &  # x 方向位置 > 1.72
        (ball2_root_state_position[:, 0] < 1.78) &  # x 方向位置 < 1.78
        (ball2_velocity_x > 0)                     # x 方向速度 > 0
    )

    suitable_for_cross_net = (
        (ball2_root_state_position[:, 2] > 0.96) &       # z 方向位置 > 0.96
        (ball2_root_state_position[:, 2] < 1.25)       # z 方向位置 < 1.10
    )
    over_height = torch.zeros_like(ball2_velocity_x)
    over_height = torch.where(
        ~(suitable_for_cross_net),
        torch.where(
            ball2_root_state_position[:, 2] > 1.25,
            ball2_root_state_position[:, 2] - 1.25,
            0.96-ball2_root_state_position[:, 2]
        ),
        over_height
    )
    
    # 分配 reward 和 penalty
    net_reward_and_penalty = torch.zeros_like(ball2_velocity_x)
    net_reward_and_penalty = torch.where(
        when_over_net & ~humanoid_die_calculated,  # 条件：满足过网条件
        torch.where(
            suitable_for_cross_net,  # 如果高度合适
            cross_net_reward_float,  # 给予过网奖励
            -400 * over_height
        ),
        net_reward_and_penalty  # 不满足条件时，保持原值
    )

    # if (net_reward_and_penalty > 0).any():
    #     print("cross the net: ", (net_reward_and_penalty > 0).sum())
   
    # 更新 cross_net_count
    cross_net_count = torch.where(
        net_reward_and_penalty > 0,  # 条件：满足 cross_net 条件
        torch.ones_like(cross_net_count),  # 满足条件时，设置为 1
        cross_net_count  # 不满足条件时，保持原值
    )

    # net_condition_calculated |= over_net_condition
    # ----------------------------------------------------------------------------------- #

    power = torch.abs(torch.multiply(dof_force_tensor, dof_vel)).sum(dim=-1)
    power_reward = -power_coefficient * power
    # power_reward[progress_buf <= 3] = 0  # First 3 frame power reward should not be counted. since they could be dropped.

    # ----------------------------------------------------------------------------------- #

    # 重置条件
    ones = torch.ones_like(reset_buf) # 全1
    die = torch.zeros_like(reset_buf) # 全0

    # NOEARLYSTOP!!!!!!!!!!!
    # 如果ball1和ball2掉到地面下，给penalty，只记录一次
    die_penalty = torch.zeros_like(ball2_velocity_x) 
    die_penalty = torch.where((ball2_root_state_position[..., 2] < threshold) & ~die_penalty_calculated & ~humanoid_die_calculated, 
                              die_penalty_float, 
                              die_penalty)
    die_penalty_calculated |= (ball2_root_state_position[..., 2] < threshold)  # 记录一次penalty

    humanoid_die_calculated |= (pelvis_height < height_threshold)  # 记录是否humanoid已经掉落
    
    # all rewards
    reward += pos_reward + power_reward + velocity_reward + hit_reward_and_penalty + net_reward_and_penalty + die_penalty + time_penalty + ref_reward # + height_penalty # + angular_penalty# + progress_reward + alive_reward(if not missing the ball); - penalty(if missing the ball)

    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)
    
    return reward, reset, paddle_condition_calculated, hit_table_calculated, die_penalty_calculated, humanoid_die_calculated, closer_to_paddle_count, hit_paddle_count, cross_net_count, hit_table_count, fall_down_count

# @torch.jit.script
def compute_imitation_reward_interos(root_states, body_states, dof_pos, dof_vel, ref_body_states, ref_dof_pos, ref_dof_vel, progress_buf, dof_force_tensor, body_states_id=None, feet_mask=None, is_g1=False, is_train=True):

    # k_posnferred) on parameter "is_g1".Because "is_g1" was not annotated with an explicit type it is assumed to be type 'Tensor'.:, k_vel, k_dof_pos, k_dof_vel = 10., 1.0, 5.0, 0.02
    # w_pos, w_vel, w_dof_pos, w_dof_vel = 0.3, 0.2, 0.3, 0.2

    if is_g1:
        k_pos, k_vel, k_dof_pos, k_dof_vel = 50., 4.0, 5.0, 0.05
        w_pos, w_vel, w_dof_pos, w_dof_vel = 0.4, 0.2, 0.2, 0.2
    else:
        k_pos, k_vel, k_dof_pos, k_dof_vel = 30., 1.0, 2.5, 0.5
        w_pos, w_vel, w_dof_pos, w_dof_vel = 0.4, 0.2, 0.2, 0.2

        # k_pos, k_vel, k_dof_pos, k_dof_vel = 50., 1.0, 20.0, 3.0
        # w_pos, w_vel, w_dof_pos, w_dof_vel = 0.3, 0.2, 0.3, 0.2

    B, J, _ = body_states.shape

    death_cost = -1.0

    power_coefficient = 0.0005  #0.0005
    if is_train:
        max_episode_length = 300
        if is_g1:
            termination_distance = 0.3# 0.32
        else:
            termination_distance = 0.4 #0.4# 0.4 # 0.4
    else:
        max_episode_length = 1e6
        termination_distance = 1e6

    if body_states_id is not None:
        body_pos = body_states[:, body_states_id, 0:3]
        body_vel = body_states[:, body_states_id, 7:10]

    ref_body_pos = ref_body_states[:, :, 0:3]
    ref_body_vel = ref_body_states[:, :, 7:10]

    # body position reward
    diff_global_body_pos = ref_body_pos - body_pos
    diff_global_body_pos[:, feet_mask] *= 2.0
    # diff_global_body_pos[:, feet_mask, 2] = 0.  # Do not penalize feet height difference
    diff_body_pos_dist = (diff_global_body_pos**2).mean(dim=-1).mean(dim=-1)
    r_body_pos = torch.exp(-k_pos * diff_body_pos_dist)

    # # body rotation reward
    # diff_global_body_rot = quat_mul(ref_body_rot, quat_conjugate(body_rot))
    # diff_global_body_angle = quat_to_angle_axis(diff_global_body_rot)[0]
    # diff_global_body_angle_dist = (diff_global_body_angle**2).mean(dim=-1)
    # r_body_rot = torch.exp(-k_rot * diff_global_body_angle_dist)

    # body linear velocity reward
    diff_global_vel = ref_body_vel - body_vel
    diff_global_vel_dist = (diff_global_vel**2).mean(dim=-1).mean(dim=-1)
    r_vel = torch.exp(-k_vel * diff_global_vel_dist)

    # # body angular velocity reward
    # diff_global_ang_vel = ref_body_ang_vel - body_ang_vel
    # diff_global_ang_vel_dist = (diff_global_ang_vel**2).mean(dim=-1).mean(dim=-1)
    # r_ang_vel = torch.exp(-k_ang_vel * diff_global_ang_vel_dist)
    if is_g1:
        diff_dof_pos = ((ref_dof_pos - dof_pos) ** 2).mean(dim=-1)
    else:
        dof_pos_quat = exp_map_to_quat(dof_pos.view(-1, 3))
        ref_dof_pos_quat = exp_map_to_quat(ref_dof_pos.view(-1, 3))
        diff_dof_pos_quat = quat_mul(ref_dof_pos_quat, quat_conjugate(dof_pos_quat))
        diff_dof_rot_angle = quat_to_angle_axis(diff_dof_pos_quat)[0].view(B, -1)
        diff_dof_pos = (diff_dof_rot_angle ** 2).mean(dim=-1)
        # dof_pos_quat = quat_to_tan_norm(exp_map_to_quat(dof_pos.view(-1, 3))).reshape(B, J - 1, -1)
        # ref_dof_pos_quat = quat_to_tan_norm(exp_map_to_quat(ref_dof_pos.view(-1, 3))).reshape(B, J - 1, -1)
        # diff_dof_pos = ((ref_dof_pos_quat - dof_pos_quat) ** 2).mean(dim=-1).mean(dim=-1)
    r_dof_pos = torch.exp(-k_dof_pos * diff_dof_pos)

    if is_g1:
        diff_dof_vel = ((ref_dof_vel - dof_vel)**2).mean(dim=-1)
    else:
        dof_vel_quat = exp_map_to_quat(dof_vel.view(-1, 3))
        ref_dof_vel_quat = exp_map_to_quat(ref_dof_vel.view(-1, 3))
        diff_dof_vel_quat = quat_mul(ref_dof_vel_quat, quat_conjugate(dof_vel_quat))
        diff_dof_vel_angle = quat_to_angle_axis(diff_dof_vel_quat)[0].view(B, -1)
        diff_dof_vel = (diff_dof_vel_angle ** 2).mean(dim=-1)
        # dof_vel_quat = quat_to_tan_norm(exp_map_to_quat(dof_vel.view(-1, 3))).reshape(B, J - 1, -1)
        # ref_dof_vel_quat = quat_to_tan_norm(exp_map_to_quat(ref_dof_vel.view(-1, 3))).reshape(B, J - 1, -1)
        # diff_dof_vel = ((ref_dof_vel_quat - dof_vel_quat)**2).mean(dim=-1).mean(dim=-1)
    r_dof_vel = torch.exp(-k_dof_vel * diff_dof_vel)

    reward = w_pos * r_body_pos + w_vel * r_vel + w_dof_pos * r_dof_pos + w_dof_vel * r_dof_vel
    # reward_raw = torch.stack([r_body_pos, r_body_rot, r_vel, r_ang_vel], dim=-1)


    # print(r_dof_vel[0].item())

    has_fallen = torch.any(torch.norm(body_pos - ref_body_pos, dim=-1).mean(dim=-1, keepdim=True) > termination_distance, dim=-1)
    # has_fallen = torch.any(torch.norm(body_pos - ref_body_pos, dim=-1) > termination_distance, dim=-1)

    # if has_fallen[0]:
    #     print('Has fallen!')

    reward = torch.where(has_fallen, torch.ones_like(reward) * death_cost, reward)

    power = torch.abs(torch.multiply(dof_force_tensor, dof_vel)).sum(dim=-1)
    power_reward = -power_coefficient * power
    # power_reward[progress_buf <= 3] = 0  # First 3 frame power reward should not be counted. since they could be dropped.
    reward += power_reward

    # print(r_body_pos[0].item(), r_vel[0].item(), r_dof_pos[0].item(), r_dof_vel[0].item(), power_reward[0].item(), flush=True)
    # print(r_body_pos[0].item(), r_vel[0].item())

    # reset agents
    reset = torch.where(has_fallen, torch.ones_like(reward), torch.zeros_like(reward))
    # reset[progress_buf >= max_episode_length] = 1

    # reset[:] = 1
    # TODO ALL one reset
    # reset = torch.ones_like(reward)

    return reward, reset


def compute_pingpong_observations(body_states, 
                                  body_states_id, 
                                  ball2_root_states,
                                  is_g1 = False):
    
    # pingpong ball --> pelvis pos and vel
    ball2_pos = ball2_root_states[..., 0:3]
    ball2_vel = ball2_root_states[..., 7:10]

    body_pos = body_states[:, body_states_id, 0:3]
    body_rot = body_states[:, body_states_id, 3:7]

    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    heading_rot_inv = calc_heading_quat_inv(root_rot)

    relative_ball2_pos = ball2_pos - root_pos  # [num_envs, 3]

    local_ball2_pos = my_quat_rotate(heading_rot_inv, relative_ball2_pos)  # [num_envs, 3]
    local_ball2_vel = my_quat_rotate(heading_rot_inv, ball2_vel)  # [num_envs, 3]

    # 3/2 15：30 增加一维球与humanoid平面在y方向上的距离
    # 这里用的全都是相对量
    local_ball2_vel_x = local_ball2_vel[..., 0] # shape[num_envs]
    local_ball2_vel_y = local_ball2_vel[..., 1]
    local_ball2_pos_x = local_ball2_pos[..., 0]
    local_ball2_pos_y = local_ball2_pos[..., 1]
    local_ball2_y_intersect = local_ball2_pos_y +(local_ball2_vel_y / (-local_ball2_vel_x + 1e-6)) * (local_ball2_pos_x)
    # print("local_ball2_y_intersect: ", local_ball2_y_intersect)

    obs = torch.cat((
        local_ball2_pos, local_ball2_vel, local_ball2_y_intersect.unsqueeze(-1)
    ), dim=-1)

    return obs


def compute_humanoid_observations(body_states, dof_pos, dof_vel, body_states_id, is_g1=False):

    body_pos = body_states[:, body_states_id, 0:3]
    body_rot = body_states[:, body_states_id, 3:7]
    body_vel = body_states[:, body_states_id, 7:10]

    B, J, _ = body_pos.shape

    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    # root_h = root_pos[:, 2:3]

    heading_rot_inv = calc_heading_quat_inv(root_rot)

    heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2)
    heading_rot_inv_expand = heading_rot_inv_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(
        heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1], heading_rot_inv_expand.shape[2])

    local_body_pos = my_quat_rotate(flat_heading_rot_inv, (body_pos - root_pos.unsqueeze(1)).view(B * J, -1)).reshape(B, -1)
    local_body_vel = my_quat_rotate(flat_heading_rot_inv, body_vel.view(B * J, -1)).reshape(B, -1)

    # flat_body_rot = body_rot.reshape(B * J, -1)  # This is global rotation of the body
    # flat_local_body_rot = quat_mul(flat_heading_rot_inv, flat_body_rot)
    # flat_local_body_rot_obs = quat_to_tan_norm(flat_local_body_rot)
    # local_body_rot_obs = flat_local_body_rot_obs.reshape(B, -1)

    obs_list = []
    # obs_list.append(root_h.clone())

    obs_list.append(local_body_pos)
    obs_list.append(local_body_vel)
    obs_list.append(dof_pos.clone())
    obs_list.append(dof_vel.clone() * 0.1)

    # obs_list += [local_body_rot_obs[:, :6]]

    obs = torch.cat(obs_list, dim=-1)
    return obs


def compute_imitation_observations(body_states, ref_body_states, ref_dof_pos, ref_dof_vel, body_states_id, is_g1=False):

    body_pos = body_states[:, body_states_id, 0:3]
    body_rot = body_states[:, body_states_id, 3:7]
    body_vel = body_states[:, body_states_id, 7:10]

    ref_body_pos = ref_body_states[:, body_states_id, 0:3]
    ref_body_vel = ref_body_states[:, body_states_id, 7:10]
    root_rot = body_rot[:, 0]

    time_steps = 1
    B, J, _ = body_pos.shape
    obs = []

    heading_inv_rot = calc_heading_quat_inv(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(
        time_steps, 0)

    ##### Body position and rotation differences
    diff_global_body_pos = ref_body_pos.view(B, time_steps, J, 3) - body_pos.view(B, 1, J, 3)
    diff_global_body_vel = ref_body_vel.view(B, time_steps, J, 3) - body_vel.view(B, 1, J, 3)
    diff_local_body_pos_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_body_pos.view(-1, 3))
    diff_local_body_vel_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_body_vel.view(-1, 3))
    # local_ref_body_pose_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), ref_body_pos.view(-1, 3))
    # local_ref_body_vel_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), ref_body_vel.view(-1, 3))

    obs.append(diff_local_body_pos_flat.view(B, -1) * 10.) #69
    obs.append(diff_local_body_vel_flat.view(B, -1)) # 69
    # obs.append(local_ref_body_pose_flat.view(B, -1))  #1.4
    # obs.append(local_ref_body_vel_flat.view(B, -1))   # 0.3
    obs.append(ref_dof_pos) # 27
    obs.append(ref_dof_vel)# 27

    obs = torch.cat(obs, dim=-1).view(B, -1)
    # print(obs.shape) #(numEnvs, 192)

    return obs


def compute_direction_observations(body_states, dof_pos, dof_vel, ref_body_states, ref_dof_pos, ref_dof_vel, body_states_id, is_g1=False, targets=None):

    body_pos = body_states[:, body_states_id, 0:3]
    body_rot = body_states[:, body_states_id, 3:7]
    body_vel = body_states[:, body_states_id, 7:10]

    ref_root_pos = ref_body_states[:, 0, 0:3]
    # ref_body_vel = ref_body_states[:, :, 7:10]
    root_pos = body_pos[:, 0]
    root_rot = body_rot[:, 0]

    time_steps = 1
    B, J, _ = body_pos.shape
    obs = []

    heading_inv_rot = calc_heading_quat_inv(root_rot)

    ##### Body position and rotation differences
    diff_global_root_pos = ref_root_pos.view(B, 3) - root_pos.view(B, 3)



    # diff_global_body_vel = ref_body_vel.view(B, time_steps, J, 3) - body_vel.view(B, 1, J, 3)
    diff_local_body_pos_flat = my_quat_rotate(heading_inv_rot.view(-1, 4), diff_global_root_pos.view(-1, 3))

    # print(diff_global_root_pos[0], diff_local_body_pos_flat[0])

    # diff_local_body_vel_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_body_vel.view(-1, 3))
    # local_ref_body_pose_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), ref_body_pos.view(-1, 3))
    # local_ref_body_vel_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), ref_body_vel.view(-1, 3))

    if targets is not None:
        print("target", targets)
        diff_local_body_pos_flat[:, 0] = targets[0]
        diff_local_body_pos_flat[:, 1] = targets[1]

    obs.append(diff_local_body_pos_flat.view(B, -1)[:, :2])
    # obs.append(diff_local_body_vel_flat.view(B, -1))
    # obs.append(local_ref_body_pose_flat.view(B, -1))  #1.4
    # obs.append(local_ref_body_vel_flat.view(B, -1))   # 0.3
    # obs.append(ref_dof_pos)
    # obs.append(ref_dof_vel)=

    obs = torch.cat(obs, dim=-1).view(B, -1)

    return obs


def compute_heightmap_observations(body_states, body_states_id, meshgrid, heightmap, trans_x, trans_y, grid_size, is_g1=False):
    body_pos = body_states[:, body_states_id, 0:3]
    body_rot = body_states[:, body_states_id, 3:7]
    body_vel = body_states[:, body_states_id, 7:10]

    root_rot = body_rot[:, 0]
    root_pos = body_pos[:, 0]

    time_steps = 1
    B, J, _ = body_pos.shape
    obs = []

    heading_rot = calc_heading_quat(root_rot)
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, meshgrid.shape[1], 1))

    meshgrid_global = my_quat_rotate(heading_rot_expand.view(-1, 4), meshgrid.view(-1, 3))
    meshgrid_global = meshgrid_global.reshape(-1, meshgrid.shape[1], 3) + root_pos.unsqueeze(1)
    meshgrid_global = meshgrid_global[..., :2]
    meshgrid_global[:, :, 0] -= trans_x
    meshgrid_global[:, :, 1] -= trans_y
    meshgrid_global = meshgrid_global // grid_size
    meshgrid_global = meshgrid_global.to(torch.int64)

    heightmap = heightmap[meshgrid_global[...,0],meshgrid_global[...,1]] - root_pos[:, 2].unsqueeze(1) + 0.9


    # heightmap_debug = heightmap[0].detach().cpu().numpy().reshape(10, 10)
    # cv2.imwrite('debug/heightmap.png', heightmap_debug * 255)
    #
    obs.append(heightmap.view(B, -1))
    obs = torch.cat(obs, dim=-1).view(B, -1)

    return obs


# # @torch.jit.script
# def compute_imitation_observations(root_states, body_states, ref_body_states, body_states_id):
#     # Adding pose information at the back
#     # Future tracks in this obs will not contain future diffs.
#
#     root_pos = root_states[:, 0:3]
#     root_rot = root_states[:, 3:7]
#     if body_states_id is not None:
#         body_pos = body_states[:, body_states_id, 0:3]
#         body_rot = body_states[:, body_states_id, 3:7]
#         body_vel = body_states[:, body_states_id, 7:10]
#         body_ang_vel = body_states[:, body_states_id, 10:13]
#     ref_body_pos = ref_body_states[:, :, 0:3]
#     ref_body_rot = ref_body_states[:, :, 3:7]
#     ref_body_vel = ref_body_states[:, :, 7:10]
#     ref_body_ang_vel = ref_body_states[:, :, 10:13]
#
#     obs = []
#     time_steps = 1
#     B, J, _ = body_pos.shape
#
#     heading_inv_rot = calc_heading_quat_inv(root_rot)
#     heading_rot = calc_heading_quat(root_rot)
#     heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(
#         time_steps, 0)
#     heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
#
#     ##### Body position and rotation differences
#     diff_global_body_pos = ref_body_pos.view(B, time_steps, J, 3) - body_pos.view(B, 1, J, 3)
#     diff_local_body_pos_flat = my_quat_rotate(heading_inv_rot_expand.view(-1, 4),
#                                                           diff_global_body_pos.view(-1, 3))
#
#     body_rot[:, None].repeat_interleave(time_steps, 1)
#     diff_global_body_rot = quat_mul(ref_body_rot.view(B, time_steps, J, 4), quat_conjugate(
#         body_rot[:, None].repeat_interleave(time_steps, 1)))
#     diff_local_body_rot_flat = quat_mul(
#         quat_mul(heading_inv_rot_expand.view(-1, 4), diff_global_body_rot.view(-1, 4)),
#         heading_rot_expand.view(-1, 4))  # Need to be change of basis
#
#     ##### linear and angular  Velocity differences
#     diff_global_vel = ref_body_vel.view(B, time_steps, J, 3) - body_vel.view(B, 1, J, 3)
#     diff_local_vel = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_vel.view(-1, 3))
#
#     diff_global_ang_vel = ref_body_ang_vel.view(B, time_steps, J, 3) - body_ang_vel.view(B, 1, J, 3)
#     diff_local_ang_vel = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_ang_vel.view(-1, 3))
#
#     ##### body pos + Dof_pos This part will have proper futuers.
#     local_ref_body_pos = ref_body_pos.view(B, time_steps, J, 3) - root_pos.view(B, 1, 1,
#                                                                                 3)  # preserves the body position
#     local_ref_body_pos = my_quat_rotate(heading_inv_rot_expand.view(-1, 4), local_ref_body_pos.view(-1, 3))
#
#     local_ref_body_rot = quat_mul(heading_inv_rot_expand.view(-1, 4), ref_body_rot.view(-1, 4))
#     local_ref_body_rot = quat_to_tan_norm(local_ref_body_rot)
#
#     # make some changes to how futures are appended.
#     obs.append(diff_local_body_pos_flat.view(B, time_steps, -1))  # 1 * timestep * 24 * 3
#     obs.append(quat_to_tan_norm(diff_local_body_rot_flat).view(B, time_steps, -1))  # 1 * timestep * 24 * 6
#     obs.append(diff_local_vel.view(B, time_steps, -1))  # timestep  * 24 * 3
#     obs.append(diff_local_ang_vel.view(B, time_steps, -1))  # timestep  * 24 * 3
#     obs.append(local_ref_body_pos.view(B, time_steps, -1))  # timestep  * 24 * 3
#     obs.append(local_ref_body_rot.view(B, time_steps, -1))  # timestep  * 24 * 6
#
#     obs = torch.cat(obs, dim=-1).view(B, -1)
#     return obs
#
# # @torch.jit.script
# def compute_humanoid_observations(body_states, body_states_id=None):
#     if body_states_id is not None:
#         body_pos = body_states[:, body_states_id, 0:3]
#         body_rot = body_states[:, body_states_id, 3:7]
#         body_vel = body_states[:, body_states_id, 7:10]
#         body_ang_vel = body_states[:, body_states_id, 10:13]
#
#     root_pos = body_pos[:, 0, :]
#     root_rot = body_rot[:, 0, :]
#
#     root_h = root_pos[:, 2:3]
#
#     heading_rot_inv = calc_heading_quat_inv(root_rot)
#
#     root_h_obs = root_h
#
#     heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2)
#     heading_rot_inv_expand = heading_rot_inv_expand.repeat((1, body_pos.shape[1], 1))
#     flat_heading_rot_inv = heading_rot_inv_expand.reshape(
#         heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1], heading_rot_inv_expand.shape[2])
#
#     root_pos_expand = root_pos.unsqueeze(-2)
#     local_body_pos = body_pos - root_pos_expand
#     flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1],
#                                                  local_body_pos.shape[2])
#     flat_local_body_pos = my_quat_rotate(flat_heading_rot_inv, flat_local_body_pos)
#     local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0],
#                                                  local_body_pos.shape[1] * local_body_pos.shape[2])
#     local_body_pos = local_body_pos[..., 3:]  # remove root pos
#
#     flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1],
#                                      body_rot.shape[2])  # This is global rotation of the body
#     flat_local_body_rot = quat_mul(flat_heading_rot_inv, flat_body_rot)
#     flat_local_body_rot_obs = quat_to_tan_norm(flat_local_body_rot)
#     local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0],
#                                                          body_rot.shape[1] * flat_local_body_rot_obs.shape[1])
#
#     flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
#     flat_local_body_vel = my_quat_rotate(flat_heading_rot_inv, flat_body_vel)
#     local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])
#
#     flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
#     flat_local_body_ang_vel = my_quat_rotate(flat_heading_rot_inv, flat_body_ang_vel)
#     local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0],
#                                                          body_ang_vel.shape[1] * body_ang_vel.shape[2])
#
#     # contact_force = vec_sensor_tensor.view(-1, 12) * 0.01
#
#     obs_list = []
#     obs_list.append(root_h_obs)
#     obs_list += [local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel]
#
#     obs = torch.cat(obs_list, dim=-1)
#     return obs

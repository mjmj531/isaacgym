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


'''
未设置randomize 代码全部注释掉了
'''

import numpy as np
import os
import math

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgymenvs.utils.torch_jit_utils import scale, unscale, quat_mul, quat_conjugate, quat_from_angle_axis, \
    to_torch, get_axis_params, torch_rand_float, tensor_clamp, compute_heading_and_up, compute_rot, normalize_angle

from isaacgymenvs.tasks.base.vec_task import VecTask

import torch
import random

# simple asset descriptor for selecting from a list
class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments


asset_descriptors = [
    AssetDesc("mjcf/nv_humanoid.xml", False),
    AssetDesc("mjcf/nv_ant.xml", False),
    AssetDesc("urdf/cartpole.urdf", False),
    AssetDesc("urdf/sektion_cabinet_model/urdf/sektion_cabinet.urdf", False),
    AssetDesc("urdf/franka_description/robots/franka_panda.urdf", True),
    AssetDesc("urdf/kinova_description/urdf/kinova.urdf", False),
    AssetDesc("urdf/anymal_b_simple_description/urdf/anymal.urdf", True),
]

class HumanoidPingpong(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        
        self.randomization_params = self.cfg["task"]["randomization_params"]
        # self.randomize = self.cfg["task"]["randomize"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
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

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        # 一个环境中最多多少步之后就reset
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        
        # Observations(copy from ball_ballance):
        # 0:3 - activated DOF positions # 这里一共有3个被激活的关节
        # 3:6 - activated DOF velocities
        # 6:9 - ball position
        # 9:12 - ball linear velocity
        # 12:15 - sensor force (same for each sensor)
        # 15:18 - sensor torque 1
        # 18:21 - sensor torque 2
        # 21:24 - sensor torque 3
        self.cfg["env"]["numObservations"] = 108
        
        # Actions: target velocities for the 3 actuated DOFs
        self.cfg["env"]["numActions"] = 21

        # speed_scale: 控制仿真中关节运动或动画的速度缩放比例
        self.speed_scale = 1.0

        # pingpong_ball initial speed and tilt angle range
        self.initial_speed_range = (6.5, 7.5)  # 初速度范围 (单位: 米/秒)
        self.tilt_angle_range = (-5.0, 5.0)  # 倾斜角范围 (单位: 度)

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(0, -2, 1)
            cam_target = gymapi.Vec3(0, 0, 1)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        actors_per_env = 5
        dofs_per_env = 26*2
        rigid_bodies_per_env = 40*2 + 1 + 1*2
        # sensors_per_env = 3

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        # sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        self.all_actor_indices = torch.arange(5 * self.num_envs, dtype=torch.int32).view(self.num_envs, 5)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # sensor
        # self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        # dof_force
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        # root state tensors
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.vec_root_states = self.root_states.view(self.num_envs, actors_per_env, 13)

        # robot and ball state tensors
        self.robot1_root_states = self.vec_root_states[:, 0, :] # shape: (num_envs, 13)
        self.robot2_root_states = self.vec_root_states[:, 1, :]
        self.table_root_states = self.vec_root_states[:, 2, :]
        self.ball1_root_states = self.vec_root_states[:, 3, :]
        self.ball2_root_states = self.vec_root_states[:, 4, :] # shape: (num_envs, 13)
        
        # initial root states
        self.initial_vec_root_states = self.vec_root_states.clone() # shape: (num_envs, 5, 13)
        self.initial_pos = self.initial_vec_root_states[:, :, 0:3]
        self.initial_rot = self.initial_vec_root_states[:, :, 3:7]
        self.robot1_initial_pos = self.initial_vec_root_states[:, 0, 0:3]
        self.robot1_initial_rot = self.initial_vec_root_states[:, 0, 3:7]
        self.robot2_initial_pos = self.initial_vec_root_states[:, 1, 0:3]
        self.robot2_initial_rot = self.initial_vec_root_states[:, 1, 3:7]
        self.table_initial_pos = self.initial_vec_root_states[:, 2, 0:3]
        self.table_initial_rot = self.initial_vec_root_states[:, 2, 3:7]
        self.ball1_initial_pos = self.initial_vec_root_states[:, 3, 0:3]
        self.ball1_initial_rot = self.initial_vec_root_states[:, 3, 3:7]
        self.ball2_initial_pos = self.initial_vec_root_states[:, 4, 0:3]
        self.ball2_initial_rot = self.initial_vec_root_states[:, 4, 3:7]

        # self.initial_velocity = self.initial_root_states[:, 7:13].clone()

        # create some wrapper tensors for different slices
        # dof state tensors
        self.dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        self.vec_dof_states = self.dof_states.view(self.num_envs, dofs_per_env, 2)
        self.dof_pos = self.vec_dof_states[..., 0]
        self.dof_vel = self.vec_dof_states[..., 1]
        self.initial_dof_states = self.vec_dof_states.clone() # shape: (num_envs, 26*2, 2)

        # rigid body state tensors
        self.rb_states = gymtorch.wrap_tensor(_rb_states)
        self.vec_rb_states = self.rb_states.view(self.num_envs, -1, 3)
        self.robot1_rb_states = self.vec_rb_states[:, 0:40, :] # torch.Size([16, 40, 13])
        self.robot1_paddle_rb_states = self.vec_rb_states[:, 39, :] # torch.Size([16, 13])
        self.robot2_rb_states = self.vec_rb_states[:, 40:80, :] # torch.Size([16, 40, 13])
        self.robot2_paddle_rb_states = self.vec_rb_states[:, 79, :] # torch.Size([16, 13])

        # initialize some data used later on
        # self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        # self.heading_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        # self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        # self.basis_vec0 = self.heading_vec.clone()
        # self.basis_vec1 = self.up_vec.clone()

        # self.targets = to_torch([1000, 0, 0], device=self.device).repeat((self.num_envs, 1))
        # self.target_dirs = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.dt = self.cfg["sim"]["dt"]
        # self.potentials = to_torch([-1000./self.dt], device=self.device).repeat(self.num_envs)
        # self.prev_potentials = self.potentials.clone()

    def clamp(self, x, min_value, max_value):
        return max(min(x, max_value), min_value)

    def initialize_dof_states(self, num_dofs, dof_names, dof_types, dof_positions, lower_limits, upper_limits, has_limits, stiffnesses, dampings, arm_dof_names):
        # initialize default positions, limits, and speeds (make sure they are in reasonable ranges)
        defaults = np.zeros(num_dofs)
        speeds = np.zeros(num_dofs)
        for i in range(num_dofs):
            dof_name = dof_names[i]
            if dof_name not in arm_dof_names:
                stiffnesses[i] = 1e6
                dampings[i] = 1e6
                speeds[i] = 0.0
                
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

        for i in range(num_dofs):
            print(defaults[i], speeds[i])
    
    def generate_random_speed_for_ball(self, initial_speed_range, tilt_angle_range):
        # ball_velocities = torch.zeros((num_envs, 2, 3))  # 速度 (vx, vy, vz)
        # 初始化球的位置，给定初速度和倾斜角
        speed_1 = random.uniform(*initial_speed_range)
        # print("i=", i, "speed=", speed_1)
        tilt_angle_1 = random.uniform(*tilt_angle_range)
        # print("i=", i, "tilt_angle=", tilt_angle_1)
        velocity_1 = gymapi.Vec3(speed_1 * math.cos(math.radians(tilt_angle_1)), speed_1 * math.sin(math.radians(tilt_angle_1)), 0.0)
        # ball_velocities_1 = torch.tensor([velocity_1.x, velocity_1.y, velocity_1.z  # 速度 (vx, vy, vz)
        #                                     ])
        
        speed_2 = -random.uniform(*initial_speed_range)
        tilt_angle_2 = random.uniform(*tilt_angle_range)
        velocity_2 = gymapi.Vec3(speed_2 * math.cos(math.radians(tilt_angle_2)), speed_2 * math.sin(math.radians(tilt_angle_2)), 0.0)
        # ball_velocities_2 = torch.tensor([velocity_2.x, velocity_2.y, velocity_2.z  # 速度 (vx, vy, vz)
        #                                         ])
        return velocity_1, velocity_2

    def create_sim(self):
        # self.up_axis_idx = 2  # index of up axis: Y=1, Z=2
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.8
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.dt = self.sim_params.dt

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # # If randomizing, apply once immediately on startup before the fist sim step
        # if self.randomize:
        #     self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # load asset
        # robot asset
        robot_asset_root = "../../assets"
        robot_asset_file = "mjcf/g1_29dof_rev_1_0_pingpong_only_right_arm.urdf"
        asset_options = gymapi.AssetOptions()
        # asset_options.fix_base_link = False
        asset_options.fix_base_link = True
        # asset_options.flip_visual_attachments = asset_descriptors[args.asset_id].flip_visual_attachments
        asset_options.use_mesh_materials = True

        print("Loading asset '%s' from '%s'" % (robot_asset_file, robot_asset_root))
        robot_asset = self.gym.load_asset(self.sim, robot_asset_root, robot_asset_file, asset_options)

        # get array of DOF names
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)

        # dof_names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 
        #             'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 
        #             'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 
        #             'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 
        #             # 'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint', 'left_shoulder_pitch_joint', 
        #             'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 
        #             'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 
        #             'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 
        #             'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']

        self.right_arm_dof_names = [
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
        ]

        # 这些是不与环境交互的不变量
        # get array of DOF properties
        self.dof_props = self.gym.get_asset_dof_properties(robot_asset)

        # create an array of DOF states that will be used to update the actors
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset) # 26

        self.dof_states_26 = np.zeros(self.num_dofs, dtype=gymapi.DofState.dtype)

        # get list of DOF types
        self.dof_types = [self.gym.get_asset_dof_type(robot_asset, i) for i in range(self.num_dofs)]

        # get the position slice of the DOF state array
        self.dof_positions = self.dof_states_26['pos']

        # get the limit-related slices of the DOF properties array
        self.stiffnesses = self.dof_props['stiffness']
        self.dampings = self.dof_props['damping']
        self.armatures = self.dof_props['armature']
        self.has_limits = self.dof_props['hasLimits']
        self.lower_limits = self.dof_props['lower']
        self.upper_limits = self.dof_props['upper']

        # pingpong table asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        # asset_options.flip_visual_attachments = asset_descriptors[args.asset_id].flip_visual_attachments
        asset_options.use_mesh_materials = True
        pingpong_table_asset = self.gym.load_asset(self.sim, '../../assets/urdf', 'pingpong_table.urdf', asset_options)
        
        # pingpong ball asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False  
        # asset_options.flip_visual_attachments = asset_descriptors[args.asset_id].flip_visual_attachments
        asset_options.use_mesh_materials = True

        pingpong_ball_asset = self.gym.load_asset(self.sim, '../../assets/urdf', 'small_ball.urdf', asset_options)

        self.actor_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            self.envs.append(env_ptr)

            # add actor 
            # robot1_handle
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            # param6: bit collision
            self.robot1_handle = self.gym.create_actor(env_ptr, robot_asset, pose, "robot1", i, 0)
            self.actor_handles.append(self.robot1_handle)

            # set default DOF positions
            self.initialize_dof_states(self.num_dofs, self.dof_names, self.dof_types, self.dof_positions, self.dof_states['vel'], self.lower_limits, self.upper_limits, self.has_limits, self.stiffnesses, self.dampings, self.arm_dof_names)
            self.gym.set_actor_dof_states(env_ptr, self.robot1_handle, self.dof_states_26, gymapi.STATE_ALL)
            # set DOF properties
            self.gym.set_actor_dof_properties(env_ptr, self.robot1_handle, self.dof_props)
            
            # add actor
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(3.5, 0.0, 1.0)
            pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

            self.robot2_handle = self.gym.create_actor(env_ptr, robot_asset, pose, "robot2", i, 0)
            self.actor_handles.append(self.robot2_handle)

            # set default DOF positions
            self.gym.set_actor_dof_states(env_ptr, self.robot2_handle, self.dof_states_26, gymapi.STATE_ALL)
            self.gym.set_actor_dof_properties(env_ptr, self.robot2_handle, self.dof_props)

            # add table
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(1.75, 0.0, 0.0)
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            self.table_handle = self.gym.create_actor(env_ptr, pingpong_table_asset, pose, "pingpong_table", i, 0)
            shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, self.table_handle)
            shape_props[0].restitution = 0.6
            shape_props[0].friction = 0.2
            shape_props[0].rolling_friction = 0.2
            self.gym.set_actor_rigid_shape_properties(env_ptr, self.table_handle, shape_props)
            self.actor_handles.append(self.table_handle)

            # generate random speed and tilt angle for each ball
            velocity_1, velocity_2 = self.generate_random_speed_for_ball(self.initial_speed_range, self.tilt_angle_range)

            # add ball 1
            name = 'pingpong_ball_1'.format(i)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.4, 0.28, 1.3)
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0) # 没有旋转

            self.ball1_handle = self.gym.create_actor(env_ptr, pingpong_ball_asset, pose, name, i, 0)   
            # set restitution coefficient
            shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, self.ball1_handle)
            shape_props[0].restitution = 0.9
            shape_props[0].friction = 0.2
            shape_props[0].rolling_friction = 0.2
            self.gym.set_actor_rigid_shape_properties(env_ptr, self.ball1_handle, shape_props)

            self.actor_handles.append(self.ball1_handle)

            # apply linear velocity to ball 1
            self.gym.set_rigid_linear_velocity(env_ptr, 
                                        self.gym.get_rigid_handle(env_ptr, name, self.gym.get_actor_rigid_body_names(env_ptr, self.ball1_handle)[0]), 
                                        velocity_1)

            # add ball 2
            name = 'pingpong_ball_2'.format(i)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(3.1, -0.28, 1.3)
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0) # 没有旋转

            self.ball2_handle = self.gym.create_actor(env_ptr, pingpong_ball_asset, pose, name, i, 0)
            # set restitution coefficient
            shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, self.ball2_handle)
            shape_props[0].restitution = 0.9
            shape_props[0].friction = 0.2
            shape_props[0].rolling_friction = 0.2
            self.gym.set_actor_rigid_shape_properties(env_ptr, self.ball2_handle, shape_props)
            self.actor_handles.append(self.ball2_handle)

            # apply linear velocity to ball 2
            self.gym.set_rigid_linear_velocity(env_ptr, 
                                        self.gym.get_rigid_handle(env_ptr, name, self.gym.get_actor_rigid_body_names(env_ptr, self.ball2_handle)[0]), 
                                        velocity_2)
            
    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_pingpong_reward(
            self.robot1_paddle_rb_states,
            self.robot2_paddle_rb_states,
            self.ball1_root_states,
            self.ball2_root_states,
            self.reset_buf,
            self.progress_buf,
            self.max_episode_length
        )

        # self.rew_buf[:], self.reset_buf = compute_humanoid_reward(
        #     self.obs_buf,
        #     self.reset_buf,
        #     self.progress_buf,
        #     self.actions,
        #     self.up_weight,
        #     self.heading_weight,
        #     self.potentials,
        #     self.prev_potentials,
        #     self.actions_cost_scale,
        #     self.energy_cost_scale,
        #     self.joints_at_limit_cost_scale,
        #     self.max_motor_effort,
        #     self.motor_efforts,
        #     self.termination_height,
        #     self.death_cost,
        #     self.max_episode_length
        # )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)

        self.gym.refresh_dof_force_tensor(self.sim)

        self.obs_buf[:] = compute_pingpong_observations(
            self.obs_buf,
            self.robot1_paddle_rb_states,
            self.robot2_paddle_rb_states,
            self.robot1_paddle_rb_states,
            self.robot2_paddle_rb_states,
            self.ball1_root_states,
            self.ball2_root_states
        )

        # self.obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:] = compute_humanoid_observations(
        #     self.obs_buf, self.root_states, self.targets, self.potentials,
        #     self.inv_start_rot, self.dof_pos, self.dof_vel, self.dof_force_tensor,
        #     self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
        #     self.vec_sensor_tensor, self.actions, self.dt, self.contact_force_scale, self.angular_velocity_scale,
        #     self.basis_vec0, self.basis_vec1)

    # def check_reset(self, env_ids, depth_threshold=0.05):
    #     # Get the height (z) position of ball 1 and ball 2
    #     ball1_z = self.vec_root_states[env_ids, 3, 2]  # Ball 1 is at index 3
    #     ball2_z = self.vec_root_states[env_ids, 4, 2]  # Ball 2 is at index 4
    #     # print(ball1_z)
        
    #     # If both balls are below the threshold, return True (reset needed)
    #     if ball1_z < depth_threshold and ball2_z < depth_threshold:
    #         return True
    #     return False

    def reset_idx(self, env_ids):
        # # Randomization can happen only at reset time, since it can reset actor positions on GPU
        # if self.randomize:
        #     self.apply_randomizations(self.randomization_params)

        # positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof), device=self.device)
        # velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        # (self.num_envs, actors_per_env, 13)
        self.vec_root_states[env_ids, :, 0:3] = self.initial_pos[env_ids, :]
        self.vec_root_states[env_ids, :, 3:7] = self.initial_rot[env_ids, :]

        velocity_1, velocity_2 = self.generate_random_speed_for_ball(self.initial_speed_range, self.tilt_angle_range)
        # device
        ball_velocities_1 = torch.tensor([velocity_1.x, velocity_1.y, velocity_1.z  # 速度 (vx, vy, vz)
                                            ], device=self.device)
        ball_velocities_2 = torch.tensor([velocity_2.x, velocity_2.y, velocity_2.z  # 速度 (vx, vy, vz)
                                            ], device=self.device)
        
        self.vec_root_states[env_ids, 3, 7:10] = ball_velocities_1
        self.vec_root_states[env_ids, 4, 7:10] = ball_velocities_2

        self.vec_dof_states[env_ids, :, :] = self.initial_dof_states[env_ids, :, :]

        # env_ids_int32 = env_ids.to(dtype=torch.int32)

        # all_actor_indices = torch.arange(5 * self.num_envs, dtype=torch.int32).view(self.num_envs, 5)
        actor_indices = self.all_actor_indices[env_ids].flatten()
        # set root state tensor to the original one
        self.gym.set_actor_root_state_tensor_indexed(self.sim, 
                                                     gymtorch.unwrap_tensor(self.root_states), 
                                                     gymtorch.unwrap_tensor(actor_indices), len(actor_indices))
        
        # set dof to the original position and velocity
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_states),
                                              gymtorch.unwrap_tensor(actor_indices), len(actor_indices))
        

        # self.gym.set_actor_dof_states(env_ids, self.robot1_handle, self.dof_states_29, gymapi.STATE_ALL)
        # self.gym.set_actor_dof_states(env_ids, self.robot2_handle, self.dof_states_29, gymapi.STATE_ALL)


        # to_target = self.targets[env_ids] - self.initial_root_states[env_ids, 0:3]
        # to_target[:, self.up_axis_idx] = 0
        # self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        # self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
        force_tensor = gymtorch.unwrap_tensor(forces)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

    def post_physics_step(self):
        self.progress_buf += 1
        # self.randomize_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

        # # debug viz
        # if self.viewer and self.debug_viz:
        #     self.gym.clear_lines(self.viewer)

        #     points = []
        #     colors = []
        #     for i in range(self.num_envs):
        #         origin = self.gym.get_env_origin(self.envs[i])
        #         pose = self.root_states[:, 0:3][i].cpu().numpy()
        #         glob_pos = gymapi.Vec3(origin.x + pose[0], origin.y + pose[1], origin.z + pose[2])
        #         points.append([glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.heading_vec[i, 0].cpu().numpy(),
        #                        glob_pos.y + 4 * self.heading_vec[i, 1].cpu().numpy(),
        #                        glob_pos.z + 4 * self.heading_vec[i, 2].cpu().numpy()])
        #         colors.append([0.97, 0.1, 0.06])
        #         points.append([glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.up_vec[i, 0].cpu().numpy(), glob_pos.y + 4 * self.up_vec[i, 1].cpu().numpy(),
        #                        glob_pos.z + 4 * self.up_vec[i, 2].cpu().numpy()])
        #         colors.append([0.05, 0.99, 0.04])

        #     self.gym.add_lines(self.viewer, None, self.num_envs * 2, points, colors)

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_pingpong_reward(
    robot1_paddle_rb_states, robot2_paddle_rb_states, 
    ball1_root_states, ball2_root_states, 
    reset_buf, progress_buf, 
    max_episode_length
    ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]
    
    threshold = 0.1  # 乒乓球掉落高度阈值

    # compute the distance between the robot's right hand and the pingpong ball
    robot1_right_hand_position = robot1_paddle_rb_states[..., 0:3] 
    ball2_root_state_position = ball2_root_states[..., 0:3]

    dist1 = torch.sqrt((robot1_right_hand_position[..., 0] - ball2_root_state_position[..., 0]) ** 2 +
                        (robot1_right_hand_position[..., 1] - ball2_root_state_position[..., 1]) ** 2 +
                        (robot1_right_hand_position[..., 2] - ball2_root_state_position[..., 2]) ** 2)

    robot2_right_hand_position = robot2_paddle_rb_states[..., 0:3]
    ball1_root_state_position = ball1_root_states[..., 0:3]

    dist2 = torch.sqrt((robot2_right_hand_position[..., 0] - ball1_root_state_position[..., 0]) ** 2 +
                        (robot2_right_hand_position[..., 1] - ball1_root_state_position[..., 1]) ** 2 +
                        (robot2_right_hand_position[..., 2] - ball1_root_state_position[..., 2]) ** 2)
    
    pos_reward1 = 1.0 / (1.0 + dist1 * dist1)  # robot1 和 ball2 之间的奖励
    pos_reward2 = 1.0 / (1.0 + dist2 * dist2)  # robot2 和 ball1 之间的奖励

    # 根据目标距离计算奖励 (目标距离越小，奖励越大)
    pos_reward = pos_reward1 + pos_reward2

    # 总奖励
    reward = pos_reward
    
    # 重置条件
    ones = torch.ones_like(reset_buf) # 全1张量
    die = torch.zeros_like(reset_buf) # 全0张量
    
    # 如果ball1和ball2掉到地面下，需要重置
    die = torch.where((ball1_root_state_position[..., 2] < threshold) & (ball1_root_state_position[..., 2] < threshold), ones, die)
    
    # 如果回合时间已结束，需要重置
    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)
    
    return reward, reset

# @torch.jit.script
# def compute_robot_hit_reward(
#     robot1_right_hand_positions,  # 机器人末端执行器的位置 [num_envs, 3]
#     ball_positions,      # 乒乓球的位置 [num_envs, 3]
#     ball_velocities,     # 乒乓球的速度 [num_envs, 3]
#     ball_radius,         # 乒乓球的半径
#     reset_buf,           # 重置缓冲区 [num_envs]
#     progress_buf,        # 进度缓冲区 [num_envs]
#     max_episode_length,  # 每个 episode 的最大步数
#     hit_threshold: float = 0.1,  # 击中距离阈值
#     drop_threshold: float = 0.05  # 掉落高度阈值
# ):
#     # 计算机器人末端执行器与乒乓球之间的距离
#     distance = torch.norm(robot1_right_hand_positions - ball_positions, dim=-1)
    
#     # 距离奖励：距离越小，奖励越大
#     distance_reward = 1.0 / (1.0 + distance)
    
#     # 击中奖励：如果距离小于阈值，给予额外奖励
#     hit_reward = torch.where(
#         distance < hit_threshold,
#         torch.ones_like(distance_reward) * 10.0,  # 击中奖励
#         torch.zeros_like(distance_reward)          # 未击中
#     )
    
#     # 鼓励尽快击中
#     time_penalty = -0.01 * (progress_buf / max_episode_length)
    
#     # 掉落惩罚：如果乒乓球高度低于阈值，给予负奖励
#     drop_penalty = torch.where(
#         ball_positions[..., 2] < drop_threshold,
#         torch.ones_like(distance_reward) * -5.0,  # 掉落惩罚
#         torch.zeros_like(distance_reward)          # 未掉落
#     )
    
#     # 总奖励 = 距离奖励 + 击中奖励 + 时间惩罚 + 掉落惩罚
#     reward = distance_reward + hit_reward + time_penalty + drop_penalty
    
#     # 判断是否需要重置环境
#     reset = torch.where(
#         progress_buf >= max_episode_length - 1,  # episode 结束
#         torch.ones_like(reset_buf),              # 重置
#         reset_buf                                # 保持
#     )
#     reset = torch.where(
#         ball_positions[..., 2] < drop_threshold,  # 乒乓球掉落
#         torch.ones_like(reset_buf),               # 重置
#         reset                                    # 保持
#     )
    
#     return reward, reset


@torch.jit.script
def compute_humanoid_reward(
    obs_buf,
    reset_buf,
    progress_buf,
    actions,
    up_weight,
    heading_weight,
    potentials,
    prev_potentials,
    actions_cost_scale,
    energy_cost_scale,
    joints_at_limit_cost_scale,
    max_motor_effort,
    motor_efforts,
    termination_height,
    death_cost,
    max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float, Tensor, Tensor, float, float, float, float, Tensor, float, float, float) -> Tuple[Tensor, Tensor]

    # reward from the direction headed
    heading_weight_tensor = torch.ones_like(obs_buf[:, 11]) * heading_weight
    heading_reward = torch.where(obs_buf[:, 11] > 0.8, heading_weight_tensor, heading_weight * obs_buf[:, 11] / 0.8)
 
    # reward for being upright
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(obs_buf[:, 10] > 0.93, up_reward + up_weight, up_reward)

    actions_cost = torch.sum(actions ** 2, dim=-1)

    # energy cost reward
    motor_effort_ratio = motor_efforts / max_motor_effort
    scaled_cost = joints_at_limit_cost_scale * (torch.abs(obs_buf[:, 12:33]) - 0.98) / 0.02
    dof_at_limit_cost = torch.sum((torch.abs(obs_buf[:, 12:33]) > 0.98) * scaled_cost * motor_effort_ratio.unsqueeze(0), dim=-1)

    electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 33:54]) * motor_effort_ratio.unsqueeze(0), dim=-1)

    # reward for duration of being alive
    alive_reward = torch.ones_like(potentials) * 2.0
    progress_reward = potentials - prev_potentials

    total_reward = progress_reward + alive_reward + up_reward + heading_reward - \
        actions_cost_scale * actions_cost - energy_cost_scale * electricity_cost - dof_at_limit_cost

    # adjust reward for fallen agents
    total_reward = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward)

    # reset agents
    reset = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    return total_reward, reset

@torch.jit.script
def compute_pingpong_observations(obs_buf, robot1_root_states, robot2_root_states, 
                                  robot1_paddle_rb_states, robot2_paddle_rb_states, 
                                  ball1_root_states, ball2_root_states
                                  ):

    # robot1
    robot1_paddle_position = robot1_paddle_rb_states[..., 0:3] 
    robot1_paddle_velocity = robot1_paddle_rb_states[..., 7:10]
    robot1_obs = torch.cat((robot1_paddle_position, robot1_paddle_velocity), dim=-1)

    # ball2
    ball2_root_state_position = ball2_root_states[..., 0:3]
    ball2_root_state_velocity = ball2_root_states[..., 7:10]
    ball2_obs = torch.cat((ball2_root_state_position, ball2_root_state_velocity), dim=-1)

    # robot2
    robot2_paddle_position = robot2_paddle_rb_states[..., 0:3]
    robot2_paddle_velocity = robot2_paddle_rb_states[..., 7:10]
    robot2_obs = torch.cat((robot2_paddle_position, robot2_paddle_velocity), dim=-1)

    # ball1
    ball1_root_state_position = ball1_root_states[:, 0:3]  
    ball1_root_state_velocity = ball1_root_states[:, 7:10] 
    ball1_obs = torch.cat((ball1_root_state_position, ball1_root_state_velocity), dim=-1)

    # # 计算潜力值（这里简单地使用负的球与机器人之间的距离来表示潜力）
    # robot1_potential_ball1 = -torch.norm(to_ball1_robot1, p=2, dim=-1) / dt
    # robot1_potential_ball2 = -torch.norm(to_ball2_robot1, p=2, dim=-1) / dt
    # robot2_potential_ball1 = -torch.norm(to_ball1_robot2, p=2, dim=-1) / dt
    # robot2_potential_ball2 = -torch.norm(to_ball2_robot2, p=2, dim=-1) / dt

    # 将所有观测合并到一个大观测张量
    obs = torch.cat((
        robot1_obs, robot2_obs, ball1_obs, ball2_obs
    ), dim=-1)

    return obs


@torch.jit.script
def compute_humanoid_observations(obs_buf, root_states, targets, potentials, inv_start_rot, dof_pos, dof_vel,
                                  dof_force, dof_limits_lower, dof_limits_upper, dof_vel_scale,
                                  sensor_force_torques, actions, dt, contact_force_scale, angular_velocity_scale,
                                  basis_vec0, basis_vec1):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float, float, float, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]

    to_target = targets - torso_position
    to_target[:, 2] = 0

    prev_potentials_new = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2)

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position)

    roll = normalize_angle(roll).unsqueeze(-1)
    yaw = normalize_angle(yaw).unsqueeze(-1)
    angle_to_target = normalize_angle(angle_to_target).unsqueeze(-1)
    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs (21), num_dofs (21), 6, num_acts (21)
    obs = torch.cat((torso_position[:, 2].view(-1, 1), vel_loc, angvel_loc * angular_velocity_scale,
                     yaw, roll, angle_to_target, up_proj.unsqueeze(-1), heading_proj.unsqueeze(-1),
                     dof_pos_scaled, dof_vel * dof_vel_scale, dof_force * contact_force_scale,
                     sensor_force_torques.view(-1, 12) * contact_force_scale, actions), dim=-1)

    return obs, potentials, prev_potentials_new, up_vec, heading_vec

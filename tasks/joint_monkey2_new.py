"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Joint Monkey
------------
- Animates degree-of-freedom ranges for a given asset.
- Demonstrates usage of DOF properties and states.
- Demonstrates line drawing utilities to visualize DOF frames (origin and axis).
"""

import math
import numpy as np

from isaacgym import gymapi, gymutil, gymtorch

from scipy.spatial.transform import Rotation as R
import torch
import random

initial_speed_range = (4.8, 6.5)  # 初速度范围 (单位: 米/秒)
tilt_angle_range = (-5.0, 5.0)  # 倾斜角范围 (单位: 度)

origin_robot1_pose = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
origin_robot2_pose = torch.tensor([3.5, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
origin_pingpong_table_pose = torch.tensor([1.7, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
origin_ball1_pose = torch.tensor([0.3, 0.5, 1.3, 0.0, 0.0, 0.0, 1.0])
origin_ball2_pose = torch.tensor([2.9, 0.0, 1.3, 0.0, 0.0, 0.0, 1.0])

def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)

def generate_random_speed_and_tilt_angle(initial_speed_range, tilt_angle_range):
    # ball_velocities = torch.zeros((num_envs, 2, 3))  # 速度 (vx, vy, vz)
    # 初始化球的位置，给定初速度和倾斜角
    speed_1 = random.uniform(*initial_speed_range)
    # print("i=", i, "speed=", speed_1)
    tilt_angle_1 = random.uniform(*tilt_angle_range)
    # print("i=", i, "tilt_angle=", tilt_angle_1)
    velocity_1 = gymapi.Vec3(speed_1 * math.cos(math.radians(tilt_angle_1)), speed_1 * math.sin(math.radians(tilt_angle_1)), 0.0)
    ball_velocities_1 = torch.tensor([velocity_1.x, velocity_1.y, velocity_1.z  # 速度 (vx, vy, vz)
                                        ])
    speed_2 = -random.uniform(*initial_speed_range)
    tilt_angle_2 = random.uniform(*tilt_angle_range)
    velocity_2 = gymapi.Vec3(speed_2 * math.cos(math.radians(tilt_angle_2)), speed_2 * math.sin(math.radians(tilt_angle_2)), 0.0)
    ball_velocities_2 = torch.tensor([velocity_2.x, velocity_2.y, velocity_2.z  # 速度 (vx, vy, vz)
                                            ])
    return ball_velocities_1, ball_velocities_2
    

def reset_ids(env_ids, root_state_tensor, initial_speed_range, tilt_angle_range):

    # for i in range(num_envs):
    # root_state_tensor[env_ids, 0, 0:7] = origin_robot1_pose
    # root_state_tensor[env_ids, 1, 0:7] = origin_robot2_pose
    # root_state_tensor[env_ids, 2, 0:7] = origin_pingpong_table_pose
    root_state_tensor[env_ids, 3, 0:7] = origin_ball1_pose
    root_state_tensor[env_ids, 4, 0:7] = origin_ball2_pose

    # Randomize speed and tilt angle for both balls
    ball_velocities_1, ball_velocities_2 = generate_random_speed_and_tilt_angle(initial_speed_range, tilt_angle_range)


    # Update ball 1 position and velocity (index 4)
    root_state_tensor[env_ids, 3, 0:7] = origin_ball1_pose
    root_state_tensor[env_ids, 3, 7:10] = ball_velocities_1

    # Update ball 2 position and velocity (index 5)
    root_state_tensor[env_ids, 4, 0:7] = origin_ball2_pose
    root_state_tensor[env_ids, 4, 7:10] = ball_velocities_2

    # Refresh actor root state after reset
    # gym.refresh_actor_root_state_tensor(sim)
    # gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_state_tensor))
    all_actor_indices = torch.arange(5 * num_envs, dtype=torch.int32).view(num_envs, 5)
    actor_indices = all_actor_indices[env_ids].flatten()
    gym.set_actor_root_state_tensor_indexed(sim, root_tensor, gymtorch.unwrap_tensor(actor_indices), len(actor_indices))


def check_reset(env_ids, root_state_tensor, depth_threshold=0.05):
    """
    Check if any of the ping pong balls have fallen below the threshold.
    If both balls are below the threshold, return True (needs reset).
    
    :param root_state_tensor: The root state tensor containing actor data (positions and velocities).
    :param depth_threshold: The threshold for the ball to reset (height below this value).
    :return: Boolean value indicating whether a reset is required.
    """
    # for i in range(num_envs):
    # Get the height (z) position of ball 1 and ball 2
    ball1_z = root_state_tensor[env_ids, 3, 2]  # Ball 1 is at index 3
    ball2_z = root_state_tensor[env_ids, 4, 2]  # Ball 2 is at index 4
    # print(ball1_z)
    
    # If both balls are below the threshold, return True (reset needed)
    if ball1_z < depth_threshold and ball2_z < depth_threshold:
        return True
    return False


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


# parse arguments
args = gymutil.parse_arguments(
    description="Joint monkey: Animate degree-of-freedom ranges",
    custom_parameters=[
        {"name": "--asset_id", "type": int, "default": 0, "help": "Asset id (0 - %d)" % (len(asset_descriptors) - 1)},
        {"name": "--speed_scale", "type": float, "default": 1.0, "help": "Animation speed scale"},
        {"name": "--show_axis", "action": "store_true", "help": "Visualize DOF axis"}])

if args.asset_id < 0 or args.asset_id >= len(asset_descriptors):
    print("*** Invalid asset_id specified.  Valid range is 0 to %d" % (len(asset_descriptors) - 1))
    quit()


# initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = dt = 1.0 / 60.0
if args.physics_engine == gymapi.SIM_FLEX:
    pass
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

device = args.sim_device if args.use_gpu_pipeline else 'cpu'

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# # load asset
asset_root = "../../assets"
asset_file = "mjcf/g1_29dof_rev_1_0.urdf"

# asset_root = '/home/jiangnan/unitree_model/unitree_mujoco/unitree_robots/g1/'
# asset_file = 'g1_29dof.xml'

# asset_root = '/home/jiangnan/unitree_model/unitree_ros/robots/g1_description/'
# asset_file = 'g1_29dof.urdf'

# robot asset
asset_options = gymapi.AssetOptions()
# asset_options.fix_base_link = False
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = asset_descriptors[args.asset_id].flip_visual_attachments
asset_options.use_mesh_materials = True

print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)


# pingpong table asset
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = asset_descriptors[args.asset_id].flip_visual_attachments
asset_options.use_mesh_materials = True

asset_pingpong = gym.load_asset(sim, '/home/mj/app/IsaacGymEnvs-main/assets/urdf', 'pingpong_table.urdf', asset_options)

# pingpong ball asset
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False  
asset_options.flip_visual_attachments = asset_descriptors[args.asset_id].flip_visual_attachments
asset_options.use_mesh_materials = True

asset_pingpong_ball = gym.load_asset(sim, '/home/mj/app/IsaacGymEnvs-main/assets/urdf', 'small_ball.urdf', asset_options)


# get array of DOF names
dof_names = gym.get_asset_dof_names(asset)


# get array of DOF properties
dof_props = gym.get_asset_dof_properties(asset)

# create an array of DOF states that will be used to update the actors
num_dofs = gym.get_asset_dof_count(asset) # 29

dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

# get list of DOF types
dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]

# get the position slice of the DOF state array
dof_positions = dof_states['pos']

# get the limit-related slices of the DOF properties array
stiffnesses = dof_props['stiffness']
dampings = dof_props['damping']
armatures = dof_props['armature']
has_limits = dof_props['hasLimits']
lower_limits = dof_props['lower']
upper_limits = dof_props['upper']

# initialize default positions, limits, and speeds (make sure they are in reasonable ranges)
defaults = np.zeros(num_dofs)
speeds = np.zeros(num_dofs)
for i in range(num_dofs):
    if has_limits[i]:
        if dof_types[i] == gymapi.DOF_ROTATION:
            lower_limits[i] = clamp(lower_limits[i], -math.pi, math.pi)
            upper_limits[i] = clamp(upper_limits[i], -math.pi, math.pi)
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
        speeds[i] = args.speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.25 * math.pi, 3.0 * math.pi)
    else:
        speeds[i] = args.speed_scale * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.1, 7.0)

# Print DOF properties
for i in range(num_dofs):
    print("DOF %d" % i)
    print("  Name:     '%s'" % dof_names[i])
    print("  Type:     %s" % gym.get_dof_type_string(dof_types[i]))
    print("  Stiffness:  %r" % stiffnesses[i])
    print("  Damping:  %r" % dampings[i])
    print("  Armature:  %r" % armatures[i])
    print("  Limited?  %r" % has_limits[i])
    if has_limits[i]:
        print("    Lower   %f" % lower_limits[i])
        print("    Upper   %f" % upper_limits[i])

# set up the env grid
num_envs = 16
num_per_row = 4
spacing = 4
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
cam_pos = gymapi.Vec3(0, -2, 1)
cam_target = gymapi.Vec3(0, 0, 1)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# cache useful handles
envs = []
actor_handles = []
# 初始化actor_root_states
actor_root_states = []

print("Creating %d environments" % num_envs)
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 0)
    actor_handles.append(actor_handle)

    # set default DOF positions
    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(3.5, 0.0, 1.0)
    pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 0)
    actor_handles.append(actor_handle)

    # set default DOF positions
    gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

    # add table
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(1.7, 0.0, 0.0)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

    actor_handle = gym.create_actor(env, asset_pingpong, pose, "actor", i, 0)
    actor_handles.append(actor_handle)

    # add ball 1
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.3, 0.5, 1.3)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0) # 没有旋转

    actor_handle = gym.create_actor(env, asset_pingpong_ball, pose, "actor", i, 0)
    actor_handles.append(actor_handle)

    # add ball 2
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(2.9, 0.0, 1.3)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0) # 没有旋转

    actor_handle = gym.create_actor(env, asset_pingpong_ball, pose, "actor", i, 0)
    actor_handles.append(actor_handle)

# set up the balls' initial speed and tilt angle
def set_velocity(root_state_tensor):

    ball_velocities = torch.zeros((num_envs, 2, 3), dtype=torch.float)
        
    # set ball 1&2's initial speed and tilt angle for each env
    for i in range(num_envs):

        ball_velocities_1, ball_velocities_2 = generate_random_speed_and_tilt_angle(initial_speed_range, tilt_angle_range)
        
        ball_velocities[i, 0] = ball_velocities_1
        ball_velocities[i, 1] = ball_velocities_2

    for i in range(num_envs):
        # # 取出根状态数据
        # root_state = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
        
        # 更新ball 1的速度
        root_state_tensor[i, 3, 7:10] = ball_velocities[i, 0]
        
        # 更新ball 2的速度
        root_state_tensor[i, 4, 7:10] = ball_velocities[i, 1]

        print("root_state_tensor[", i, "] = ", root_state_tensor[i])

    # 刷新actor的根状态
    # gym.refresh_actor_root_state_tensor(sim) # 不受控
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_state_tensor))



# joint animation states
ANIM_SEEK_LOWER = 1
ANIM_SEEK_UPPER = 2
ANIM_SEEK_DEFAULT = 3
ANIM_FINISHED = 4

# initialize animation state
anim_state = ANIM_SEEK_LOWER
current_dof = 0
print("Animating DOF %d ('%s')" % (current_dof, dof_names[current_dof]))

num_bodies = gym.get_asset_rigid_body_count(asset)
print('num_bodies', num_bodies) # 39

frame_count = 0

while not gym.query_viewer_has_closed(viewer):

    gym.simulate(sim)

    gym.refresh_actor_root_state_tensor(sim)

    root_tensor = gym.acquire_actor_root_state_tensor(sim)
    root_state_tensor_origin = gymtorch.wrap_tensor(root_tensor)
    # print("root_state_tensor.shape=", root_state_tensor_origin.shape)
    root_state_tensor = root_state_tensor_origin.view(num_envs, -1, 13)
    if frame_count == 0:
        set_velocity(root_state_tensor)

    gym.fetch_results(sim, True)

    # gym.refresh_actor_root_state_tensor(sim)
    
    # if frame_count > 5:
    for i, env in enumerate(envs):
        # 检查球是否掉落
        check = check_reset([i], root_state_tensor)
        # print(f"check = {check}")
        if check:  # [i] 传入单一的env_id
            print(f"Environment {i} needs to be reset.")
            reset_ids([i], root_state_tensor, initial_speed_range, tilt_angle_range)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

    frame_count += 1
    print("frame_count = ", frame_count)


# motion_data = np.load('/home/jiangnan/IsaacGym_Preview_4_Package/IsaacGymEnvs-main/assets/amp/motions/amp_humanoid_cartwheel.npy', allow_pickle=True).item()
# motion_loc = motion_data['root_translation']['arr']
# motion_quat = motion_data['rotation']['arr']
# orig_shape = motion_quat.shape
# motion_euler = R.from_quat(motion_quat.reshape((-1, 4))).as_euler('xyz').reshape((orig_shape[0], -1))
# dof_matching = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 18, 19, 20, 22, 27, 28, 29, 31, 33, 34, 35, 36, 37, 38, 40, 42, 43, 44]
# motion_dof = motion_euler[:, dof_matching]
#
# motion_loc_tensor = torch.from_numpy(motion_loc).float()
# motion_quat_tensor = torch.from_numpy(motion_quat).float()
#
# frame = 0
# while not gym.query_viewer_has_closed(viewer):
#     frame += 1
#     if frame >= motion_dof.shape[0]:
#         frame = 1
#
#     # step the physics
#     gym.simulate(sim)
#     gym.fetch_results(sim, True)
#
#     root_state = torch.cat([motion_loc_tensor[frame], motion_quat_tensor[frame][0], torch.zeros(6)]).reshape(1, -1)
#     gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_state))
#
#     dof_positions[:] = motion_dof[frame, :]
#
#     # speed = speeds[current_dof]
#     #
#     # # animate the dofs
#     # if anim_state == ANIM_SEEK_LOWER:
#     #     dof_positions[current_dof] -= speed * dt
#     #     if dof_positions[current_dof] <= lower_limits[current_dof]:
#     #         dof_positions[current_dof] = lower_limits[current_dof]
#     #         anim_state = ANIM_SEEK_UPPER
#     # elif anim_state == ANIM_SEEK_UPPER:
#     #     dof_positions[current_dof] += speed * dt
#     #     if dof_positions[current_dof] >= upper_limits[current_dof]:
#     #         dof_positions[current_dof] = upper_limits[current_dof]
#     #         anim_state = ANIM_SEEK_DEFAULT
#     # if anim_state == ANIM_SEEK_DEFAULT:
#     #     dof_positions[current_dof] -= speed * dt
#     #     if dof_positions[current_dof] <= defaults[current_dof]:
#     #         dof_positions[current_dof] = defaults[current_dof]
#     #         anim_state = ANIM_FINISHED
#     # elif anim_state == ANIM_FINISHED:
#     #     dof_positions[current_dof] = defaults[current_dof]
#     #     current_dof = (current_dof + 1) % num_dofs
#     #     anim_state = ANIM_SEEK_LOWER
#     #     print("Animating DOF %d ('%s')" % (current_dof, dof_names[current_dof]))
#
#     if args.show_axis:
#         gym.clear_lines(viewer)
#
#     # clone actor state in all of the environments
#     for i in range(num_envs):
#         gym.set_actor_dof_states(envs[i], actor_handles[i], dof_states, gymapi.STATE_POS)
#
#         if args.show_axis:
#             # get the DOF frame (origin and axis)
#             dof_handle = gym.get_actor_dof_handle(envs[i], actor_handles[i], current_dof)
#             frame = gym.get_dof_frame(envs[i], dof_handle)
#
#             # draw a line from DOF origin along the DOF axis
#             p1 = frame.origin
#             p2 = frame.origin + frame.axis * 0.7
#             color = gymapi.Vec3(1.0, 0.0, 0.0)
#             gymutil.draw_line(p1, p2, color, gym, viewer, envs[i])
#
#     # update the viewer
#     gym.step_graphics(sim)
#     gym.draw_viewer(viewer, sim, True)
#
#     # Wait for dt to elapse in real time.
#     # This synchronizes the physics simulation with the rendering rate.
#     gym.sync_frame_time(sim)
#
# print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

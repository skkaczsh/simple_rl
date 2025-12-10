# Copyright (c) 2025, Go2 SAC LiDAR Expert Training Project.
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for Go2 robot with LiDAR-based voxel observations.

This config defines the complete RL environment for training a Go2 quadruped
to navigate using voxelized LiDAR perception.
"""

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


@configclass
class Go2LidarEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for Go2 with LiDAR voxel observations."""
    
    # ========================
    # Environment Settings
    # ========================
    episode_length_s = 20.0  # 20 seconds per episode
    decimation = 4  # Control frequency = 50Hz (200Hz physics / 4)
    num_actions = 12  # Go2 has 12 actuated joints
    
    # Observation space (dict-based for skrl)
    # Will be defined in environment class
    
    # Scene settings
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16,  # Number of parallel environments
        env_spacing=2.5,  # Spacing between environments (meters)
        replicate_physics=True
    )
    
    # ========================
    # Simulation
    # ========================
    sim = sim_utils.SimulationCfg(
        dt=1.0 / 200.0,  # 200Hz physics
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    # ========================
    # Robot (Go2 Quadruped)
    # ========================
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.4),
            joint_pos={
                ".*L_hip_joint": 0.1,
                ".*R_hip_joint": -0.1,
                "F[L,R]_thigh_joint": 0.8,
                "R[L,R]_thigh_joint": 1.0,
                ".*_calf_joint": -1.5,
            },
        ),
    )
    
    # ========================
    # LiDAR Sensor
    # ========================
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=patterns.BpearlPatternCfg(
            horizontal_fov=360.0,  # Full 360-degree scan
            vertical_ray_angles=[-25.0, -15.0, 0.0, 15.0, 25.0],  # 5 vertical layers
            horizontal_res=1.0,  # 1-degree horizontal resolution
        ),
        max_distance=10.0,
        debug_vis=False,  # Set to True for visualization
    )
    
    # ========================
    # Terrain
    # ========================
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",  # Start with flat terrain
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    
    # ========================
    # Voxelization Parameters  
    # ========================
    voxel_grid_dims = (64, 48, 12)  # (W, H, D)
    voxel_size = 0.1  # 10cm per voxel
    voxel_range_x = (-3.2, 3.2)  # 6.4m forward/backward
    voxel_range_y = (-2.4, 2.4)  # 4.8m left/right
    voxel_range_z = (-0.6, 0.6)  # 1.2m up/down
    
    # ========================
    # Reward Weights
    # (For flat terrain walking expert)
    # ========================
    
    # Tracking rewards
    lin_vel_reward_scale = 1.5
    ang_vel_reward_scale = 0.5
    
    # Penalties
    lin_vel_z_penalty_scale = -2.0
    ang_vel_xy_penalty_scale = -0.05
    joint_torque_penalty_scale = -0.0002
    joint_accel_penalty_scale = -2.5e-7
    action_rate_penalty_scale = -0.01
    flat_orientation_penalty_scale = -5.0
    
    # Bonus
    alive_reward_scale = 1.0
    
    # Termination
    base_height_threshold = 0.25  # Terminate if base drops below this

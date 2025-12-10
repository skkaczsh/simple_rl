
# Copyright (c) 2025, Go2 SAC LiDAR Expert Training Project.

"""
Go2 Quadruped Environment with LiDAR Voxel Observations (DirectRLEnv Implementation)

Design Philosophy:
- Uses DirectRLEnv for simplicity and rapid iteration
- Modular design with clear migration path to ManagerBasedRLEnv
- Each reward/termination/observation component is isolated
- Function signatures match ManagerBased conventions for easy migration

Migration Guide:
- All _reward_* functions can become RewardTermCfg entries
- All _termination_* functions can become TerminationTermCfg entries
- _get_observations() logic can become ObservationTermCfg entries
"""

import math
import torch
from typing import Dict, Tuple

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import DCMotorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCaster, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# Import our custom voxelizer
from voxelizer import Voxelizer

# Import Go2 configuration
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


@configclass
class Go2LidarDirectEnvCfg(DirectRLEnvCfg):
    """Configuration for Go2 LiDAR Direct RL Environment."""
    
    # ========== Basic Settings ==========
    episode_length_s = 20.0
    decimation = 4  # 50Hz control (200Hz physics / 4)
    num_actions = 12

    # Flattened observation space (critical for skrl memory management)
    # voxels: 64*48*12 = 36,864 + state: 51 = 36,915 total
    observation_space: gym.spaces.Box = gym.spaces.Box(
        low=-float("inf"), 
        high=float("inf"), 
        shape=(36915,), 
        dtype="float32"
    )

    
    action_space: gym.spaces.Box = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype="float32")
    
    # ========== Scene Configuration ==========
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4,
        env_spacing=2.5,
        replicate_physics=True
    )
    
    # ========== Simulation ==========
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / 200.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    # ========== Robot ==========
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",  # 修复：使用DirectRLEnv标准格式
        actuators={
            "base_legs": DCMotorCfg(  # 修复：使用DCMotor（官方Go2标准）
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit=23.5,
                saturation_effort=23.5,
                velocity_limit=30.0,
                stiffness=25.0,
                damping=0.5,
                friction=0.0,
            ),
        }
    )
    # ========== Terrain ==========
    terrain_cfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    
    # ========== LiDAR Sensor ==========
    lidar_cfg: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=patterns.LidarPatternCfg(
            channels=64,
            vertical_fov_range=(-25.0, 15.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=1.0,
        ),
        max_distance=10.0,
        debug_vis=False,
    )
    
    # ========== Voxelization Parameters ==========
    voxel_grid_dims = (64, 48, 12)  # (W, H, D)
    voxel_size = 0.1  # 10cm per voxel
    voxel_range_x = (-3.2, 3.2)  # 6.4m
    voxel_range_y = (-2.4, 2.4)  # 4.8m
    voxel_range_z = (-0.6, 0.6)  # 1.2m
    
    # ========== Reward Weights ==========
    # MIGRATION NOTE: These will become RewardTermCfg.weight in ManagerBased
    reward_weights = {
        "lin_vel_tracking": 1.5,
        "ang_vel_tracking": 0.5,
        "lin_vel_z_penalty": -0.5,
        "ang_vel_xy_penalty": -0.05,
        "joint_torque_penalty": -0.0002,
        "joint_accel_penalty": -2.5e-7,
        "action_rate_penalty": -0.01,
        "orientation_penalty": -1.0,
        "alive_bonus": 1.0,
    }
    
    # ========== Command Ranges ==========
    # MIGRATION NOTE: These will become CommandTermCfg in ManagerBased
    command_ranges = {
        "lin_vel_x": (0.5, 1.5),
        "lin_vel_y": (-0.5, 0.5),
        "ang_vel_z": (-1.0, 1.0),
    }
    
    # ========== Termination Conditions ==========
    base_height_threshold = 0.25  # Terminate if base drops below this


class Go2LidarDirectEnv(DirectRLEnv):
    """
    Go2 Quadruped with LiDAR Voxel Observations (DirectRLEnv).
    
    Architecture:
        Observations: Dict["voxels": (B,64,48,12), "robot_state": (B,51)]
        Actions: (B, 12) joint position commands
        
    Design for Migration:
        - All reward computation is modularized in _reward_* methods
        - All termination logic is in _termination_* methods
        - Easy to convert to ManagerBased reward/termination terms
    """
    
    cfg: Go2LidarDirectEnvCfg
    
    def __init__(self, cfg: Go2LidarDirectEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment."""
        super().__init__(cfg, render_mode, **kwargs)
        
        # ===== Initialize Voxelizer =====
        self.voxelizer = Voxelizer(
            grid_dims=cfg.voxel_grid_dims,
            voxel_size=cfg.voxel_size,
            range_x=cfg.voxel_range_x,
            range_y=cfg.voxel_range_y,
            range_z=cfg.voxel_range_z,
            device=self.device,
        )
        
        # ===== Velocity Commands =====
        # MIGRATION NOTE: Will become CommandManager in ManagerBased
        self.cmd_lin_vel_x = torch.zeros(self.num_envs, device=self.device)
        self.cmd_lin_vel_y = torch.zeros(self.num_envs, device=self.device)
        self.cmd_ang_vel_z = torch.zeros(self.num_envs, device=self.device)
        
        # ===== Action History (for penalties) =====
        self.previous_actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self.previous_joint_vel = torch.zeros(self.num_envs, 12, device=self.device)
        
        print(f"[Go2LidarDirectEnv] Initialized with {self.num_envs} environments")
        print(f"  - Voxel grid: {cfg.voxel_grid_dims}")
        print(f"  - Device: {self.device}")

               # ===== Recovery Curriculum =====
        # Counter for fallen steps - give agents time to recover before reset
        self.fallen_steps_counter = torch.zeros(self.num_envs, device=self.device)
        self.recovery_steps = 30  # Give 300 steps to recover before reset (increased from 20)
    
    # =========================================================================
    # DirectRLEnv Interface (Required Methods)
    # =========================================================================
    
    def _setup_scene(self):
        """Setup the scene entities.
        
        MIGRATION NOTE: Scene setup is similar in both DirectRLEnv and ManagerBased
        """
        # Add robot
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot
        
        # Add LiDAR sensor
        self._lidar = RayCaster(self.cfg.lidar_cfg)
        self.scene.sensors["lidar"] = self._lidar
        
        # Add terrain
        self.cfg.terrain_cfg.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain_cfg.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain_cfg.class_type(self.cfg.terrain_cfg)
        
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain_cfg.prim_path])
        
        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions before physics step."""
        # Store history for penalties
        # CRITICAL: Update action buffer with new actions
        self.actions = actions
        self.previous_actions = self.actions.clone()
        self.previous_joint_vel = self._robot.data.joint_vel.clone()
    
    def _apply_action(self) -> None:
        """Apply actions to the robot with proper action scaling.

        Scale actions from [-1, 1] to joint position targets around default pose.
        This is CRITICAL for the robot to move properly!
        """
        # Action scaling: map [-1, 1] to joint positions around default stance
        # INCREASED from 0.5 to 1.5 radians to allow more dynamic movement
        # ±1.5 rad (±86°) allows full range of motion for locomotion
        action_scale = 0.5

        # Compute target joint positions: default_pos + scaled_action
        joint_pos_target = self._robot.data.default_joint_pos + self.actions * action_scale

        # Apply targets to robot
        self._robot.set_joint_position_target(joint_pos_target)
    
    def _get_observations(self) -> dict:
        """Compute observations (flattened for skrl memory management).
        
        Returns:
            Dict with "policy" key containing flattened tensor:
            - Shape: (num_envs, 36915)
            - Format: [voxels_flat (36864) + robot_state (51)]
            
        MIGRATION NOTE: Flattened format is required for skrl to avoid
        memory allocation errors with large Dict observation spaces.
        The dict wrapper is needed for IsaacLabWrapper compatibility.
        """
        # Voxelize LiDAR point cloud
        voxels = self._compute_voxel_observations()
        
        # Compute robot proprioceptive state
        robot_state = self._compute_robot_state()
        
        # Flatten observations for skrl (critical for memory management)
        # voxels: (num_envs, 64, 48, 12) -> flatten to (num_envs, 36864)
        # robot_state: (num_envs, 51)
        # combined: (num_envs, 36915)
        voxels_flat = voxels.view(self.num_envs, -1)
        obs = torch.cat([voxels_flat, robot_state], dim=-1)
        
        # Return in dict format for skrl IsaacLabWrapper
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards.
        
        MIGRATION NOTE: Each _reward_* method can become a RewardTermCfg:
            RewardTermCfg(func=_reward_lin_vel_tracking, weight=1.5)
        """
        total_reward = torch.zeros(self.num_envs, device=self.device)
        
        # Tracking rewards
        total_reward += self.cfg.reward_weights["lin_vel_tracking"] * self._reward_lin_vel_tracking()
        total_reward += self.cfg.reward_weights["ang_vel_tracking"] * self._reward_ang_vel_tracking()
        
        # Penalties
        total_reward += self.cfg.reward_weights["lin_vel_z_penalty"] * self._reward_lin_vel_z_penalty()
        total_reward += self.cfg.reward_weights["ang_vel_xy_penalty"] * self._reward_ang_vel_xy_penalty()
        total_reward += self.cfg.reward_weights["joint_torque_penalty"] * self._reward_joint_torque_penalty()
        total_reward += self.cfg.reward_weights["joint_accel_penalty"] * self._reward_joint_accel_penalty()
        total_reward += self.cfg.reward_weights["action_rate_penalty"] * self._reward_action_rate_penalty()
        total_reward += self.cfg.reward_weights["orientation_penalty"] * self._reward_orientation_penalty()
        
        # Alive bonus (with recovery state detection)
        # During recovery (fallen state), no alive bonus is given
        # Only reward survival when standing normally or successfully recovered
        base_height = self._robot.data.root_pos_w[:, 2]
        is_fallen = base_height < 0.25
        
        # Check which envs are in recovery state (was fallen, now recovering)
        is_in_recovery = self.fallen_steps_counter > 0
        
        # Give alive bonus only if NOT in recovery state
        # This means:
        # 1. Standing normally (never fallen) - get alive bonus
        # 2. Successfully recovered (was fallen but now standing) - get alive bonus  
        # 3. Currently fallen/recovering - NO alive bonus
        has_recovered = is_in_recovery & ~is_fallen
        standing_normal = ~is_in_recovery & ~is_fallen
        eligible_for_alive_bonus = has_recovered | standing_normal
        
        alive_bonus = self.cfg.reward_weights["alive_bonus"] * eligible_for_alive_bonus.float()
        total_reward += alive_bonus

        # Fallen penalty (continuous penalty while base is low)
        fallen_penalty = -1.0 * is_fallen.float()
        total_reward += fallen_penalty

        return total_reward
    
    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute termination and truncation with recovery curriculum.

        Instead of immediately terminating when fallen, give the agent
        a chance to recover. This enables learning "get up" behaviors.
        """
        # Timeout
        time_out = self._termination_timeout()

        # Check if fallen (base height < 0.25m)
        base_height = self._robot.data.root_pos_w[:, 2]
        is_fallen = base_height < 0.25

        # Recovery curriculum logic:
        # - If fallen, increment counter
        # - If standing (recovered), reset counter
        self.fallen_steps_counter[is_fallen] += 1
        self.fallen_steps_counter[~is_fallen] = 0

        # Only terminate if failed to recover within recovery_steps
        give_up = self.fallen_steps_counter > self.recovery_steps

        # Reset counter for terminated envs
        self.fallen_steps_counter[give_up] = 0

        terminated = give_up
        truncated = time_out

        return terminated, truncated
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        print(f"[DEBUG] Resetting envs: {env_ids}")
        print(f"[DEBUG] Resetting envs: {env_ids}")
        """Reset specified environments.

        MIGRATION NOTE: In ManagerBased, randomization becomes EventManager
        """
        # Safety check: super().__init__() may call this before our __init__() completes
        if not hasattr(self, 'fallen_steps_counter'):
            # During parent __init__, delegate to parent and return
            super()._reset_idx(env_ids)
            return

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        
        # Reset robot to default state
        num_resets = len(env_ids)

        # self._robot.reset(env_ids)  # Explicit write below
        
        # --- Explicit Reset Logic ---
        # 1. Reset Root State
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        self._robot.write_root_state_to_sim(root_state, env_ids)
        
        # STRONG randomization for diverse initial states
        # Randomize initial joint positions: default ± 30% + absolute offset
        default_joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        # Multiplicative noise: ±30%
        joint_mult_noise = (torch.rand_like(default_joint_pos) - 0.5) * 0.6  # ±30%
        # Additive noise: ±0.5 radians
        joint_add_noise = (torch.rand_like(default_joint_pos) - 0.5) * 1.0  # ±0.5 rad
        self._robot.data.joint_pos[env_ids] = default_joint_pos * (1 + joint_mult_noise) + joint_add_noise

        # Randomize initial joint velocities: ±2 rad/s for dynamic starts
        vel_noise = (torch.rand(num_resets, 12, device=self.device) - 0.5) * 4.0  # ±2 rad/s
        self._robot.data.joint_vel[env_ids] = vel_noise
        
        # Apply joint state to simulation immediately
        self._robot.write_joint_state_to_sim(self._robot.data.joint_pos[env_ids], self._robot.data.joint_vel[env_ids], env_ids=env_ids)

        # Randomize velocity commands
        self._randomize_commands(env_ids)
        
        # Reset action buffers
        self.previous_actions[env_ids] = 0.0
        self.previous_joint_vel[env_ids] = 0.0
        
        # Reset fallen counter for reset environments
        self.fallen_steps_counter[env_ids] = 0

        # Call parent reset
        super()._reset_idx(env_ids)
    
    # =========================================================================
    # Observation Computation (Modular for Migration)
    # =========================================================================
    
    def _compute_voxel_observations(self) -> torch.Tensor:
        """Compute voxelized LiDAR observations.
        
        Returns:
            (B, 64, 48, 12) float32 voxel grid
            
        MIGRATION: ObservationTermCfg(func=_compute_voxel_observations)
        """
        lidar_points = self._lidar.data.ray_hits_w  # (B, N, 3)
        robot_pos = self._robot.data.root_pos_w  # (B, 3)
        robot_quat = self._robot.data.root_quat_w  # (B, 4)
        
        voxels = self.voxelizer(lidar_points, robot_pos, robot_quat)
        return voxels.float()
    
    def _compute_robot_state(self) -> torch.Tensor:
        """Compute proprioceptive robot state.
        
        Returns:
            (B, 51) state vector:
                - base_lin_vel (3)
                - base_ang_vel (3)
                - projected_gravity (3)
                - joint_pos (12)
                - joint_vel (12)
                - cmd_lin_vel (3)
                - cmd_ang_vel (3)
                - previous_actions (12)
                
        MIGRATION: ObservationTermCfg(func=_compute_robot_state)
        """
        base_lin_vel = self._robot.data.root_lin_vel_b  # (B, 3)
        base_ang_vel = self._robot.data.root_ang_vel_b  # (B, 3)
        projected_gravity = self._robot.data.projected_gravity_b  # (B, 3)
        joint_pos = self._robot.data.joint_pos  # (B, 12)
        joint_vel = self._robot.data.joint_vel  # (B, 12)
        
        # Commands
        cmd_lin_vel = torch.stack([
            self.cmd_lin_vel_x,
            self.cmd_lin_vel_y,
            torch.zeros_like(self.cmd_lin_vel_x)
        ], dim=-1)  # (B, 3)
        
        cmd_ang_vel = torch.stack([
            torch.zeros_like(self.cmd_ang_vel_z),
            torch.zeros_like(self.cmd_ang_vel_z),
            self.cmd_ang_vel_z
        ], dim=-1)  # (B, 3)
        
        # Concatenate all
        robot_state = torch.cat([
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            joint_pos,
            joint_vel,
            cmd_lin_vel,
            cmd_ang_vel,
            self.previous_actions,
        ], dim=-1)
        
        return robot_state
    
    # =========================================================================
    # Modular Reward Functions (Easy Migration to RewardTermCfg)
    # =========================================================================
    
    def _reward_lin_vel_tracking(self) -> torch.Tensor:
        """Reward for tracking linear velocity commands.
        
        MIGRATION: RewardTermCfg(func=_reward_lin_vel_tracking, weight=1.5)
        """
        lin_vel = self._robot.data.root_lin_vel_b[:, :2]  # (B, 2) XY only
        target_vel = torch.stack([self.cmd_lin_vel_x, self.cmd_lin_vel_y], dim=-1)
        
        error = torch.sum(torch.square(lin_vel - target_vel), dim=-1)
        reward = torch.exp(-error / 0.25)
        return reward
    
    def _reward_ang_vel_tracking(self) -> torch.Tensor:
        """Reward for tracking angular velocity commands.
        
        MIGRATION: RewardTermCfg(func=_reward_ang_vel_tracking, weight=0.5)
        """
        ang_vel_z = self._robot.data.root_ang_vel_b[:, 2]
        error = torch.square(ang_vel_z - self.cmd_ang_vel_z)
        reward = torch.exp(-error / 0.25)
        return reward
    
    def _reward_lin_vel_z_penalty(self) -> torch.Tensor:
        """Penalty for vertical motion.
        
        MIGRATION: RewardTermCfg(func=_reward_lin_vel_z_penalty, weight=-2.0)
        """
        lin_vel_z = self._robot.data.root_lin_vel_b[:, 2]
        return torch.square(lin_vel_z)
    
    def _reward_ang_vel_xy_penalty(self) -> torch.Tensor:
        """Penalty for roll/pitch angular velocity.
        
        MIGRATION: RewardTermCfg(func=_reward_ang_vel_xy_penalty, weight=-0.05)
        """
        ang_vel_xy = self._robot.data.root_ang_vel_b[:, :2]
        return torch.sum(torch.square(ang_vel_xy), dim=-1)
    
    def _reward_joint_torque_penalty(self) -> torch.Tensor:
        """Penalty for high joint torques (energy efficiency).
        
        MIGRATION: RewardTermCfg(func=_reward_joint_torque_penalty, weight=-0.0002)
        """
        # Approximate torque by action magnitude
        return torch.sum(torch.square(self.actions), dim=-1)
    
    def _reward_joint_accel_penalty(self) -> torch.Tensor:
        """Penalty for joint accelerations (smoothness).
        
        MIGRATION: RewardTermCfg(func=_reward_joint_accel_penalty, weight=-2.5e-7)
        """
        joint_accel = (self._robot.data.joint_vel - self.previous_joint_vel) / self.step_dt
        return torch.sum(torch.square(joint_accel), dim=-1)
    
    def _reward_action_rate_penalty(self) -> torch.Tensor:
        """Penalty for action rate changes (smoothness).
        
        MIGRATION: RewardTermCfg(func=_reward_action_rate_penalty, weight=-0.01)
        """
        action_rate = self.actions - self.previous_actions
        return torch.sum(torch.square(action_rate), dim=-1)
    
    def _reward_orientation_penalty(self) -> torch.Tensor:
        """Penalty for non-flat orientation (keep robot upright).
        
        MIGRATION: RewardTermCfg(func=_reward_orientation_penalty, weight=-5.0)
        """
        projected_gravity = self._robot.data.projected_gravity_b
        return torch.sum(torch.square(projected_gravity[:, :2]), dim=-1)
    
    # =========================================================================
    # Modular Termination Functions (Easy Migration to TerminationTermCfg)
    # =========================================================================
    
    def _termination_timeout(self) -> torch.Tensor:
        """Episode timeout termination.
        
        MIGRATION: TerminationTermCfg(func=_termination_timeout)
        """
        return self.episode_length_buf >= self.max_episode_length - 1
    
    def _termination_base_contact(self) -> torch.Tensor:
        """Base contact (fell over) termination.
        
        MIGRATION: TerminationTermCfg(func=_termination_base_contact)
        """
        base_height = self._robot.data.root_pos_w[:, 2]
        return base_height < self.cfg.base_height_threshold
    
    # =========================================================================
    # Helper Functions
    # =========================================================================
    
    def _randomize_commands(self, env_ids: torch.Tensor):
        """Randomize velocity commands for reset environments.
        
        MIGRATION NOTE: In ManagerBased, becomes EventTermCfg for command randomization
        """
        num_resets = len(env_ids)
        
        # Linear velocity X: [0.5, 1.5]
        self.cmd_lin_vel_x[env_ids] = (
            torch.rand(num_resets, device=self.device) 
            * (self.cfg.command_ranges["lin_vel_x"][1] - self.cfg.command_ranges["lin_vel_x"][0])
            + self.cfg.command_ranges["lin_vel_x"][0]
        )
        
        # Linear velocity Y: [-0.5, 0.5]
        self.cmd_lin_vel_y[env_ids] = (
            torch.rand(num_resets, device=self.device) 
            * (self.cfg.command_ranges["lin_vel_y"][1] - self.cfg.command_ranges["lin_vel_y"][0])
            + self.cfg.command_ranges["lin_vel_y"][0]
        )
        
        # Angular velocity Z: [-1.0, 1.0]
        self.cmd_ang_vel_z[env_ids] = (
            torch.rand(num_resets, device=self.device) 
            * (self.cfg.command_ranges["ang_vel_z"][1] - self.cfg.command_ranges["ang_vel_z"][0])
            + self.cfg.command_ranges["ang_vel_z"][0]
        )

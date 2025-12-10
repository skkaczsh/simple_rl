# Copyright (c) 2025, Go2 SAC LiDAR Expert Training Project.
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 SAC LiDAR Expert Training Package."""

import gymnasium as gym

from .go2_lidar_env import Go2LidarRLEnv, Go2LidarRLEnvCfg

# Register environment with gymnasium
gym.register(
    id="Isaac-Go2-Lidar-SAC-v0",
    entry_point="go2_sac_lidar_clean.go2_lidar_env:Go2LidarRLEnv",
    disable_env_checker=True,
)

__all__ = ["Go2LidarRLEnv", "Go2LidarRLEnvCfg"]

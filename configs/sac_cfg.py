# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SAC configuration for Go2 voxel training."""

from dataclasses import MISSING

from configs.go2_voxel_env import Go2VoxelEnvCfg

from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab_tasks.utils.wrappers import RslRlVecEnvCfgWrapper, RslRlVecEnvCfg

import omni.isaac.lab_tasks.utils.rsl_rl as rsl_rl_utils


@configclass
class Go2VoxelSACCfg(RslRlVecEnvCfg):
    """Configuration for Go2 voxel SAC training."""
    
    # RL Environment configuration
    env_cfg = Go2VoxelEnvCfg()
    
    # SAC Algorithm configuration
    class algorithm_cfg:
        # Learning rate
        actor_lr = 3e-4
        critic_lr = 3e-4
        alpha_lr = 3e-4
        
        # Training hyperparameters
        gamma = 0.99
        tau = 0.005
        alpha = 0.2
        target_entropy = -12.0  # For Go2 12-dimensional action space
        
        # Learning policy
        num_learning_epochs = 5
        num_mini_batches = 4
        batch_size = 1024
        
        # Experience collection
        num_envs = 4096
        max_episode_length = 500
        random_timesteps = 5000
        learning_starts = 5000
        
        # Replay buffer
        replay_buffer_size = 200_000  # VRAM optimized for L20
        prioritized_replay = False
        
        # Gradient clipping
        max_grad_norm = 1.0
        
        # Network architecture
        mlp_hidden_dims = (256, 256, 256)
        transformer_dim = 256
        transformer_layers = 2
        voxel_feature_dim = 128
        
        # Device configuration
        device = "cuda:0"
        
        # Checkpointing
        save_interval = 50
        resume = False
        load_run = ""
        load_checkpoint = "model_.*.pt"
    
    # Logging configuration
    class logging_cfg:
        experiment_name = "go2_voxel_sac"
        run_name = ""
        logger = "tensorboard"  # "tensorboard", "wandb", "neptune"
        tags = ["go2", "voxel", "sac", "locomotion"]
        
    # SAC runner configuration
    class runner_cfg:
        # Experience collection
        num_steps_per_env = 500
        max_iterations = 3000
        
        # Device
        device = "cuda:0"
        
        # Logging interval
        log_interval = 50
        save_interval = 50
        
        # Evaluation
        run_name = ""
        resume = False
        
    # Environment and agent configuration
    env: DirectRLEnvCfg = Go2VoxelEnvCfg()
    agent: algorithm_cfg = algorithm_cfg()
    logging: logging_cfg = logging_cfg()
    
    # Direct runners configuration
    class runners:
        train_cfg = runner_cfg()
        play_cfg = runner_cfg()
    
    # Checkpointing configuration
    class checkpointing_cfg:
        enabled = True
        save_after_timesteps = 10000
        save_after_iters = 100
        keep_only_latest_checkpoints = True
        max_num_latest_checkpoints = 3
        clear_buffer = True
        """
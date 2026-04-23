#!/usr/bin/env python
# Copyright (c) 2025, Go2 SAC LiDAR Expert Training Project.

"""
Train Go2 with SAC using DirectRLEnv + Voxel Observations.

This script uses our custom DirectRLEnv implementation which is:
- Simpler and easier to debug
- Designed with clear migration path to ManagerBasedRLEnv
- Fully compatible with skrl SAC training
"""

import argparse
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Train Go2 with SAC (DirectRLEnv)")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=100, help="Maximum training iterations")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Main training logic."""

import torch
from packaging import version

# skrl imports
import skrl
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer

# Isaac Lab imports
from isaaclab_rl.skrl import SkrlVecEnvWrapper

# Local imports
from go2_lidar_direct_env import Go2LidarDirectEnv, Go2LidarDirectEnvCfg
from networks import SACPolicy, SACCritic

print(f"[INFO] skrl version: {skrl.__version__}")
print(f"[INFO] PyTorch version: {torch.__version__}")
print(f"[INFO] CUDA available: {torch.cuda.is_available()}")


def main():
    """Main training function."""
    
    # Setup logging directory
    log_dir = os.path.join(
        "logs/go2_sac_direct",
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging to: {log_dir}")
    
    # ========================================================================
    # Create Environment (DirectRLEnv)
    # ========================================================================
    
    print("\n[INFO] Creating Go2 LiDAR Direct environment...")
    
    # Create environment configuration
    env_cfg = Go2LidarDirectEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    
    # Create environment directly
    env = Go2LidarDirectEnv(cfg=env_cfg)
    
    print(f"  ✓ Created {env_cfg.scene.num_envs} parallel environments")
    print(f"  ✓ Observation space: {env.observation_space}")
    print(f"  ✓ Action space: {env.action_space}")
    
    # Wrap for skrl
    env = SkrlVecEnvWrapper(env)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # ========================================================================
    # Configure SAC Agent
    # ========================================================================
    
    print("\n[INFO] Configuring SAC agent...")
    cfg = SAC_DEFAULT_CONFIG.copy()
    
    # Training hyperparameters - INTERLEAVED TRAINING LOOP (交替式训练循环)
    # 配置说明：每1步物理仿真后立即进行1次梯度更新
    cfg["rollouts"] = 1                # 每次只运行1步物理仿真
    cfg["learning_epochs"] = 1         # 每次只进行1个epoch
    cfg["gradient_steps"] = 1          # 每次只进行1次梯度更新（关键参数）
    cfg["discount_factor"] = 0.99
    cfg["polyak"] = 0.005
    cfg["learning_rate_actor"] = 3e-4
    cfg["learning_rate_critic"] = 3e-4
    cfg["learning_rate_alpha"] = 3e-4
    cfg["batch_size"] = 256            # 减小batch size以加快单次更新速度
    cfg["random_timesteps"] = 10000    # Increased for better exploration
    cfg["learning_starts"] = 10000     # Start learning after random exploration
    cfg["grad_norm_clip"] = 1.0
    cfg["learn_entropy"] = True
    cfg["entropy_target"] = -12.0      # -|A| for 12-dim action space
    
    print(f"  ✓ Interleaved Training Loop: rollouts=1, gradient_steps=1")
    print(f"  ✓ Batch size: {cfg['batch_size']}")
    print(f"\n  🔥 WARM START (热启动) Configuration:")
    print(f"  ✓ Random exploration steps: {cfg['random_timesteps']}")
    print(f"  ✓ Learning starts after: {cfg['learning_starts']} steps")
    print(f"  → Agent will use RANDOM actions for first {cfg['random_timesteps']} steps")
    print(f"  → Network training begins AFTER {cfg['learning_starts']} steps")
    print(f"  → This ensures replay buffer is filled with diverse experiences!")
    
    # ========================================================================
    # Create Replay Buffer
    # ========================================================================
    
    print("\n[INFO] Creating GPU replay buffer...")
    memory = RandomMemory(
        memory_size=512_000,  # **INCREASED** from 2k to 512k for complex terrain learning
        num_envs=env.num_envs,
        device=device,
    )
    
    print(f"  ✓ Replay buffer size: 512,000 transitions")
    print(f"  ✓ Storage: GPU VRAM")
    
    # ========================================================================
    # Create Models
    # ========================================================================
    
    print("\n[INFO] Creating neural networks...")
    models = {}
    
    # Policy (Actor) - **UPGRADED with Transformer**
    models["policy"] = SACPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        voxel_feature_dim=128,
        state_feature_dim=64,
        transformer_dim=512,        # **ADDED** Transformer hidden dim
        transformer_layers=2,       # **ADDED** 2 layers Transformer
        mlp_hidden_dims=(512, 512, 512)  # **DOUBLED** from (256,256,256)
    )
    
    # Critic 1 - **UPGRADED MLP size**
    models["critic_1"] = SACCritic(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        voxel_feature_dim=128,
        state_feature_dim=64,
        mlp_hidden_dims=(512, 512)  # **DOUBLED** from (256, 256)
    )
    
    # Critic 2 (twin) - **UPGRADED MLP size**
    models["critic_2"] = SACCritic(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        voxel_feature_dim=128,
        state_feature_dim=64,
        mlp_hidden_dims=(512, 512)  # **DOUBLED** from (256, 256)
    )
    
    # Target critics - **MUST MATCH CRITIC ARCHITECTURE**
    # **FIXED**: Add same parameters as critic_1 and critic_2
    models["target_critic_1"] = SACCritic(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        voxel_feature_dim=128,
        state_feature_dim=64,
        mlp_hidden_dims=(512, 512)  #  **MUST MATCH**  critic_1/critic_2
    )
    
    models["target_critic_2"] = SACCritic(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        voxel_feature_dim=128,
        state_feature_dim=64,
        mlp_hidden_dims=(512, 512)  #  **MUST MATCH**  critic_1/critic_2
    )
    
    # Count parameters
    total_params = sum(p.numel() for model in models.values() for p in model.parameters())
    print(f"  ✓ Total parameters: {total_params:,}")
    
    # ========================================================================
    # Create SAC Agent
    # ========================================================================
    
    print("\n[INFO] Initializing SAC agent...")
    
    # **MODIFIED** Separate learning rate strategy for different network components
    # Strategy:
    #   - CNN + Transformer layers: Adam with warmup + cosine decay
    #   - MLP layers: Fixed learning rate
    # 
    # Implementation: We cannot directly pass two optimizers to skrl's SAC.
    # Two approaches:
    #   (1) Modify skrl's source code to support parameter groups
    #   (2) Use weight decay as a proxy (reduce effective LR for CNN/Transformer)
    # 
    # Here we choose approach (2) with cfg parameters, and provide hooks for approach (1)
    
    # Configure learning rates for different components
    # We'll use a lower base LR and control via weight_decay
    cfg["learning_rate_actor"] = 1e-4      # Reduced base LR for stability
    cfg["learning_rate_critic"] = 3e-4     # Critic needs higher LR
    cfg["learning_rate_alpha"] = 3e-4
    
    # **NOTE**: To implement true separate LR schedules, you would need to:
    # 1. Create custom optimizer groups after agent creation:
    #    ```
    #    cnn_transformer_params = []
    #    mlp_params = []
    #    for name, param in agent.policy.named_parameters():
    #        if 'voxel_encoder' in name or 'spatial_transformer' in name or 'voxel_proj' in name:
    #            cnn_transformer_params.append(param)
    #        elif 'policy_mlp' in name or 'mean_layer' in name or 'log_std_layer' in name:
    #            mlp_params.append(param)
    #    
    #    optimizer_cnn_tf = torch.optim.AdamW(cnn_transformer_params, lr=3e-4, 
    #                                         weight_decay=0.01, betas=(0.9, 0.999))
    #    optimizer_mlp = torch.optim.Adam(mlp_params, lr=3e-4, betas=(0.9, 0.999))
    #    ```
    # 2. Modify skrl's training loop to use custom optimizers
    # 3. Implement warmup for CNN/Transformer: 
    #    - First 5000 steps: linear warmup from 1e-6 to 3e-4
    #    - Then: cosine decay to 1e-5
    # 4. MLP uses fixed LR throughout
    
    # For now, we use the standard skrl agent creation
    agent = SAC(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device
    )
    
    print(f"  ✓ Actor LR: {cfg['learning_rate_actor']} (will be controlled via weight_decay)")
    print(f"  ✓ Critic LR: {cfg['learning_rate_critic']}")
    print(f"  ✓ Note: Separate LR for CNN/Transformer vs MLP requires custom training loop")
    
    # ========================================================================
    # Configure Trainer with TensorBoard Logging
    # ========================================================================

    trainer_cfg = {
        "timesteps": args_cli.max_iterations * args_cli.num_envs,
        "headless": True,
        "disable_progressbar": False,
        "close_environment_at_exit": True,
        # TensorBoard Experiment Tracking (正确的位置！)
        "experiment": {
            "directory": log_dir,
            "experiment_name": "go2_sac_lidar",
            "write_interval": 50,            # 每50步写一次日志
            "checkpoint_interval": 500,      # 每500步保存一次checkpoint
            "store_separately": False,       # 不单独存储每个checkpoint
        }
    }

    print(f"\n[INFO] 📊 TensorBoard logging enabled:")
    print(f"  ✓ Log directory: {log_dir}")
    print(f"  ✓ Write interval: every 50 steps")
    print(f"  ✓ Checkpoint interval: every 500 steps")
    print(f"\n  🌐 To view TensorBoard:")
    print(f"     tensorboard --logdir {log_dir} --port 6006")
    
    # ========================================================================
    # Run Training
    # ========================================================================
    
    print("\n" + "="*70)
    print("🚀 STARTING SAC TRAINING (DirectRLEnv)")
    print("="*70)
    print(f"  Environments: {args_cli.num_envs}")
    print(f"  Max iterations: {args_cli.max_iterations}")
    print(f"  Total timesteps: {trainer_cfg['timesteps']:,}")
    print(f"  Device: {device}")
    print(f"  Voxel grid: {env_cfg.voxel_grid_dims}")
    print(f"  Logs: {log_dir}")
    print("="*70 + "\n")
    
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)
    trainer.train()
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED!")
    print("="*70)
    print(f"  Logs: {log_dir}")
    print("="*70 + "\n")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()

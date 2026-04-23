# Copyright (c) 2025, Go2 SAC LiDAR Expert Training Project.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Neural network architectures for SAC training with voxel observations.

This module defines skrl-compatible Actor-Critic networks that process:
- 3D voxel grids (from LiDAR) via 3D-CNN
- Robot state vectors via MLP
- Fused multimodal representations

Architecture:
    Voxels (B,64,48,12) ──┐
                          ├──> [3D-CNN] ──> feat_voxel (128)
                          │                              ├──> [Fusion] ──> [MLP] ──> Actor/Critic
    RobotState (B,48) ────┘──> [MLP] ────> feat_state (64)

Design Philosophy:
- 端到端学习 (End-to-End): CNN自己学会从体素中提取导航特征
- 模块化 (Modular): 编码器可以独立预训练
- GPU高效 (GPU-Efficient): 全部使用in-place操作
"""

import torch
import torch.nn as nn
from typing import Union, Tuple, Dict

# skrl imports
from skrl.models.torch import Model, DeterministicMixin, GaussianMixin


class VoxelEncoder3D(nn.Module):
    """
    3D Convolutional encoder for voxel grid perception.
    
    Input: (B, 1, W, H, D) = (B, 1, 64, 48, 12)
    Output: (B, feature_dim) OR (B, channels, H', W', D') for Transformer mode
    
    Architecture:
        Mode 1 (feature vector): Conv3D -> Flatten -> FC -> ReLU
        Mode 2 (spatial tokens): Conv3D -> Feature map (no flatten)
    """
    
    def __init__(self, grid_dims=(64, 48, 12), feature_dim=128, transformer_mode=False):
        super().__init__()
        
        W, H, D = grid_dims
        self.transformer_mode = transformer_mode
        
        # 3D CNN backbone (shared for both modes)
        # Layer 3 output channels = 64 (hardcoded)
        self.output_channels = 64
        
        self.backbone = nn.Sequential(
            # Layer 1: (B,1,64,48,12) -> (B,16,32,24,6)
            nn.Conv3d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            
            # Layer 2: (B,16,32,24,6) -> (B,32,16,12,3)
            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            # Layer 3: (B,32,16,12,3) -> (B,64,8,6,2)
            nn.Conv3d(32, self.output_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(self.output_channels),
            nn.ReLU(inplace=True),
        )
        
        if not transformer_mode:
            # **Feature vector mode** (original): Flatten -> FC
            self.spatial_shape = (8, 6, 2)  # Output spatial size of backbone
            
            with torch.no_grad():
                dummy = torch.zeros(1, 1, W, H, D)
                backbone_out = self.backbone(dummy)
                flat_dim = backbone_out.view(1, -1).shape[1]  # Flatten all spatial dims
            
            self.global_pool = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),  # Global average pooling
                nn.Flatten(),  # (B, 64)
            )
            
            self.fc = nn.Sequential(
                nn.Linear(64, feature_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(feature_dim * 2, feature_dim),
                nn.ReLU(inplace=True),
            )
        else:
            # **Transformer mode**: Return spatial feature map as tokens
            # Output: (B, 64, 8, 6, 2) → reshape to (B, seq_len=8*6*2=96, channels=64)
            # Note: self.output_channels = 64 already set in backbone definition
            self.spatial_shape = (8, 6, 2)
            seq_len = 8 * 6 * 2  # 96 spatial tokens
            
            # Add projection layer to match transformer_dim
            self.token_projection = nn.Linear(self.output_channels, feature_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """He initialization for Conv3D layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, voxel_grid: torch.Tensor, return_spatial_tokens=False) -> Union[torch.Tensor, torch.Tensor]:
        """
        Args:
            voxel_grid: (B, W, H, D) or (B, 1, W, H, D)
            return_spatial_tokens: If True, return spatial tokens for Transformer
        Returns:
            if not return_spatial_tokens:
                features: (B, feature_dim) - global feature vector
            else:
                tokens: (B, seq_len, token_dim) - spatial tokens for Transformer
        """
        # Ensure 5D input (add channel dim if needed)
        if voxel_grid.ndim == 4:
            voxel_grid = voxel_grid.unsqueeze(1)  # (B,W,H,D) -> (B,1,W,H,D)
        
        # Ensure float32 for CNN
        if voxel_grid.dtype != torch.float32:
            voxel_grid = voxel_grid.float()
        
        # Pass through CNN backbone
        backbone_out = self.backbone(voxel_grid)  # (B, 64, 8, 6, 2)
        
        if not self.transformer_mode or not return_spatial_tokens:
            # **Feature vector mode**: Global pooling + FC
            # Used by Critic networks (no Transformer)
            x = self.global_pool(backbone_out)  # (B, 64)
            features = self.fc(x)  # (B, feature_dim)
            return features
        else:
            # **Transformer mode**: Return spatial tokens
            # Reshape: (B, 64, 8, 6, 2) -> (B, seq_len=8*6*2=96, channels=64)
            B, C, H, W, D = backbone_out.shape
            tokens = backbone_out.permute(0, 2, 3, 4, 1)  # (B, H, W, D, C)
            tokens = tokens.reshape(B, H * W * D, C)  # (B, seq_len, channels)
            
            # Project to transformer dimension
            tokens = self.token_projection(tokens)  # (B, seq_len, feature_dim)
            return tokens


class RobotStateEncoder(nn.Module):
    """
    MLP encoder for robot proprioceptive state.
    
    Input: (B, state_dim) - joint positions, velocities, IMU, etc.
    Output: (B, feature_dim)
    """
    
    def __init__(self, state_dim=48, feature_dim=64):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.mlp(state)


class SpatialTransformerBlock(nn.Module):
    """
    **ADDED** Spatial Transformer block for processing 3D voxel features.
    
    This transformer processes spatially-aware token sequences from voxel features.
    Each token corresponds to a region in the 3D voxel grid, preserving spatial relationships.
    
    Input: (B, seq_len, d_model) spatial feature tokens
    Output: (B, seq_len, d_model) transformed tokens
    """
    
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, 
                 spatial_shape=(8, 6, 4)):
        """
        Args:
            d_model: Transformer hidden dimension
            nhead: Number of attention heads
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            spatial_shape: (H, W, D) spatial dimensions for learnable position encoding
        """
        super().__init__()
        
        self.d_model = d_model
        self.spatial_shape = spatial_shape
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feedforward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Learnable spatial position encoding
        # Create a 3D grid of position embeddings that can be interpolated
        self.spatial_embedding = nn.Parameter(
            torch.randn(1, *spatial_shape, d_model) * 0.02
        )
        
    def forward(self, x, attention_mask=None):
        """
        Args:
            x: Input tokens (B, seq_len, d_model)
            attention_mask: Optional mask for attention
        Returns:
            Transformed tokens (B, seq_len, d_model)
        """
        # Self-attention with residual connection
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=attention_mask)
        x = self.norm1(x + attn_out)
        
        # Feedforward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x
    
    def get_spatial_encoding(self, coords):
        """
        Get interpolated position encoding for arbitrary coordinates.
        
        Args:
            coords: Normalized coordinates (B, seq_len, 3) in range [-1, 1]
        Returns:
            Position embeddings (B, seq_len, d_model)
        """
        B, seq_len, _ = coords.shape
        
        # Interpolate from learned spatial embedding
        # coords: [-1, 1] -> [0, spatial_shape]
        coords = (coords + 1) / 2  # Normalize to [0, 1]
        coords = coords * (torch.tensor(self.spatial_shape, device=coords.device) - 1).float()
        
        # Use grid_sample for differentiable interpolation
        # Reshape for grid_sample: (B, D, H, W, C) -> (B, C, D, H, W)
        spatial_emb = self.spatial_embedding.permute(0, 4, 1, 2, 3)  # (1, d_model, H, W, D)
        
        # grid_sample expects (B, seq_len, 3) where 3 -> (x, y, z)
        # but our spatial_shape is (H, W, D) -> need to reorder
        grid = coords[:, :, [1, 2, 0]].unsqueeze(1).unsqueeze(1)  # (B, 1, 1, seq_len, 3)
        
        # Interpolate
        pos_emb = nn.functional.grid_sample(
            spatial_emb.expand(B, -1, -1, -1, -1), 
            grid, 
            mode='bilinear', 
            align_corners=True
        )
        
        # Reshape back: (B, d_model, 1, 1, seq_len) -> (B, seq_len, d_model)
        return pos_emb.squeeze(2).squeeze(2).permute(0, 2, 1)


# ============================================================================
# SKRL-Compatible Models
# ============================================================================

class SACPolicy(GaussianMixin, Model):
    """
    SAC Actor (Policy) network with voxel + state + TRANSFORMER fusion.
    
    **UPGRADED** Architecture:
      Voxels → [3D-CNN] → voxel_feat (128)
        ├───────→ [SpatialTransformer] → spatial_feat (512)
      State → [MLP] → state_feat (64)
        └─────────────┬─────────────────┘
                      ↓
                [FUSION CAT]
                      ↓
                [MLP: 512→512→512]
                      ↓
              [GaussianPolicy]
    
    This model is compatible with skrl's SAC agent.
    It uses GaussianMixin to output a stochastic policy.
    
    Observation space (dict):
        - 'voxels': (B, 64, 48, 12)
        - 'robot_state': (B, 48)
    
    Action space:
        - continuous: (B, 12) for Go2 joints
    """
    
    def __init__(self, observation_space, action_space, device,
                 voxel_feature_dim=128,
                 state_feature_dim=64,
                 transformer_dim=512,        # **ADDED**
                 transformer_layers=2,       # **ADDED**
                 mlp_hidden_dims=(512, 512, 512),  # **DOUBLED**
                 **kwargs):
        """
        Args:
            observation_space: gym.spaces.Box with shape (36915,) - flattened [voxels + state]
            action_space: gym.spaces.Box
            device: torch device
            voxel_feature_dim: output dim of voxel encoder (before transformer)
            state_feature_dim: output dim of state encoder
            transformer_dim: **ADDED** Transformer hidden dimension
            transformer_layers: **ADDED** Number of transformer layers
            mlp_hidden_dims: **DOUBLED** hidden layer sizes for policy MLP
        """
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=True, clip_log_std=True,
                               min_log_std=-20, max_log_std=2)
        
        # Extract dimensions from flattened observation space
        # observation_space.shape[0] = 36915 (36864 voxels + 51 state)
        total_obs_dim = observation_space.shape[0]  # 36915
        voxel_dim = 64 * 48 * 12  # 36,864
        state_dim = total_obs_dim - voxel_dim  # 51
        action_dim = action_space.shape[0]  # 12
        
        # Voxel shape for encoder
        voxel_shape = (64, 48, 12)
        
        # **MODIFIED** Create voxel encoder with TRANSFORMER MODE enabled
        # In transformer mode, encoder returns spatial tokens (B, seq_len, channels) instead of global features
        self.voxel_encoder = VoxelEncoder3D(grid_dims=voxel_shape, feature_dim=voxel_feature_dim, transformer_mode=True)
        self.state_encoder = RobotStateEncoder(state_dim=state_dim, feature_dim=state_feature_dim)
        
        # **ADDED** Spatial Transformer for voxel features
        # CNN backbone output: (B, 64, 8, 6, 2) → reshape to (B, seq_len=96, channels=64)
        # BUT: VoxelEncoder3D in transformer_mode internally projects to feature_dim (default 128)
        # So actual output channels = voxel_feature_dim (128), not output_channels (64)
        self.transformer_dim = transformer_dim
        self.spatial_token_size = (8, 6, 2)  # **FIXED** Correct spatial dimensions from CNN output (8,6,2), not (8,6,4)
        
        # VoxelEncoder3D in transformer_mode returns (B, seq_len, feature_dim) where feature_dim=128
        # So we need to project from 128 to transformer_dim (512)
        actual_encoder_output_dim = voxel_feature_dim  # 128, not self.voxel_encoder.output_channels (64)
        self.token_projection = nn.Linear(actual_encoder_output_dim, transformer_dim)
        
        # **ADDED** Create spatial positions for each token
        # These are fixed 3D positions in voxel space
        H, W, D = self.spatial_token_size
        token_positions = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            torch.linspace(-1, 1, D),
            indexing='ij'
        ), dim=-1).reshape(-1, 3)  # (H*W*D, 3)
        self.register_buffer('token_positions', token_positions)
        
        # **ADDED** Spatial Transformer blocks
        self.spatial_transformer = nn.ModuleList([
            SpatialTransformerBlock(
                d_model=transformer_dim,
                nhead=8,
                dim_feedforward=transformer_dim * 4,
                dropout=0.1,
                spatial_shape=self.spatial_token_size
            )
            for _ in range(transformer_layers)
        ])
        
        # Pooling: Mean pooling over spatial tokens
        self.spatial_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fusion dimension (after pooling)
        fused_dim = transformer_dim + state_feature_dim
        
        # Policy MLP (DOUBLED size)
        mlp_layers = []
        in_dim = fused_dim
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        
        self.policy_mlp = nn.Sequential(*mlp_layers)
        
        # Output heads
        self.mean_layer = nn.Linear(in_dim, action_dim)
        self.log_std_layer = nn.Linear(in_dim, action_dim)
        
        # Initialize output layers with small weights (for stable initial policy)
        nn.init.uniform_(self.mean_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_layer.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_layer.bias, -3e-3, 3e-3)
    
    def compute(self, inputs: Dict[str, torch.Tensor], role: str) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        skrl's required compute method.
        
        Args:
            inputs: dict with keys 'states' containing observations
            role: string ('policy', 'target_policy', etc.)
        
        Returns:
            mean: (B, action_dim)
            log_std: (B, action_dim)
            outputs: dict (can contain additional info)
        """
        # Extract observations
        # Extract flattened observations (skrl flattens Dict to tensor)
        # voxels: 64*48*12=36864, robot_state: 51, total: 36915
        obs_flat = inputs["states"]  # (B, 36915)
        batch_size = obs_flat.shape[0]
        voxels_flat = obs_flat[:, :36864]
        robot_state = obs_flat[:, 36864:]
        voxels = voxels_flat.reshape(batch_size, 64, 48, 12)
        
        # CRITICAL: Ensure tensors are on the same device as the model
        # When buffer is on CPU, sampled data is on CPU but model is on GPU
        device = self.device
        if voxels.device != device:
            voxels = voxels.to(device)
        if robot_state.device != device:
            robot_state = robot_state.to(device)
        
        # **UPGRADED** Encode with Spatial Transformer
        # Encode voxels through 3D-CNN (returns spatial tokens for Transformer mode)
        spatial_tokens = self.voxel_encoder(voxels, return_spatial_tokens=True)  # (B, seq_len=96, channels=64)
        
        # Project token channels to transformer dimension
        spatial_tokens = self.token_projection(spatial_tokens)  # (B, seq_len, transformer_dim)
        
        # **ADDED** Get learnable position embeddings
        # These are fixed 3D positions for each token (learned during training)
        position_embeddings = self.spatial_transformer[0].get_spatial_encoding(
            self.token_positions.unsqueeze(0).expand(batch_size, -1, -1)
        )  # (B, seq_len, transformer_dim)
        
        # Add spatial position encoding
        spatial_tokens = spatial_tokens + position_embeddings
        
        # **ADDED** Pass through Spatial Transformer layers
        for transformer_layer in self.spatial_transformer:
            spatial_tokens = transformer_layer(spatial_tokens)  # (B, seq_len, transformer_dim)
        
        # **ADDED** Pool spatial tokens (mean pooling)
        # Transpose for pooling: (B, transformer_dim, seq_len) -> pool -> (B, transformer_dim, 1)
        pooled_voxel_feat = self.spatial_pool(spatial_tokens.transpose(1, 2)).squeeze(-1)  # (B, transformer_dim)
        
        # Encode state
        state_feat = self.state_encoder(robot_state)  # (B, 64)
        
        # Fuse transformer-processed voxel features with state
        fused_feat = torch.cat([pooled_voxel_feat, state_feat], dim=-1)  # (B, transformer_dim + state_dim)
        
        # Policy MLP (upgraded size)
        x = self.policy_mlp(fused_feat)  # (B, last_hidden_dim)
        
        # Output
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        
        return mean, log_std, {}


class SACCritic(DeterministicMixin, Model):
    """
    SAC Critic (Q-function) network with voxel + state + action fusion.
    
    This model is compatible with skrl's SAC agent.
    It uses DeterministicMixin to output Q-values.
    
    Note: SAC uses twin critics, so this will be instantiated twice.
    """
    
    def __init__(self, observation_space, action_space, device,
                 voxel_feature_dim=128,
                 state_feature_dim=64,
                 mlp_hidden_dims=(256, 256),
                 **kwargs):
        """
        Args:
            observation_space: gym.spaces.Box with shape (36915,) - flattened [voxels + state]
            action_space: gym.spaces.Box
            device: torch device
            voxel_feature_dim: output dim of voxel encoder
            state_feature_dim: output dim of state encoder
            mlp_hidden_dims: hidden layer sizes for Q-function MLP
        """
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        
        # Extract dimensions from flattened observation space
        total_obs_dim = observation_space.shape[0]  # 36915
        voxel_dim = 64 * 48 * 12  # 36,864
        state_dim = total_obs_dim - voxel_dim  # 51
        action_dim = action_space.shape[0]  # 12
        
        # Voxel shape for encoder
        voxel_shape = (64, 48, 12)
        
        # Encoders (separate from policy, allows different learning rates if needed)
        self.voxel_encoder = VoxelEncoder3D(grid_dims=voxel_shape, feature_dim=voxel_feature_dim)
        self.state_encoder = RobotStateEncoder(state_dim=state_dim, feature_dim=state_feature_dim)
        
        # Fusion dimension (state + action)
        fused_dim = voxel_feature_dim + state_feature_dim + action_dim
        
        # Q-function MLP
        mlp_layers = []
        in_dim = fused_dim
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        
        mlp_layers.append(nn.Linear(in_dim, 1))  # Output: Q-value
        self.q_mlp = nn.Sequential(*mlp_layers)
        
        # Initialize output layer with small weights
        nn.init.uniform_(self.q_mlp[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.q_mlp[-1].bias, -3e-3, 3e-3)
    
    def compute(self, inputs: Dict[str, torch.Tensor], role: str) -> Tuple[torch.Tensor, dict]:
        """
        skrl's required compute method for critics.
        
        Args:
            inputs: dict with keys 'states' (observations) and 'taken_actions'
            role: string ('critic_1', 'critic_2', 'target_critic_1', etc.)
        
        Returns:
            q_value: (B, 1)
            outputs: dict
        """
        # Extract observations and actions
        # Extract flattened observations (skrl flattens Dict to tensor)
        # voxels: 64*48*12=36864, robot_state: 51, total: 36915
        obs_flat = inputs["states"]  # (B, 36915)
        actions = inputs["taken_actions"]  # (B, action_dim)
        
        batch_size = obs_flat.shape[0]
        voxels_flat = obs_flat[:, :36864]
        robot_state = obs_flat[:, 36864:]
        voxels = voxels_flat.reshape(batch_size, 64, 48, 12)
        
        # CRITICAL: Ensure tensors are on the same device as the model
        device = self.device
        if voxels.device != device:
            voxels = voxels.to(device)
        if robot_state.device != device:
            robot_state = robot_state.to(device)
        if actions.device != device:
            actions = actions.to(device)
        
        # Encode
        voxel_feat = self.voxel_encoder(voxels)
        state_feat = self.state_encoder(robot_state)
        
        # Fuse with action
        fused_feat = torch.cat([voxel_feat, state_feat, actions], dim=-1)
        
        # Q-value
        q = self.q_mlp(fused_feat)
        
        return q, {}

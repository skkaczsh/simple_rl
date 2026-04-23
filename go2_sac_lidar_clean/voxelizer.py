# Copyright (c) 2025, Go2 SAC LiDAR Expert Training Project.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
GPU-accelerated voxelization module for LiDAR point clouds.

This module provides efficient, batched voxelization of 3D point clouds
on GPU, designed for real-time robotics applications in Isaac Lab.

Design Philosophy:
- 100% GPU computation (zero CPU-GPU transfers)
- Batched processing for multiple parallel environments
- Memory-efficient uint8 binary voxel grids
- JIT-compilable for maximum performance
"""

import torch
import torch.nn as nn
from typing import Tuple


class Voxelizer(nn.Module):
    """
    Fast GPU voxelizer for converting LiDAR point clouds to 3D voxel grids.
    
    This is a PyTorch nn.Module so it can be part of the observation pipeline
    and benefit from autograd if needed (though typically we don't backprop through it).
    
    Args:
        grid_dims: Tuple of (width, height, depth) for voxel grid (64, 48, 12)
        voxel_size: Physical size of each voxel in meters (0.1m = 10cm)
        range_x: (min, max) range in x-axis (robot forward), e.g., (-3.2, 3.2) for 6.4m
        range_y: (min, max) range in y-axis (robot left), e.g., (-2.4, 2.4) for 4.8m  
        range_z: (min, max) range in z-axis (robot up), e.g., (-0.6, 0.6) for 1.2m
        device: torch device
    """
    
    def __init__(
        self,
        grid_dims: Tuple[int, int, int] = (64, 48, 12),
        voxel_size: float = 0.1,
        range_x: Tuple[float, float] = (-3.2, 3.2),
        range_y: Tuple[float, float] = (-2.4, 2.4),
        range_z: Tuple[float, float] = (-0.6, 0.6),
        device: str = "cuda"
    ):
        super().__init__()
        
        self.grid_dims = grid_dims  # (W, H, D)
        self.voxel_size = voxel_size
        self.device = device
        
        # Register ranges as buffers (will auto-move to correct device)
        self.register_buffer("range_x", torch.tensor(range_x, device=device))
        self.register_buffer("range_y", torch.tensor(range_y, device=device))
        self.register_buffer("range_z", torch.tensor(range_z, device=device))
        
        # Pre-compute grid parameters
        self.register_buffer("origin", torch.tensor([range_x[0], range_y[0], range_z[0]], device=device))
        
    def forward(
        self,
        points_world: torch.Tensor,
        robot_pos: torch.Tensor,
        robot_quat: torch.Tensor
    ) -> torch.Tensor:
        """
        Voxelize point clouds from multiple environments in parallel.
        
        Args:
            points_world: Point cloud in world frame (B, N, 3)
            robot_pos: Robot base position in world frame (B, 3)
            robot_quat: Robot base orientation as quaternion (B, 4) [w, x, y, z]
        
        Returns:
            voxel_grid: Binary voxel occupancy grid (B, W, H, D) dtype=uint8
                        1 = occupied, 0 = free
        """
        # Handle both (B, N, 3) and (N, 3) input shapes
        if points_world.dim() == 2:
            # (N, 3) -> (1, N, 3)
            points_world = points_world.unsqueeze(0)
            robot_pos = robot_pos.unsqueeze(0) if robot_pos.dim() == 1 else robot_pos
            robot_quat = robot_quat.unsqueeze(0) if robot_quat.dim() == 1 else robot_quat
        
        B, N, _ = points_world.shape
        W, H, D = self.grid_dims
        
        # 1. Transform points from world frame to robot base frame
        points_robot = self._transform_to_robot_frame(points_world, robot_pos, robot_quat)
        
        # 2. Convert continuous coordinates to discrete voxel indices
        voxel_coords = self._points_to_voxel_coords(points_robot)
        
        # 3. Filter out-of-bounds points
        valid_mask = self._get_valid_mask(voxel_coords)
        
        # 4. Create voxel grid and scatter occupied voxels
        voxel_grid = self._scatter_to_grid(voxel_coords, valid_mask, B)
        
        return voxel_grid
    
    def _transform_to_robot_frame(
        self,
        points_w: torch.Tensor,
        robot_pos: torch.Tensor,
        robot_quat: torch.Tensor
    ) -> torch.Tensor:
        """
        Transform points from world frame to robot base frame.
        
        Args:
            points_w: (B, N, 3) points in world frame
            robot_pos: (B, 3) robot position
            robot_quat: (B, 4) robot orientation [w, x, y, z]
        
        Returns:
            points_b: (B, N, 3) points in robot base frame
        """
        # Translate: world -> robot origin
        points_rel = points_w - robot_pos.unsqueeze(1)  # (B, N, 3)
        
        # Rotate: use inverse quaternion rotation
        # For a unit quaternion q, the inverse is q* = [w, -x, -y, -z]
        quat_inv = robot_quat.clone()
        quat_inv[:, 1:] = -quat_inv[:, 1:]  # Conjugate
        
        # Rotate each point using quaternion
        points_b = self._quat_rotate(quat_inv.unsqueeze(1), points_rel)  # (B, N, 3)
        
        return points_b
    
    def _quat_rotate(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Rotate vector v by quaternion q.
        
        Args:
            q: quaternion (B, 1, 4) or (B, N, 4) [w, x, y, z]
            v: vector (B, N, 3)
        
        Returns:
            rotated vector (B, N, 3)
        """
        # Extract components
        qw = q[..., 0:1]
        qx = q[..., 1:2]
        qy = q[..., 2:3]
        qz = q[..., 3:4]
        
        vx = v[..., 0:1]
        vy = v[..., 1:2]
        vz = v[..., 2:3]
        
        # Quaternion rotation formula
        # v' = v + 2*qw*(qxyz x v) + 2*(qxyz x (qxyz x v))
        
        # Cross product: qxyz x v
        qvec = torch.cat([qx, qy, qz], dim=-1)
        cross1 = torch.cross(qvec, v, dim=-1)
        
        # 2*qw*(qxyz x v)
        term1 = 2.0 * qw * cross1
        
        # qxyz x (qxyz x v)
        cross2 = torch.cross(qvec, cross1, dim=-1)
        
        # 2*(qxyz x (qxyz x v))
        term2 = 2.0 * cross2
        
        return v + term1 + term2
    
    def _points_to_voxel_coords(self, points: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous point coordinates to discrete voxel indices.
        
        Args:
            points: (B, N, 3) points in robot frame
        
        Returns:
            coords: (B, N, 3) voxel indices [ix, iy, iz]
        """
        # Subtract origin and divide by voxel size
        coords = (points - self.origin) / self.voxel_size
        
        # Floor to get integer indices
        coords = coords.long()
        
        return coords
    
    def _get_valid_mask(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Create mask for points that fall within voxel grid bounds.
        
        Args:
            coords: (B, N, 3) voxel indices
        
        Returns:
            mask: (B, N) boolean mask
        """
        W, H, D = self.grid_dims
        
        valid_x = (coords[..., 0] >= 0) & (coords[..., 0] < W)
        valid_y = (coords[..., 1] >= 0) & (coords[..., 1] < H)
        valid_z = (coords[..., 2] >= 0) & (coords[..., 2] < D)
        
        return valid_x & valid_y & valid_z
    
    def _scatter_to_grid(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Scatter valid points into voxel grid.
        
        Args:
            coords: (B, N, 3) voxel indices
            mask: (B, N) valid points mask
            batch_size: B
        
        Returns:
            grid: (B, W, H, D) binary voxel grid
        """
        W, H, D = self.grid_dims
        
        # Initialize empty grid
        grid = torch.zeros(
            (batch_size, W, H, D),
            dtype=torch.uint8,
            device=self.device
        )
        
        # Get valid coordinates
        valid_coords = coords[mask]  # (M, 3) where M = total valid points
        
        # Get batch indices for each valid point
        batch_indices = torch.arange(batch_size, device=self.device)
        batch_indices = batch_indices.view(-1, 1).expand(-1, coords.shape[1])  # (B, N)
        valid_batch_idx = batch_indices[mask]  # (M,)
        
        # Extract individual coordinate components
        ix = valid_coords[:, 0]
        iy = valid_coords[:, 1]
        iz = valid_coords[:, 2]
        
        # Set occupied voxels to 1
        # Note: If multiple points fall into same voxel, it's still just 1
        grid[valid_batch_idx, ix, iy, iz] = 1
        
        return grid
    
    @torch.jit.export
    def get_voxel_centers(self) -> torch.Tensor:
        """
        Get the center coordinates of all voxels in robot frame.
        Useful for visualization.
        
        Returns:
            centers: (W, H, D, 3) voxel center positions
        """
        W, H, D = self.grid_dims
        
        # Create meshgrid of indices
        ix = torch.arange(W, device=self.device)
        iy = torch.arange(H, device=self.device)
        iz = torch.arange(D, device=self.device)
        
        grid_x, grid_y, grid_z = torch.meshgrid(ix, iy, iz, indexing='ij')
        
        # Convert indices to center coordinates
        centers = torch.stack([grid_x, grid_y, grid_z], dim=-1).float()
        centers = centers * self.voxel_size + self.origin + self.voxel_size / 2.0
        
        return centers



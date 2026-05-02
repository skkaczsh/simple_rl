"""Curriculum learning manager for Phase 1 locomotion training.

Defines 7 stages from standing to velocity tracking with domain randomization.
Each stage has specific command ranges and upgrade conditions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CurriculumStage:
    """A single curriculum stage with command ranges and upgrade conditions."""
    name: str
    description: str
    # Command ranges for this stage
    vx_range: tuple[float, float] = (0.0, 0.0)
    vy_range: tuple[float, float] = (0.0, 0.0)
    yaw_rate_range: tuple[float, float] = (0.0, 0.0)
    height_range: tuple[float, float] = (0.71, 0.77)
    # Training config overrides
    num_iterations: int = 200
    learning_rate: float = 5e-5
    horizon_steps: int = 500
    # Upgrade conditions
    min_iterations: int = 100
    min_reward: float = 3.5
    min_survival_rate: float = 0.8
    max_velocity_rmse: float | None = None  # If set, RMSE must be below this


# Default curriculum stages
DEFAULT_CURRICULUM = [
    CurriculumStage(
        name="1A_stand_balance",
        description="Stand still, maintain upright posture",
        vx_range=(0.0, 0.0),
        vy_range=(0.0, 0.0),
        yaw_rate_range=(0.0, 0.0),
        height_range=(0.71, 0.77),
        num_iterations=300,
        min_reward=3.5,
        min_survival_rate=0.85,
    ),
    CurriculumStage(
        name="1B_forward_walk",
        description="Walk forward at slow speed",
        vx_range=(0.1, 0.5),
        vy_range=(0.0, 0.0),
        yaw_rate_range=(0.0, 0.0),
        height_range=(0.71, 0.77),
        num_iterations=400,
        min_reward=3.0,
        min_survival_rate=0.75,
    ),
    CurriculumStage(
        name="1C_velocity_tracking",
        description="Track commanded forward and lateral velocity",
        vx_range=(-0.3, 0.8),
        vy_range=(-0.2, 0.2),
        yaw_rate_range=(0.0, 0.0),
        height_range=(0.71, 0.77),
        num_iterations=500,
        min_reward=2.5,
        min_survival_rate=0.7,
    ),
    CurriculumStage(
        name="1D_yaw_tracking",
        description="Track yaw rate commands",
        vx_range=(-0.3, 0.8),
        vy_range=(-0.2, 0.2),
        yaw_rate_range=(-0.5, 0.5),
        height_range=(0.71, 0.77),
        num_iterations=500,
        min_reward=2.5,
        min_survival_rate=0.7,
    ),
    CurriculumStage(
        name="1E_command_switch",
        description="Handle sudden command changes",
        vx_range=(-0.5, 1.0),
        vy_range=(-0.3, 0.3),
        yaw_rate_range=(-0.8, 0.8),
        height_range=(0.68, 0.80),
        num_iterations=500,
        min_reward=2.0,
        min_survival_rate=0.65,
    ),
    CurriculumStage(
        name="1F_full_command",
        description="Full command space with all directions",
        vx_range=(-0.5, 1.0),
        vy_range=(-0.3, 0.3),
        yaw_rate_range=(-1.0, 1.0),
        height_range=(0.65, 0.83),
        num_iterations=500,
        min_reward=2.0,
        min_survival_rate=0.6,
    ),
    CurriculumStage(
        name="1G_domain_randomization",
        description="Full command space with heavy domain randomization",
        vx_range=(-0.5, 1.0),
        vy_range=(-0.3, 0.3),
        yaw_rate_range=(-1.0, 1.0),
        height_range=(0.65, 0.83),
        num_iterations=500,
        min_reward=2.0,
        min_survival_rate=0.6,
    ),
]


@dataclass
class CurriculumState:
    """Tracks the current state of curriculum progression."""
    current_stage_idx: int = 0
    stage_iterations: int = 0
    stage_rewards: list[float] = field(default_factory=list)
    stage_survival_rates: list[float] = field(default_factory=list)
    stage_velocity_rmses: list[float] = field(default_factory=list)
    completed_stages: list[str] = field(default_factory=list)
    total_iterations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_stage_idx": self.current_stage_idx,
            "stage_iterations": self.stage_iterations,
            "completed_stages": self.completed_stages,
            "total_iterations": self.total_iterations,
            "current_stage_rewards": self.stage_rewards[-10:] if self.stage_rewards else [],
            "current_stage_survival": self.stage_survival_rates[-10:] if self.stage_survival_rates else [],
        }


class CurriculumManager:
    """Manages curriculum progression for locomotion training.

    Tracks training metrics and automatically advances to the next stage
    when upgrade conditions are met.
    """

    def __init__(self, stages: list[CurriculumStage] | None = None) -> None:
        self.stages = stages or DEFAULT_CURRICULUM
        self.state = CurriculumState()

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.state.current_stage_idx]

    @property
    def is_complete(self) -> bool:
        return self.state.current_stage_idx >= len(self.stages)

    def record_iteration(self, avg_reward: float, survival_rate: float, velocity_rmse: float = 0.0) -> None:
        """Record metrics from a training iteration."""
        self.state.stage_rewards.append(avg_reward)
        self.state.stage_survival_rates.append(survival_rate)
        self.state.stage_velocity_rmses.append(velocity_rmse)
        self.state.stage_iterations += 1
        self.state.total_iterations += 1

    def should_upgrade(self) -> bool:
        """Check if upgrade conditions are met for the current stage."""
        if self.is_complete:
            return False

        stage = self.current_stage
        state = self.state

        # Need minimum iterations
        if state.stage_iterations < stage.min_iterations:
            return False

        # Check recent reward (last 20 iterations)
        recent_rewards = state.stage_rewards[-20:]
        if len(recent_rewards) < 10:
            return False
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        if avg_reward < stage.min_reward:
            return False

        # Check recent survival rate (last 20 iterations)
        recent_survival = state.stage_survival_rates[-20:]
        if len(recent_survival) < 10:
            return False
        avg_survival = sum(recent_survival) / len(recent_survival)
        if avg_survival < stage.min_survival_rate:
            return False

        # Check velocity RMSE if threshold is set
        if stage.max_velocity_rmse is not None:
            recent_rmse = state.stage_velocity_rmses[-20:]
            if len(recent_rmse) < 10:
                return False
            avg_rmse = sum(recent_rmse) / len(recent_rmse)
            if avg_rmse > stage.max_velocity_rmse:
                return False

        return True

    def advance_stage(self) -> CurriculumStage | None:
        """Advance to the next curriculum stage. Returns the new stage or None if complete."""
        if self.is_complete:
            return None

        completed = self.current_stage.name
        self.state.completed_stages.append(completed)
        self.state.current_stage_idx += 1
        self.state.stage_iterations = 0
        self.state.stage_rewards = []
        self.state.stage_survival_rates = []
        self.state.stage_velocity_rmses = []

        if self.is_complete:
            return None
        return self.current_stage

    def get_training_config(self) -> dict[str, Any]:
        """Get training config for the current stage."""
        stage = self.current_stage
        return {
            "vx_range": list(stage.vx_range),
            "vy_range": list(stage.vy_range),
            "yaw_rate_range": list(stage.yaw_rate_range),
            "height_range": list(stage.height_range),
            "num_iterations": stage.num_iterations,
            "learning_rate": stage.learning_rate,
            "horizon_steps": stage.horizon_steps,
        }

    def get_status(self) -> dict[str, Any]:
        """Get current curriculum status for logging."""
        if self.is_complete:
            return {
                "status": "complete",
                "completed_stages": self.state.completed_stages,
                "total_iterations": self.state.total_iterations,
            }

        stage = self.current_stage
        recent_rewards = self.state.stage_rewards[-20:]
        recent_survival = self.state.stage_survival_rates[-20:]
        return {
            "status": "training",
            "current_stage": stage.name,
            "stage_description": stage.description,
            "stage_iterations": self.state.stage_iterations,
            "stage_target": stage.num_iterations,
            "recent_avg_reward": sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0,
            "recent_avg_survival": sum(recent_survival) / len(recent_survival) if recent_survival else 0.0,
            "upgrade_threshold_reward": stage.min_reward,
            "upgrade_threshold_survival": stage.min_survival_rate,
            "completed_stages": self.state.completed_stages,
            "total_iterations": self.state.total_iterations,
        }

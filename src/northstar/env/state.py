from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MockPhase1State:
    step_index: int = 0
    time_s: float = 0.0
    base_linear_velocity_m_s: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    base_angular_velocity_rad_s: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    base_height_m: float = 0.74
    projected_gravity_base: list[float] = field(default_factory=lambda: [0.0, 0.0, -1.0])

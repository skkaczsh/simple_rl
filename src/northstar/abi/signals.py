from __future__ import annotations

from typing import Any


def make_confidence(
    overall: float,
    stability: float,
    tracking: float,
    fallback: float,
    source: str = "debug",
) -> dict[str, Any]:
    return {
        "schema_version": "confidence.northstar.v0",
        "overall": float(overall),
        "stability": float(stability),
        "tracking": float(tracking),
        "fallback": float(fallback),
        "source": source,
    }


def make_dangerous_signal(
    overall_risk: float,
    triggered: list[str],
    fall_risk: float = 0.0,
    near_fall_risk: float = 0.0,
    limit_risk: float = 0.0,
    tracking_risk: float = 0.0,
) -> dict[str, Any]:
    return {
        "schema_version": "dangerous_signal.northstar.v0",
        "overall_risk": float(overall_risk),
        "fall_risk": float(fall_risk),
        "near_fall_risk": float(near_fall_risk),
        "limit_risk": float(limit_risk),
        "tracking_risk": float(tracking_risk),
        "triggered": list(triggered),
    }

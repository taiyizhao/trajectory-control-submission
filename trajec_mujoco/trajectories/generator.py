"""Time-varying 3D Cartesian trajectories for end-effector tracking."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np


SUPPORTED_TRAJECTORIES = ("circle", "figure8", "moving")


@dataclass(slots=True)
class TrajectoryParams:
    """Parameters for analytic 3D target trajectories."""

    trajectory_type: str = "circle"
    center: tuple[float, float, float] = (0.50, 0.00, 0.42)
    radius: float = 0.12
    y_radius: float = 0.10
    z_amplitude: float = 0.055
    frequency: float = 0.16
    phase: float = 0.0
    unreachable: bool = False


class TrajectoryGenerator:
    """Analytic trajectory family with optional per-episode randomization."""

    def __init__(
        self,
        base_params: TrajectoryParams | None = None,
        randomize: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.base_params = base_params or TrajectoryParams()
        self.randomize = randomize
        self.rng = rng or np.random.default_rng()
        self.params = self.base_params

    def reset(
        self,
        trajectory_type: str | None = None,
        unreachable: bool | None = None,
    ) -> TrajectoryParams:
        """Sample trajectory parameters for a new episode."""

        params = replace(self.base_params)
        if trajectory_type is not None:
            params.trajectory_type = trajectory_type
        if params.trajectory_type not in SUPPORTED_TRAJECTORIES:
            raise ValueError(
                f"Unknown trajectory_type={params.trajectory_type!r}; "
                f"expected one of {SUPPORTED_TRAJECTORIES}."
            )

        if self.randomize:
            center = np.asarray(params.center, dtype=np.float64)
            center += self.rng.uniform([-0.035, -0.035, -0.020], [0.035, 0.035, 0.020])
            params.center = tuple(center.tolist())
            params.radius *= float(self.rng.uniform(0.80, 1.20))
            params.y_radius *= float(self.rng.uniform(0.80, 1.25))
            params.z_amplitude *= float(self.rng.uniform(0.70, 1.25))
            params.frequency *= float(self.rng.uniform(0.80, 1.30))
            params.phase = float(self.rng.uniform(0.0, 2.0 * np.pi))

        if unreachable is not None:
            params.unreachable = bool(unreachable)

        if params.unreachable:
            center = np.asarray(params.center, dtype=np.float64)
            center[0] += 0.22
            center[2] += 0.03
            params.center = tuple(center.tolist())
            params.radius *= 1.25
            params.y_radius *= 1.20

        self.params = params
        return self.params

    def position_velocity(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return desired Cartesian position and velocity at time ``t``."""

        p = self.params
        center = np.asarray(p.center, dtype=np.float64)
        w = 2.0 * np.pi * p.frequency
        u = w * t + p.phase

        if p.trajectory_type == "circle":
            pos = center + np.array(
                [
                    p.radius * np.cos(u),
                    p.y_radius * np.sin(u),
                    p.z_amplitude * np.sin(0.5 * u),
                ],
                dtype=np.float64,
            )
            vel = np.array(
                [
                    -p.radius * w * np.sin(u),
                    p.y_radius * w * np.cos(u),
                    0.5 * p.z_amplitude * w * np.cos(0.5 * u),
                ],
                dtype=np.float64,
            )
            return pos, vel

        if p.trajectory_type == "figure8":
            pos = center + np.array(
                [
                    p.radius * np.sin(u),
                    0.5 * p.y_radius * np.sin(2.0 * u),
                    p.z_amplitude * np.sin(0.5 * u + 0.35),
                ],
                dtype=np.float64,
            )
            vel = np.array(
                [
                    p.radius * w * np.cos(u),
                    p.y_radius * w * np.cos(2.0 * u),
                    0.5 * p.z_amplitude * w * np.cos(0.5 * u + 0.35),
                ],
                dtype=np.float64,
            )
            return pos, vel

        if p.trajectory_type == "moving":
            pos = center + np.array(
                [
                    p.radius * np.sin(u) + 0.065 * np.sin(0.31 * u + 1.40),
                    p.y_radius * np.cos(0.72 * u) + 0.040 * np.sin(1.70 * u),
                    p.z_amplitude * np.sin(1.15 * u + 0.30) + 0.025 * np.sin(2.10 * u),
                ],
                dtype=np.float64,
            )
            vel = np.array(
                [
                    p.radius * w * np.cos(u) + 0.065 * 0.31 * w * np.cos(0.31 * u + 1.40),
                    -p.y_radius * 0.72 * w * np.sin(0.72 * u)
                    + 0.040 * 1.70 * w * np.cos(1.70 * u),
                    p.z_amplitude * 1.15 * w * np.cos(1.15 * u + 0.30)
                    + 0.025 * 2.10 * w * np.cos(2.10 * u),
                ],
                dtype=np.float64,
            )
            return pos, vel

        raise RuntimeError(f"Unhandled trajectory type: {p.trajectory_type}")

    def align_start_to(self, target_position: np.ndarray, t: float = 0.0) -> TrajectoryParams:
        """Translate the trajectory so its position at ``t`` equals ``target_position``.

        This preserves the sampled trajectory shape, phase, frequency, and
        velocity profile. Only the Cartesian center is shifted, which makes the
        first target point consistent with the robot's reset end-effector pose.
        """

        target_position = np.asarray(target_position, dtype=np.float64)
        current_start, _ = self.position_velocity(t)
        center = np.asarray(self.params.center, dtype=np.float64)
        center += target_position - current_start
        self.params = replace(self.params, center=tuple(center.tolist()))
        return self.params

    def sample(self, times: Iterable[float]) -> np.ndarray:
        """Sample desired positions for plotting or visual trace markers."""

        return np.asarray([self.position_velocity(float(t))[0] for t in times], dtype=np.float64)

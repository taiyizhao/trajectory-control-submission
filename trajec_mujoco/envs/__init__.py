"""Gymnasium environment registrations."""

import gymnasium as gym
from gymnasium.envs.registration import registry

from trajec_mujoco.envs.panda_trajectory_env import PandaTrajectoryTrackingEnv


if "PandaTrajectoryTrack-v0" not in registry:
    gym.register(
        id="PandaTrajectoryTrack-v0",
        entry_point="trajec_mujoco.envs:PandaTrajectoryTrackingEnv",
    )

__all__ = ["PandaTrajectoryTrackingEnv"]

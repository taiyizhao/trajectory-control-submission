"""Stable-Baselines3 environment helpers."""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Callable

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize

from trajec_mujoco.envs import PandaTrajectoryTrackingEnv
from trajec_mujoco.utils.config import deep_update


def make_env_fn(
    config: dict[str, Any],
    seed: int,
    rank: int = 0,
    render_mode: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> Callable[[], Monitor]:
    """Create a picklable environment factory for SB3 vector envs."""

    env_config = deep_update(config, overrides or {})

    def _init() -> Monitor:
        env = PandaTrajectoryTrackingEnv(deepcopy(env_config), render_mode=render_mode)
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _init


def make_vec_env(
    config: dict[str, Any],
    n_envs: int,
    seed: int,
    render_mode: str | None = None,
    overrides: dict[str, Any] | None = None,
    vec_env_type: str = "dummy",
    normalize: bool = True,
) -> VecEnv:
    env_fns = [
        make_env_fn(config, seed=seed, rank=i, render_mode=render_mode, overrides=overrides)
        for i in range(n_envs)
    ]
    use_subproc = vec_env_type == "subproc" and n_envs > 1 and os.name != "nt"
    vec_env: VecEnv = SubprocVecEnv(env_fns) if use_subproc else DummyVecEnv(env_fns)
    if normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return vec_env


def eval_profile_overrides(profile: str, base_config: dict[str, Any]) -> dict[str, Any]:
    """Named robustness profiles used for competition evaluation plots."""

    env = base_config.get("env", {})
    default_obs_noise = float(env.get("observation_noise_std", 0.003))
    default_action_noise = float(env.get("action_noise_std", 0.015))
    default_delay = int(env.get("action_delay_steps", 1))

    clean = {
        "env": {
            "randomize_trajectory": False,
            "observation_noise_std": 0.0,
            "action_noise_std": 0.0,
            "action_delay_steps": 0,
            "unreachable_prob": 0.0,
        }
    }

    profiles = {
        "clean": clean,
        "obs_noise": deep_update(clean, {"env": {"observation_noise_std": max(default_obs_noise, 0.006)}}),
        "action_delay": deep_update(clean, {"env": {"action_delay_steps": max(default_delay, 2)}}),
        "action_noise": deep_update(clean, {"env": {"action_noise_std": max(default_action_noise, 0.030)}}),
        "unreachable": deep_update(clean, {"env": {"unreachable_prob": 1.0}}),
        "combined": {
            "env": {
                "randomize_trajectory": True,
                "observation_noise_std": max(default_obs_noise, 0.006),
                "action_noise_std": max(default_action_noise, 0.030),
                "action_delay_steps": max(default_delay, 2),
                "unreachable_prob": 0.35,
            }
        },
    }
    if profile not in profiles:
        raise ValueError(f"Unknown eval profile {profile!r}. Expected one of {sorted(profiles)}.")
    return profiles[profile]

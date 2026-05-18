"""MuJoCo/Gymnasium environment for dynamic 3D end-effector tracking."""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from trajec_mujoco.trajectories import SUPPORTED_TRAJECTORIES, TrajectoryGenerator, TrajectoryParams


DEFAULT_REWARD_WEIGHTS = {
    "position": 3.0,
    "velocity_tracking": 0.45,
    "position_error": 0.0,
    "phase_lag": 0.0,
    "action_magnitude": 0.008,
    "action_smoothness": 0.045,
    "action_jerk": 0.010,
    "joint_velocity": 0.0025,
    "joint_limit": 0.070,
    "orientation": 0.0,
    "large_error": 4.0,
}


class PandaTrajectoryTrackingEnv(gym.Env):
    """Official Franka Panda arm tracking a dynamic Cartesian trajectory.

    The policy can output either normalized joint-target increments or a
    normalized Cartesian end-effector correction. A MuJoCo position actuator
    layer acts as the low-level servo, which mirrors how a real robot stack
    often separates learned high-level motion from stable joint control.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or {}
        env_cfg = self.config.get("env", self.config)

        root = Path(__file__).resolve().parents[2]
        model_path = Path(
            env_cfg.get(
                "model_path",
                root / "assets" / "mujoco_menagerie" / "franka_emika_panda" / "tracking_scene.xml",
            )
        )
        if not model_path.is_absolute():
            model_path = root / model_path

        self.trace_points = int(env_cfg.get("trace_points", 90))
        self._generated_model_path: Path | None = None
        xml = self._load_xml_with_markers(model_path, self.trace_points)
        if "<include " in xml:
            generated_path = model_path.with_name(
                f"{model_path.stem}_generated_{self.trace_points}_{os.getpid()}_{id(self)}.xml"
            )
            generated_path.write_text(xml, encoding="utf-8")
            self._generated_model_path = generated_path
            self.model = mujoco.MjModel.from_xml_path(str(generated_path))
        else:
            self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.render_width = int(env_cfg.get("render_width", 1280))
        self.render_height = int(env_cfg.get("render_height", 720))
        self.camera_name = str(env_cfg.get("camera", "overview"))
        self._renderer: mujoco.Renderer | None = None

        self.joint_names = [f"joint{i}" for i in range(1, 8)]
        self.joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names],
            dtype=np.int32,
        )
        if np.any(self.joint_ids < 0):
            raise RuntimeError("Robot model is missing one or more expected joints.")
        self.qpos_ids = np.array([self.model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.dof_ids = np.array([self.model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=np.int32)
        self.q_lower = self.model.jnt_range[self.joint_ids, 0].copy()
        self.q_upper = self.model.jnt_range[self.joint_ids, 1].copy()

        ee_site_name = str(env_cfg.get("ee_site_name", "ee_site"))
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        if self.ee_site_id < 0 and ee_site_name == "ee_site":
            ee_site_name = "gripper"
            self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        if self.ee_site_id < 0:
            raise RuntimeError(f"Robot model is missing end-effector site {ee_site_name!r}.")
        self.ee_site_name = ee_site_name

        self.n_act = 7
        self.control_mode = str(env_cfg.get("control_mode", "joint_delta"))
        if self.control_mode not in {"joint_delta", "cartesian_delta"}:
            raise ValueError("control_mode must be either 'joint_delta' or 'cartesian_delta'.")
        self.action_dim = 3 if self.control_mode == "cartesian_delta" else self.n_act
        self.frame_skip = int(env_cfg.get("frame_skip", 10))
        self.dt = float(self.model.opt.timestep * self.frame_skip)
        self.episode_steps = int(env_cfg.get("episode_steps", 500))
        self.action_scale = float(env_cfg.get("action_scale", 0.035))
        self.action_filter_alpha = float(np.clip(env_cfg.get("action_filter_alpha", 1.0), 0.0, 1.0))
        self.ik_damping = float(env_cfg.get("ik_damping", 0.05))
        self.nullspace_gain = float(env_cfg.get("nullspace_gain", 0.0))
        self.max_joint_delta = float(env_cfg.get("max_joint_delta", 0.06))
        self.cartesian_feedforward_gain = float(env_cfg.get("cartesian_feedforward_gain", 0.0))
        self.cartesian_feedback_gain = float(env_cfg.get("cartesian_feedback_gain", 0.0))
        self.action_delay_steps = int(env_cfg.get("action_delay_steps", 1))
        self.action_noise_std = float(env_cfg.get("action_noise_std", 0.015))
        self.observation_noise_std = float(env_cfg.get("observation_noise_std", 0.003))
        self.unreachable_prob = float(env_cfg.get("unreachable_prob", 0.10))
        self.preview_steps = tuple(int(v) for v in env_cfg.get("preview_steps", [1, 5, 12]))
        self.randomize_trajectory = bool(env_cfg.get("randomize_trajectory", True))
        self.align_trajectory_start_to_ee = bool(env_cfg.get("align_trajectory_start_to_ee", False))
        self.trajectory_type = str(env_cfg.get("trajectory_type", "circle"))
        trajectory_types = env_cfg.get("trajectory_types", None)
        if trajectory_types is None and self.trajectory_type in {"mixed", "random"}:
            trajectory_types = list(SUPPORTED_TRAJECTORIES)
        self.trajectory_types = tuple(str(v) for v in trajectory_types or [])
        for trajectory_type in self.trajectory_types:
            if trajectory_type not in SUPPORTED_TRAJECTORIES:
                raise ValueError(
                    f"Unknown trajectory type {trajectory_type!r}; expected one of {SUPPORTED_TRAJECTORIES}."
                )
        base_trajectory_type = self.trajectory_types[0] if self.trajectory_types else self.trajectory_type

        reward_cfg = dict(DEFAULT_REWARD_WEIGHTS)
        reward_cfg.update(self.config.get("reward", env_cfg.get("reward", {})))
        self.reward_weights = reward_cfg
        self.position_alpha = float(env_cfg.get("position_alpha", 35.0))
        self.velocity_beta = float(env_cfg.get("velocity_beta", 6.0))
        self.large_error_threshold = float(env_cfg.get("large_error_threshold", 0.32))
        self.success_threshold = float(env_cfg.get("success_threshold", 0.055))

        traj_params = TrajectoryParams(
            trajectory_type=base_trajectory_type,
            center=tuple(env_cfg.get("trajectory_center", [0.50, 0.00, 0.42])),
            radius=float(env_cfg.get("trajectory_radius", 0.12)),
            y_radius=float(env_cfg.get("trajectory_y_radius", 0.10)),
            z_amplitude=float(env_cfg.get("trajectory_z_amplitude", 0.055)),
            frequency=float(env_cfg.get("trajectory_frequency", 0.16)),
        )

        self.rng = np.random.default_rng(env_cfg.get("seed", None))
        self.trajectory = TrajectoryGenerator(traj_params, randomize=self.randomize_trajectory, rng=self.rng)

        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        home_from_cfg = env_cfg.get("home_qpos", None)
        if home_from_cfg is not None:
            self.home_qpos = np.asarray(home_from_cfg, dtype=np.float64)
            if self.home_qpos.shape != (self.n_act,):
                raise ValueError(f"home_qpos must have shape {(self.n_act,)}, got {self.home_qpos.shape}.")
            self.home_ctrl = self.home_qpos.copy()
        elif key_id >= 0:
            self.home_qpos = self.model.key_qpos[key_id, self.qpos_ids].copy()
            self.home_ctrl = self.model.key_ctrl[key_id, : self.n_act].copy()
        else:
            self.home_qpos = np.zeros(self.n_act, dtype=np.float64)
            self.home_ctrl = self.home_qpos.copy()
        self.extra_ctrl = np.asarray(env_cfg.get("extra_ctrl", []), dtype=np.float64)
        self.fixed_joint_qpos = dict(env_cfg.get("fixed_joint_qpos", {}))
        self.q_target = self.home_ctrl.copy()

        orientation_cfg = dict(env_cfg.get("orientation_constraint", {}))
        self.orientation_enabled = bool(orientation_cfg.get("enabled", False))
        self.orientation_include_in_observation = bool(
            orientation_cfg.get("include_in_observation", True)
        )
        self.orientation_feedback_gain = float(orientation_cfg.get("feedback_gain", 0.0))
        self.orientation_reference = str(orientation_cfg.get("reference", "home"))
        desired_xmat = orientation_cfg.get("desired_xmat", None)
        if desired_xmat is not None:
            self.desired_ee_xmat = np.asarray(desired_xmat, dtype=np.float64).reshape(3, 3)
        elif self.orientation_reference == "gripper_down":
            self.desired_ee_xmat = self._compute_gripper_down_xmat()
        else:
            if self.orientation_reference != "home":
                raise ValueError(
                    "orientation_constraint.reference supports 'home', 'gripper_down', or explicit desired_xmat."
                )
            self.desired_ee_xmat = self._compute_home_ee_xmat()
        self.orientation_obs_dim = (
            3 if self.orientation_enabled and self.orientation_include_in_observation else 0
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        self._obs_dim = (
            7
            + 7
            + 3
            + 3
            + 3
            + 3
            + 3 * len(self.preview_steps)
            + 3
            + self.orientation_obs_dim
            + 2
            + self.action_dim
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

        self._geom_ids = self._collect_marker_ids()
        self._hide_all_trace_markers()
        self._reset_episode_state()

    @staticmethod
    def _marker_line(name: str, size: float, rgba: str) -> str:
        return (
            f'    <geom name="{name}" type="sphere" size="{size:.4f}" pos="0 0 -10" '
            f'rgba="{rgba}" contype="0" conaffinity="0" mass="0.0001"/>'
        )

    def _load_xml_with_markers(self, model_path: Path, trace_points: int) -> str:
        xml = model_path.read_text(encoding="utf-8")
        markers = [
            self._marker_line("target_marker", 0.020, "0.10 1.00 0.35 0.95"),
        ]
        for prefix, size, rgba in (
            ("reference_trace", 0.0042, "0.50 0.82 1.00 0.38"),
            ("generated_trace", 0.0045, "0.15 1.00 0.40 0.70"),
            ("actual_trace", 0.0048, "1.00 0.38 0.12 0.82"),
        ):
            for i in range(trace_points):
                markers.append(self._marker_line(f"{prefix}_{i:03d}", size, rgba))
        return xml.replace("    <!-- TRACE_MARKERS -->", "\n".join(markers))

    def _collect_marker_ids(self) -> dict[str, Any]:
        def geom_id(name: str) -> int:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

        return {
            "target": geom_id("target_marker"),
            "reference": [geom_id(f"reference_trace_{i:03d}") for i in range(self.trace_points)],
            "generated": [geom_id(f"generated_trace_{i:03d}") for i in range(self.trace_points)],
            "actual": [geom_id(f"actual_trace_{i:03d}") for i in range(self.trace_points)],
        }

    def _reset_episode_state(self) -> None:
        self.step_count = 0
        self.elapsed_time = 0.0
        self.prev_action = np.zeros(self.action_dim, dtype=np.float64)
        self.prev_prev_action = np.zeros(self.action_dim, dtype=np.float64)
        self.last_applied_action = np.zeros(self.action_dim, dtype=np.float64)
        self.action_buffer = deque(
            [np.zeros(self.action_dim, dtype=np.float64) for _ in range(self.action_delay_steps)]
        )
        self.desired_history: list[np.ndarray] = []
        self.actual_history: list[np.ndarray] = []

    def _hide_all_trace_markers(self) -> None:
        hidden = np.array([0.0, 0.0, -10.0], dtype=np.float64)
        for group in ("reference", "generated", "actual"):
            for gid in self._geom_ids[group]:
                if gid >= 0:
                    self.model.geom_pos[gid] = hidden
                    self.model.geom_rgba[gid, 3] = 0.0
        if self._geom_ids["target"] >= 0:
            self.model.geom_pos[self._geom_ids["target"]] = hidden
            self.model.geom_rgba[self._geom_ids["target"], 3] = 0.0

    def _set_marker(self, gid: int, pos: np.ndarray, alpha: float | None = None) -> None:
        if gid < 0:
            return
        self.model.geom_pos[gid] = pos
        if alpha is not None:
            self.model.geom_rgba[gid, 3] = alpha

    def _set_marker_history(self, group: str, history: list[np.ndarray], base_alpha: float) -> None:
        gids = self._geom_ids[group]
        hidden = np.array([0.0, 0.0, -10.0], dtype=np.float64)
        if not history:
            for gid in gids:
                self._set_marker(gid, hidden, 0.0)
            return

        if len(history) > self.trace_points:
            idx = np.linspace(0, len(history) - 1, self.trace_points).astype(np.int32)
            sampled = [history[i] for i in idx]
        else:
            sampled = history

        n = len(sampled)
        for i, gid in enumerate(gids):
            if i < n:
                alpha = base_alpha * (0.35 + 0.65 * (i + 1) / max(n, 1))
                self._set_marker(gid, sampled[i], alpha)
            else:
                self._set_marker(gid, hidden, 0.0)

    def _update_reference_markers(self) -> None:
        duration = self.episode_steps * self.dt
        times = np.linspace(0.0, duration, self.trace_points)
        positions = self.trajectory.sample(times)
        for gid, pos in zip(self._geom_ids["reference"], positions):
            self._set_marker(gid, pos, 0.42)

    def _update_visual_markers(self, desired_pos: np.ndarray, actual_pos: np.ndarray) -> None:
        self.desired_history.append(desired_pos.copy())
        self.actual_history.append(actual_pos.copy())
        self._set_marker(self._geom_ids["target"], desired_pos, 0.95)
        self._set_marker_history("generated", self.desired_history, 0.62)
        self._set_marker_history("actual", self.actual_history, 0.82)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.trajectory.rng = self.rng

        options = options or {}
        requested_trajectory_type = options.get("trajectory_type", None)
        if requested_trajectory_type is None:
            if self.trajectory_type in {"mixed", "random"}:
                trajectory_type = str(self.rng.choice(self.trajectory_types))
            else:
                trajectory_type = self.trajectory_type
        elif str(requested_trajectory_type) in {"mixed", "random"}:
            trajectory_type = str(self.rng.choice(self.trajectory_types))
        else:
            trajectory_type = str(requested_trajectory_type)
        unreachable = options.get("unreachable", self.rng.random() < self.unreachable_prob)
        self.trajectory.reset(trajectory_type=trajectory_type, unreachable=bool(unreachable))
        self._reset_episode_state()

        mujoco.mj_resetData(self.model, self.data)
        reset_noise = self.rng.normal(0.0, 0.025, size=self.n_act)
        qpos = np.clip(self.home_qpos + reset_noise, self.q_lower + 0.02, self.q_upper - 0.02)
        self.data.qpos[self.qpos_ids] = qpos
        self._apply_fixed_joint_qpos()
        self.data.qvel[self.dof_ids] = self.rng.normal(0.0, 0.01, size=self.n_act)
        self.q_target = np.clip(qpos.copy(), self.q_lower, self.q_upper)
        self.data.ctrl[: self.n_act] = self.q_target
        self._apply_extra_ctrl()
        mujoco.mj_forward(self.model, self.data)
        if self.align_trajectory_start_to_ee:
            self.trajectory.align_start_to(self._ee_pos(), t=self.elapsed_time)

        self._hide_all_trace_markers()
        self._update_reference_markers()
        desired_pos, _ = self.trajectory.position_velocity(self.elapsed_time)
        ee_pos = self._ee_pos()
        self._update_visual_markers(desired_pos, ee_pos)

        obs = self._get_obs()
        return obs, self._get_info(reward_terms={})

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        raw_action = np.asarray(action, dtype=np.float64)
        raw_action = np.clip(raw_action, -1.0, 1.0)

        if self.action_noise_std > 0.0:
            raw_action = np.clip(
                raw_action + self.rng.normal(0.0, self.action_noise_std, size=self.action_dim),
                -1.0,
                1.0,
            )

        self.action_buffer.append(raw_action)
        if len(self.action_buffer) > self.action_delay_steps:
            delayed_action = self.action_buffer.popleft()
        else:
            delayed_action = np.zeros(self.action_dim, dtype=np.float64)

        applied_action = (
            self.action_filter_alpha * delayed_action
            + (1.0 - self.action_filter_alpha) * self.last_applied_action
        )

        if self.control_mode == "cartesian_delta":
            q_delta = self._cartesian_action_to_q_delta(applied_action)
        else:
            q_delta = self.action_scale * applied_action
        q_delta = q_delta + self._cartesian_tracking_prior()
        self.q_target = np.clip(self.q_target + q_delta, self.q_lower, self.q_upper)
        self.data.ctrl[: self.n_act] = self.q_target
        self._apply_extra_ctrl()

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.elapsed_time += self.dt
        self.step_count += 1

        desired_pos, _ = self.trajectory.position_velocity(self.elapsed_time)
        ee_pos = self._ee_pos()
        self._update_visual_markers(desired_pos, ee_pos)

        reward, reward_terms = self._compute_reward(applied_action)

        obs = self._get_obs()
        terminated = not np.isfinite(obs).all()
        truncated = self.step_count >= self.episode_steps

        self.prev_prev_action = self.prev_action.copy()
        self.prev_action = applied_action.copy()
        self.last_applied_action = applied_action.copy()

        return obs, float(reward), bool(terminated), bool(truncated), self._get_info(reward_terms)

    def _qpos(self) -> np.ndarray:
        return self.data.qpos[self.qpos_ids].copy()

    def _apply_extra_ctrl(self) -> None:
        if self.model.nu <= self.n_act or self.extra_ctrl.size == 0:
            return
        n = min(self.model.nu - self.n_act, self.extra_ctrl.size)
        self.data.ctrl[self.n_act : self.n_act + n] = self.extra_ctrl[:n]

    def _apply_fixed_joint_qpos(self) -> None:
        for joint_name, joint_value in self.fixed_joint_qpos.items():
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_name))
            if joint_id >= 0:
                self.data.qpos[self.model.jnt_qposadr[joint_id]] = float(joint_value)

    def _qvel(self) -> np.ndarray:
        return self.data.qvel[self.dof_ids].copy()

    def _ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()

    def _ee_xmat(self) -> np.ndarray:
        return self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()

    def _ee_vel(self) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        return jacp @ self.data.qvel

    def _ee_jacobians(self) -> tuple[np.ndarray, np.ndarray]:
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        return jacp[:, self.dof_ids].copy(), jacr[:, self.dof_ids].copy()

    def _ee_jacobian(self) -> np.ndarray:
        jacp, _ = self._ee_jacobians()
        return jacp

    def _cartesian_delta_to_q_delta(self, desired_delta: np.ndarray, include_nullspace: bool) -> np.ndarray:
        return self._task_delta_to_q_delta(desired_delta, None, include_nullspace)

    def _task_delta_to_q_delta(
        self,
        cartesian_delta: np.ndarray,
        orientation_delta: np.ndarray | None,
        include_nullspace: bool,
    ) -> np.ndarray:
        jacp, jacr = self._ee_jacobians()
        if orientation_delta is None:
            jac = jacp
            desired_delta = cartesian_delta
        else:
            jac = np.vstack([jacp, jacr])
            desired_delta = np.concatenate([cartesian_delta, orientation_delta])

        damping_matrix = (self.ik_damping**2) * np.eye(jac.shape[0])
        jac_pinv = jac.T @ np.linalg.inv(jac @ jac.T + damping_matrix)
        q_delta = jac_pinv @ desired_delta

        if include_nullspace and self.nullspace_gain > 0.0:
            nullspace = np.eye(self.n_act) - jac_pinv @ jac
            posture_error = self.home_qpos - self._qpos()
            q_delta = q_delta + self.nullspace_gain * (nullspace @ posture_error)

        return np.clip(q_delta, -self.max_joint_delta, self.max_joint_delta)

    def _cartesian_action_to_q_delta(self, applied_action: np.ndarray) -> np.ndarray:
        return self._cartesian_delta_to_q_delta(self.action_scale * applied_action, include_nullspace=True)

    def _cartesian_tracking_prior(self) -> np.ndarray:
        has_orientation_prior = self.orientation_enabled and self.orientation_feedback_gain != 0.0
        if (
            self.cartesian_feedforward_gain == 0.0
            and self.cartesian_feedback_gain == 0.0
            and not has_orientation_prior
        ):
            return np.zeros(self.n_act, dtype=np.float64)

        desired_pos, desired_vel = self.trajectory.position_velocity(self.elapsed_time)
        ee_pos = self._ee_pos()
        cartesian_delta = (
            self.cartesian_feedforward_gain * desired_vel * self.dt
            + self.cartesian_feedback_gain * (desired_pos - ee_pos) * self.dt
        )
        orientation_delta = None
        if self.orientation_enabled and self.orientation_feedback_gain != 0.0:
            orientation_delta = self.orientation_feedback_gain * self._orientation_error_vector() * self.dt
        return self._task_delta_to_q_delta(cartesian_delta, orientation_delta, include_nullspace=False)

    def _compute_home_ee_xmat(self) -> np.ndarray:
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        saved_ctrl = self.data.ctrl.copy()

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_ids] = self.home_qpos
        self._apply_fixed_joint_qpos()
        self.data.qvel[:] = 0.0
        self.data.ctrl[: self.n_act] = self.home_ctrl
        self._apply_extra_ctrl()
        mujoco.mj_forward(self.model, self.data)
        desired_xmat = self._ee_xmat()

        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        self.data.ctrl[:] = saved_ctrl
        mujoco.mj_forward(self.model, self.data)
        return desired_xmat

    def _compute_gripper_down_xmat(self) -> np.ndarray:
        home_xmat = self._compute_home_ee_xmat()
        z_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        x_axis = home_xmat[:, 0].copy()
        x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
        if np.linalg.norm(x_axis) < 1e-8:
            x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        return np.column_stack([x_axis, y_axis, z_axis])

    @staticmethod
    def _rotation_error_vector(current_xmat: np.ndarray, desired_xmat: np.ndarray) -> np.ndarray:
        return 0.5 * (
            np.cross(current_xmat[:, 0], desired_xmat[:, 0])
            + np.cross(current_xmat[:, 1], desired_xmat[:, 1])
            + np.cross(current_xmat[:, 2], desired_xmat[:, 2])
        )

    @staticmethod
    def _rotation_angle_error(current_xmat: np.ndarray, desired_xmat: np.ndarray) -> float:
        error_xmat = desired_xmat.T @ current_xmat
        cos_angle = 0.5 * (float(np.trace(error_xmat)) - 1.0)
        return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    def _orientation_error_vector(self) -> np.ndarray:
        if not self.orientation_enabled:
            return np.zeros(3, dtype=np.float64)
        return self._rotation_error_vector(self._ee_xmat(), self.desired_ee_xmat)

    def _orientation_error(self) -> float:
        if not self.orientation_enabled:
            return 0.0
        return self._rotation_angle_error(self._ee_xmat(), self.desired_ee_xmat)

    def _tool_axis_error(self) -> float:
        if not self.orientation_enabled:
            return 0.0
        current_axis = self._ee_xmat()[:, 2]
        desired_axis = self.desired_ee_xmat[:, 2]
        cos_angle = float(np.dot(current_axis, desired_axis))
        return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    def _get_obs(self) -> np.ndarray:
        qpos = self._qpos()
        qvel = self._qvel()
        ee_pos = self._ee_pos()
        ee_vel = self._ee_vel()
        desired_pos, desired_vel = self.trajectory.position_velocity(self.elapsed_time)
        future_positions = [
            self.trajectory.position_velocity(self.elapsed_time + step * self.dt)[0]
            for step in self.preview_steps
        ]
        error = desired_pos - ee_pos
        orientation_obs = []
        if self.orientation_enabled and self.orientation_include_in_observation:
            orientation_obs.append(self._orientation_error_vector())
        phase = 2.0 * np.pi * self.step_count / max(self.episode_steps, 1)
        obs = np.concatenate(
            [
                qpos,
                qvel,
                ee_pos,
                ee_vel,
                desired_pos,
                desired_vel,
                *future_positions,
                error,
                *orientation_obs,
                np.array([np.sin(phase), np.cos(phase)], dtype=np.float64),
                self.prev_action,
            ]
        ).astype(np.float32)

        if self.observation_noise_std > 0.0:
            noise = np.zeros_like(obs)
            measured_state_dim = 7 + 7 + 3 + 3
            noise[:measured_state_dim] = self.rng.normal(
                0.0,
                self.observation_noise_std,
                size=measured_state_dim,
            ).astype(np.float32)
            error_start = 7 + 7 + 3 + 3 + 3 + 3 + 3 * len(self.preview_steps)
            noise[error_start : error_start + 3] = self.rng.normal(
                0.0,
                self.observation_noise_std,
                size=3,
            ).astype(np.float32)
            obs = obs + noise
        return obs

    def _joint_limit_penalty(self, qpos: np.ndarray) -> float:
        joint_range = self.q_upper - self.q_lower
        margin = np.maximum(0.12 * joint_range, 1e-6)
        lower = np.clip((self.q_lower + margin - qpos) / margin, 0.0, 1.0)
        upper = np.clip((qpos - (self.q_upper - margin)) / margin, 0.0, 1.0)
        return float(np.sum((lower + upper) ** 2))

    def _compute_reward(self, applied_action: np.ndarray) -> tuple[float, dict[str, float]]:
        ee_pos = self._ee_pos()
        ee_vel = self._ee_vel()
        desired_pos, desired_vel = self.trajectory.position_velocity(self.elapsed_time)
        position_error = float(np.linalg.norm(ee_pos - desired_pos))
        velocity_error = float(np.linalg.norm(ee_vel - desired_vel))
        desired_speed = float(np.linalg.norm(desired_vel))
        if desired_speed > 1e-8:
            tangent = desired_vel / desired_speed
            phase_lag = float(abs(np.dot(ee_pos - desired_pos, tangent)))
        else:
            phase_lag = 0.0
        action_magnitude = float(np.sum(applied_action**2))
        action_delta = float(np.sum((applied_action - self.prev_action) ** 2))
        action_jerk = float(np.sum((applied_action - 2.0 * self.prev_action + self.prev_prev_action) ** 2))
        joint_velocity = float(np.sum(self._qvel() ** 2))
        joint_limit = self._joint_limit_penalty(self._qpos())
        orientation_error = self._orientation_error()
        large_error = max(0.0, position_error - self.large_error_threshold) ** 2

        terms = {
            "reward_position": self.reward_weights["position"]
            * float(np.exp(-self.position_alpha * position_error**2)),
            "reward_velocity_tracking": self.reward_weights["velocity_tracking"]
            * float(np.exp(-self.velocity_beta * velocity_error**2)),
            "penalty_position_error": self.reward_weights["position_error"] * position_error,
            "penalty_phase_lag": self.reward_weights["phase_lag"] * phase_lag,
            "penalty_action_magnitude": self.reward_weights["action_magnitude"] * action_magnitude,
            "penalty_action_smoothness": self.reward_weights["action_smoothness"] * action_delta,
            "penalty_action_jerk": self.reward_weights["action_jerk"] * action_jerk,
            "penalty_joint_velocity": self.reward_weights["joint_velocity"] * joint_velocity,
            "penalty_joint_limit": self.reward_weights["joint_limit"] * joint_limit,
            "penalty_orientation": self.reward_weights["orientation"] * orientation_error**2,
            "penalty_large_error": self.reward_weights["large_error"] * large_error,
        }
        reward = (
            terms["reward_position"]
            + terms["reward_velocity_tracking"]
            - terms["penalty_position_error"]
            - terms["penalty_phase_lag"]
            - terms["penalty_action_magnitude"]
            - terms["penalty_action_smoothness"]
            - terms["penalty_action_jerk"]
            - terms["penalty_joint_velocity"]
            - terms["penalty_joint_limit"]
            - terms["penalty_orientation"]
            - terms["penalty_large_error"]
        )
        terms.update(
            {
                "position_error": position_error,
                "velocity_error": velocity_error,
                "phase_lag": phase_lag,
                "action_magnitude": float(np.sqrt(action_magnitude)),
                "action_delta": float(np.sqrt(action_delta)),
                "action_jerk": float(np.sqrt(action_jerk)),
                "joint_velocity_norm": float(np.linalg.norm(self._qvel())),
                "joint_limit_penalty": joint_limit,
                "orientation_error": orientation_error,
                "orientation_error_deg": float(np.degrees(orientation_error)),
                "tool_axis_error_deg": float(np.degrees(self._tool_axis_error())),
                "success": float(position_error < self.success_threshold),
            }
        )
        return reward, terms

    def _get_info(self, reward_terms: dict[str, float]) -> dict[str, Any]:
        desired_pos, desired_vel = self.trajectory.position_velocity(self.elapsed_time)
        ee_pos = self._ee_pos()
        ee_vel = self._ee_vel()
        orientation_error = self._orientation_error()
        tool_axis_error = self._tool_axis_error()
        info: dict[str, Any] = {
            "time": float(self.elapsed_time),
            "step": int(self.step_count),
            "trajectory_type": self.trajectory.params.trajectory_type,
            "unreachable": bool(self.trajectory.params.unreachable),
            "control_mode": self.control_mode,
            "ee_pos": ee_pos.copy(),
            "ee_vel": ee_vel.copy(),
            "desired_pos": desired_pos.copy(),
            "desired_vel": desired_vel.copy(),
            "tracking_error": float(np.linalg.norm(ee_pos - desired_pos)),
            "orientation_error": orientation_error,
            "orientation_error_deg": float(np.degrees(orientation_error)),
            "tool_axis_error_deg": float(np.degrees(tool_axis_error)),
            "qpos": self._qpos(),
            "qvel": self._qvel(),
            "applied_action": self.last_applied_action.copy(),
        }
        info.update(reward_terms)
        return info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.render_height, width=self.render_width)
        mujoco.mj_forward(self.model, self.data)
        self._renderer.update_scene(self.data, camera=self.camera_name)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._generated_model_path is not None:
            try:
                self._generated_model_path.unlink(missing_ok=True)
            except OSError:
                pass
            self._generated_model_path = None

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from trajec_mujoco.utils.config import deep_update, load_config
from trajec_mujoco.utils.plotting import make_evaluation_plots
from trajec_mujoco.utils.sb3_helpers import eval_profile_overrides, make_env_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO trajectory tracker.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--vecnormalize", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--profiles", nargs="*", default=None)
    parser.add_argument("--trajectory-type", type=str, default=None, choices=["circle", "figure8", "moving", "mixed"])
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def infer_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None, Path]:
    if args.run_dir is None and (args.config is None or args.model is None):
        raise SystemExit("Provide --run-dir or both --config and --model.")

    if args.run_dir is not None:
        run_dir = args.run_dir
        config_path = args.config or run_dir / "config.yaml"
        best_model = run_dir / "models" / "best_model.zip"
        final_model = run_dir / "models" / "final_model.zip"
        model_path = args.model or (best_model if best_model.exists() else final_model)
        vecnorm_path = args.vecnormalize or run_dir / "models" / "vecnormalize.pkl"
        out_dir = args.out_dir or run_dir / "evaluation"
    else:
        config_path = args.config
        model_path = args.model
        vecnorm_path = args.vecnormalize
        out_dir = args.out_dir or PROJECT_ROOT / "results" / "evaluation"

    if vecnorm_path is not None and not vecnorm_path.exists():
        vecnorm_path = None
    return Path(config_path), Path(model_path), vecnorm_path, Path(out_dir)


def make_eval_env(config: dict, profile: str, seed: int, vecnorm_path: Path | None):
    profile_config = deep_update(config, eval_profile_overrides(profile, config))
    env_fn = make_env_fn(profile_config, seed=seed, rank=0)
    vec_env = DummyVecEnv([env_fn])
    if vecnorm_path is not None:
        vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    return vec_env


def row_from_info(profile: str, episode: int, reward: float, info: dict) -> dict:
    desired = np.asarray(info["desired_pos"])
    ee = np.asarray(info["ee_pos"])
    qpos = np.asarray(info["qpos"])
    qvel = np.asarray(info["qvel"])
    row = {
        "profile": profile,
        "episode": episode,
        "time": info["time"],
        "step": info["step"],
        "reward": reward,
        "tracking_error": info["tracking_error"],
        "action_delta": info.get("action_delta", 0.0),
        "action_jerk": info.get("action_jerk", 0.0),
        "joint_velocity_norm": info.get("joint_velocity_norm", float(np.linalg.norm(qvel))),
        "orientation_error": info.get("orientation_error", 0.0),
        "orientation_error_deg": info.get("orientation_error_deg", 0.0),
        "tool_axis_error_deg": info.get("tool_axis_error_deg", 0.0),
        "success": info.get("success", 0.0),
        "trajectory_type": info.get("trajectory_type", ""),
        "unreachable": info.get("unreachable", False),
    }
    for i, axis in enumerate("xyz"):
        row[f"desired_{axis}"] = desired[i]
        row[f"ee_{axis}"] = ee[i]
    for i in range(len(qpos)):
        row[f"q{i + 1}"] = qpos[i]
        row[f"qdot{i + 1}"] = qvel[i]
    for key, value in info.items():
        if key.startswith("reward_") or key.startswith("penalty_"):
            row[key] = value
    return row


def summarize(profile: str, episode_rows: list[dict]) -> dict:
    df = pd.DataFrame(episode_rows)
    err = df["tracking_error"].to_numpy()
    summary = {
        "profile": profile,
        "mean_error": float(np.mean(err)),
        "rmse_error": float(np.sqrt(np.mean(err**2))),
        "max_error": float(np.max(err)),
        "mean_action_delta": float(df["action_delta"].mean()),
        "mean_joint_velocity": float(df["joint_velocity_norm"].mean()),
        "success_rate": float(df["success"].mean()),
        "mean_reward": float(df["reward"].mean()),
    }
    if "orientation_error" in df.columns:
        ori = df["orientation_error"].to_numpy()
        summary["mean_orientation_error_deg"] = float(np.degrees(np.mean(ori)))
        summary["max_orientation_error_deg"] = float(np.degrees(np.max(ori)))
    if "tool_axis_error_deg" in df.columns:
        summary["mean_tool_axis_error_deg"] = float(df["tool_axis_error_deg"].mean())
        summary["max_tool_axis_error_deg"] = float(df["tool_axis_error_deg"].max())
    return summary


def main() -> None:
    args = parse_args()
    config_path, model_path, vecnorm_path, out_dir = infer_paths(args)
    config = load_config(config_path)
    if args.trajectory_type is not None:
        config = deep_update(config, {"env": {"trajectory_type": args.trajectory_type}})
    eval_cfg = config.get("eval", {})
    episodes = int(args.episodes or eval_cfg.get("episodes", 3))
    profiles = args.profiles or list(eval_cfg.get("profiles", ["clean"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    model = PPO.load(str(model_path), device=config.get("train", {}).get("device", "cuda"))

    all_rows: list[dict] = []
    summaries: list[dict] = []
    for profile in profiles:
        profile_rows: list[dict] = []
        vec_env = make_eval_env(config, profile=profile, seed=args.seed, vecnorm_path=vecnorm_path)
        for episode in range(episodes):
            obs = vec_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = vec_env.step(action)
                row = row_from_info(profile, episode, float(rewards[0]), infos[0])
                profile_rows.append(row)
                all_rows.append(row)
                done = bool(dones[0])
        vec_env.close()
        summaries.append(summarize(profile, profile_rows))

    pd.DataFrame(all_rows).to_csv(out_dir / "tracking_log.csv", index=False)
    pd.DataFrame(summaries).to_csv(out_dir / "summary.csv", index=False)
    plots = make_evaluation_plots(out_dir)
    print(f"Saved evaluation CSVs and {len(plots)} plots to {out_dir}")


if __name__ == "__main__":
    main()

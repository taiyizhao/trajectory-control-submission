from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import imageio.v2 as imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from trajec_mujoco.utils.config import deep_update, load_config
from trajec_mujoco.utils.sb3_helpers import eval_profile_overrides, make_env_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a 10-30s MuJoCo tracking video.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--vecnormalize", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--profile", type=str, default="clean")
    parser.add_argument("--trajectory-type", type=str, default=None, choices=["circle", "figure8", "moving"])
    parser.add_argument("--random-policy", action="store_true")
    parser.add_argument("--seed", type=int, default=321)
    return parser.parse_args()


def infer_paths(args: argparse.Namespace) -> tuple[Path, Path | None, Path | None, Path]:
    if args.run_dir is not None:
        run_dir = args.run_dir
        config_path = args.config or run_dir / "config.yaml"
        best_model = run_dir / "models" / "best_model.zip"
        final_model = run_dir / "models" / "final_model.zip"
        model_path = args.model or (best_model if best_model.exists() else final_model)
        vecnorm_path = args.vecnormalize or run_dir / "models" / "vecnormalize.pkl"
        out_path = args.out or run_dir / "videos" / "tracking_demo.mp4"
    else:
        config_path = args.config or PROJECT_ROOT / "configs" / "default.yaml"
        model_path = args.model
        vecnorm_path = args.vecnormalize
        out_path = args.out or PROJECT_ROOT / "results" / "videos" / "tracking_demo.mp4"

    if args.random_policy:
        model_path = None
        vecnorm_path = None
    if vecnorm_path is not None and not vecnorm_path.exists():
        vecnorm_path = None
    return Path(config_path), model_path, vecnorm_path, Path(out_path)


def main() -> None:
    args = parse_args()
    config_path, model_path, vecnorm_path, out_path = infer_paths(args)
    config = load_config(config_path)
    video_cfg = config.get("video", {})
    fps = int(args.fps or video_cfg.get("fps", 50))
    seconds = float(args.seconds or video_cfg.get("seconds", 20))
    trajectory_type = args.trajectory_type or video_cfg.get("trajectory_type", config.get("env", {}).get("trajectory_type", "circle"))

    overrides = eval_profile_overrides(args.profile, config)
    overrides = deep_update(overrides, {"env": {"trajectory_type": trajectory_type, "trace_points": 120}})
    render_config = deep_update(config, overrides)

    env_holder = {}

    def env_factory():
        env = make_env_fn(render_config, seed=args.seed, rank=0, render_mode="rgb_array")()
        env_holder["env"] = env.env
        return env

    vec_env = DummyVecEnv([env_factory])
    if vecnorm_path is not None:
        vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = None
    if model_path is not None:
        model = PPO.load(str(model_path), device=config.get("train", {}).get("device", "cuda"))

    obs = vec_env.reset()
    base_env = env_holder["env"]
    frames = [base_env.render()]
    n_steps = max(1, int(seconds * fps))
    for _ in range(n_steps):
        if model is None:
            action = [base_env.action_space.sample()]
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = vec_env.step(action)
        frame = base_env.render()
        if frame is not None:
            frames.append(frame)
        if bool(dones[0]):
            obs = vec_env.reset()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps, macro_block_size=16)
    vec_env.close()
    print(f"Saved video to {out_path}")


if __name__ == "__main__":
    main()

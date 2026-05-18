from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import VecNormalize

from trajec_mujoco.utils.config import load_config, save_config
from trajec_mujoco.utils.sb3_helpers import make_vec_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO for MuJoCo 3D trajectory tracking.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    parser.add_argument("--resume-model", type=Path, default=None)
    parser.add_argument("--resume-vecnormalize", type=Path, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_cfg = config.setdefault("train", {})

    seed = int(train_cfg.get("seed", 42))
    total_timesteps = int(args.timesteps or train_cfg.get("total_timesteps", 300000))
    n_envs = int(args.n_envs or train_cfg.get("n_envs", 4))
    device = args.device or str(train_cfg.get("device", "cuda"))
    normalize = bool(train_cfg.get("normalize", True))
    learning_rate = float(args.learning_rate or train_cfg.get("learning_rate", 3e-4))
    train_cfg["learning_rate"] = learning_rate
    if args.resume_run_dir is not None:
        train_cfg["resume_run_dir"] = str(args.resume_run_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or config.get("project", {}).get("name", "franka_panda_tracking")
    run_dir = PROJECT_ROOT / "results" / "runs" / f"{timestamp}_{run_name}"
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")

    vec_env = make_vec_env(
        config,
        n_envs=n_envs,
        seed=seed,
        vec_env_type=str(train_cfg.get("vec_env", "dummy")),
        normalize=normalize and args.resume_run_dir is None,
    )
    if args.resume_run_dir is not None and normalize:
        resume_vecnorm = args.resume_vecnormalize or args.resume_run_dir / "models" / "vecnormalize.pkl"
        vec_env = VecNormalize.load(str(resume_vecnorm), vec_env)
        vec_env.training = True
        vec_env.norm_reward = True

    eval_env = make_vec_env(
        config,
        n_envs=1,
        seed=seed + 10_000,
        overrides={
            "env": {
                "randomize_trajectory": False,
                "observation_noise_std": 0.0,
                "action_noise_std": 0.0,
                "action_delay_steps": 0,
                "unreachable_prob": 0.0,
            }
        },
        normalize=normalize,
    )
    if isinstance(eval_env, VecNormalize):
        eval_env.training = False
        eval_env.norm_reward = False

    policy_kwargs = {
        "activation_fn": th.nn.Tanh,
        "net_arch": list(train_cfg.get("policy_hidden_sizes", [256, 256, 128])),
    }

    if args.resume_run_dir is not None:
        best_model = args.resume_run_dir / "models" / "best_model.zip"
        final_model = args.resume_run_dir / "models" / "final_model.zip"
        resume_model = args.resume_model or (best_model if best_model.exists() else final_model)
        model = PPO.load(str(resume_model), env=vec_env, device=device)
        model.learning_rate = learning_rate
        model.lr_schedule = get_schedule_fn(learning_rate)
        model.clip_range = get_schedule_fn(float(train_cfg.get("clip_range", 0.20)))
        model.ent_coef = float(train_cfg.get("ent_coef", model.ent_coef))
        model.vf_coef = float(train_cfg.get("vf_coef", model.vf_coef))
        model.max_grad_norm = float(train_cfg.get("max_grad_norm", model.max_grad_norm))
        model.n_epochs = int(train_cfg.get("n_epochs", model.n_epochs))
        model.batch_size = int(train_cfg.get("batch_size", model.batch_size))
        model.tensorboard_log = str(run_dir / "tensorboard")
        model.verbose = 1
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=learning_rate,
            n_steps=int(train_cfg.get("n_steps", 1024)),
            batch_size=int(train_cfg.get("batch_size", 256)),
            n_epochs=int(train_cfg.get("n_epochs", 10)),
            gamma=float(train_cfg.get("gamma", 0.985)),
            gae_lambda=float(train_cfg.get("gae_lambda", 0.94)),
            clip_range=float(train_cfg.get("clip_range", 0.20)),
            ent_coef=float(train_cfg.get("ent_coef", 0.002)),
            vf_coef=float(train_cfg.get("vf_coef", 0.50)),
            max_grad_norm=float(train_cfg.get("max_grad_norm", 0.50)),
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(run_dir / "tensorboard"),
            verbose=1,
            seed=seed,
            device=device,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(int(train_cfg.get("checkpoint_freq", 50000)) // max(n_envs, 1), 1),
        save_path=str(model_dir / "checkpoints"),
        name_prefix="ppo_franka_panda_tracking",
        save_replay_buffer=False,
        save_vecnormalize=normalize,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(run_dir / "eval"),
        eval_freq=max(int(train_cfg.get("eval_freq", 10000)) // max(n_envs, 1), 1),
        n_eval_episodes=int(config.get("eval", {}).get("episodes", 3)),
        deterministic=True,
        render=False,
    )

    print(f"Run directory: {run_dir}")
    if args.resume_run_dir is not None:
        print(f"Fine-tuning PPO from {args.resume_run_dir}")
    reset_num_timesteps = args.resume_run_dir is None
    learn_timesteps = total_timesteps

    timestep_label = "additional timesteps" if not reset_num_timesteps else "timesteps"
    print(
        f"Training PPO for {total_timesteps:,} {timestep_label} on device={device}, "
        f"n_envs={n_envs}, learning_rate={learning_rate:g}"
    )
    if not reset_num_timesteps:
        print(f"Starting from {model.num_timesteps:,}; requesting {total_timesteps:,} additional steps.")
    model.learn(
        total_timesteps=learn_timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
        reset_num_timesteps=reset_num_timesteps,
    )

    model.save(model_dir / "final_model")
    if isinstance(vec_env, VecNormalize):
        vec_env.save(model_dir / "vecnormalize.pkl")
    vec_env.close()
    eval_env.close()
    print(f"Saved final model to {model_dir / 'final_model.zip'}")


if __name__ == "__main__":
    main()

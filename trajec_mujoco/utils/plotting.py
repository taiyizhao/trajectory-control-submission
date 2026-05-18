"""Evaluation plots for trajectory tracking experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


AXES = ("x", "y", "z")


def _first_episode(df: pd.DataFrame) -> pd.DataFrame:
    first_profile = df["profile"].iloc[0]
    first_episode = int(df.loc[df["profile"] == first_profile, "episode"].iloc[0])
    return df[(df["profile"] == first_profile) & (df["episode"] == first_episode)].copy()


def make_evaluation_plots(eval_dir: str | Path) -> list[Path]:
    eval_dir = Path(eval_dir)
    tracking_csv = eval_dir / "tracking_log.csv"
    summary_csv = eval_dir / "summary.csv"
    plots_dir = eval_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(tracking_csv)
    summary = pd.read_csv(summary_csv) if summary_csv.exists() else None
    created: list[Path] = []
    sns.set_theme(style="whitegrid", context="talk")

    sample = _first_episode(df)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(sample["desired_x"], sample["desired_y"], sample["desired_z"], color="#44aaff", lw=2.5, label="desired")
    ax.plot(sample["ee_x"], sample["ee_y"], sample["ee_z"], color="#ff6a2a", lw=2.5, label="actual")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("3D Desired vs Actual End-Effector Trajectory")
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "desired_vs_actual_3d.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    created.append(path)

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.lineplot(data=df, x="time", y="tracking_error", hue="profile", errorbar=None, ax=ax)
    ax.set_title("Tracking Error Over Time")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("error [m]")
    fig.tight_layout()
    path = plots_dir / "tracking_error_over_time.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    created.append(path)

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.lineplot(data=df, x="time", y="action_delta", hue="profile", errorbar=None, ax=ax)
    ax.set_title("Action Smoothness Over Time")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("||a_t - a_{t-1}||")
    fig.tight_layout()
    path = plots_dir / "action_smoothness_over_time.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    created.append(path)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for axis_name, ax in zip(AXES, axes):
        ax.plot(sample["time"], sample[f"desired_{axis_name}"], color="#44aaff", lw=2.0, label="desired")
        ax.plot(sample["time"], sample[f"ee_{axis_name}"], color="#ff6a2a", lw=2.0, label="actual")
        ax.set_ylabel(f"{axis_name} [m]")
    axes[0].set_title("Cartesian Position Tracking by Axis")
    axes[-1].set_xlabel("time [s]")
    axes[0].legend(loc="best")
    fig.tight_layout()
    path = plots_dir / "xyz_position_vs_time.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    created.append(path)

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.lineplot(data=df, x="time", y="joint_velocity_norm", hue="profile", errorbar=None, ax=ax)
    ax.set_title("Joint Velocity Magnitude")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("||qdot||")
    fig.tight_layout()
    path = plots_dir / "joint_velocity_over_time.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    created.append(path)

    if "orientation_error_deg" in df.columns:
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.lineplot(data=df, x="time", y="orientation_error_deg", hue="profile", errorbar=None, ax=ax)
        ax.set_title("End-Effector Orientation Error Over Time")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("orientation error [deg]")
        fig.tight_layout()
        path = plots_dir / "orientation_error_over_time.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        created.append(path)

    if "tool_axis_error_deg" in df.columns:
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.lineplot(data=df, x="time", y="tool_axis_error_deg", hue="profile", errorbar=None, ax=ax)
        ax.set_title("Gripper Down-Axis Error Over Time")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("tool axis error [deg]")
        fig.tight_layout()
        path = plots_dir / "tool_axis_error_over_time.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        created.append(path)

    reward_cols = [
        "reward_position",
        "reward_velocity_tracking",
        "penalty_position_error",
        "penalty_phase_lag",
        "penalty_action_magnitude",
        "penalty_action_smoothness",
        "penalty_action_jerk",
        "penalty_joint_velocity",
        "penalty_joint_limit",
        "penalty_orientation",
        "penalty_large_error",
    ]
    available = [c for c in reward_cols if c in sample.columns]
    if available:
        fig, ax = plt.subplots(figsize=(11, 6))
        for col in available:
            ax.plot(sample["time"], sample[col], lw=1.8, label=col)
        ax.set_title("Reward Components")
        ax.set_xlabel("time [s]")
        ax.legend(ncol=2, fontsize=9)
        fig.tight_layout()
        path = plots_dir / "reward_components_over_time.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        created.append(path)

    if summary is not None and "rmse_error" in summary.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=summary, x="profile", y="rmse_error", ax=ax, color="#4c9fd8")
        ax.set_title("Robustness Summary: RMSE by Uncertainty Profile")
        ax.set_xlabel("profile")
        ax.set_ylabel("RMSE [m]")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        path = plots_dir / "robustness_rmse_summary.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        created.append(path)

    return created

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont


DEFAULT_RUN = PROJECT_ROOT / "results" / "runs" / "panda_ppo_trajectory_tracker"
DEFAULT_OUT = PROJECT_ROOT
DEFAULT_EVAL_DIR = "evaluation"

DESIRED = "#64c7ff"
GENERATED = "#38e082"
ACTUAL = "#ff8a2a"
INK = "#eef5f7"
MUTED = "#aab8bf"
BG = "#11181d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create final competition figures and demo video.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--eval-dir-name", type=str, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--clip-seconds", type=float, default=6.0)
    parser.add_argument("--title-seconds", type=float, default=2.5)
    parser.add_argument("--metrics-seconds", type=float, default=3.5)
    return parser.parse_args()


def final_eval_dir(run_dir: Path, eval_dir_name: str) -> Path:
    return run_dir / eval_dir_name


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def first_clean_episode(df: pd.DataFrame) -> pd.DataFrame:
    clean = df[df["profile"] == "clean"]
    episode = int(clean["episode"].iloc[0])
    return clean[clean["episode"] == episode].copy()


def read_clean_summary(summary_path: Path) -> pd.Series:
    summary = pd.read_csv(summary_path)
    return summary[summary["profile"] == "clean"].iloc[0]


def draw_box(ax, xy: tuple[float, float], size: tuple[float, float], title: str, body: str, color: str) -> None:
    from matplotlib.patches import FancyBboxPatch

    x, y = xy
    w, h = size
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.8,
        edgecolor=color,
        facecolor="#f8fbfd",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.055, title, ha="center", va="top", fontsize=14, fontweight="bold", color="#172129")
    ax.text(x + w / 2, y + h / 2 - 0.025, body, ha="center", va="center", fontsize=10.5, color="#40515b", linespacing=1.35)


def arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=2.0, color="#455a64"),
    )


def make_system_design_figure(out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(20, 7))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.patch.set_facecolor("white")

    ax.text(
        0.5,
        0.93,
        "Complete RL System for Dynamic 3D Cartesian Tracking",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color="#172129",
    )
    ax.text(
        0.5,
        0.875,
        "Official Franka Panda + MuJoCo + Gymnasium-style environment + Stable-Baselines3 PPO",
        ha="center",
        va="center",
        fontsize=13,
        color="#5f6f78",
    )

    boxes = [
        ((0.025, 0.47), (0.13, 0.25), "Trajectory Generator", "circle\nfigure-8\nmoving target\nstart aligned to EE", DESIRED),
        ((0.189, 0.47), (0.13, 0.25), "MuJoCo Environment", "official Panda model\n50 Hz control\nvisual markers", "#546e7a"),
        ((0.353, 0.47), (0.13, 0.25), "Observation", "robot state\ntarget state\nshort preview\naction history", GENERATED),
        ((0.517, 0.47), (0.13, 0.25), "PPO Policy", "7D joint residual\ndeterministic eval\nVecNormalize", ACTUAL),
        ((0.681, 0.47), (0.13, 0.25), "Control Command", "PPO residual\n+ damped Jacobian\n+ low-pass filter", "#7c4dff"),
        ((0.845, 0.47), (0.13, 0.25), "Evaluation", "tracking RMSE\naction delta\njoint velocity\nrobustness", "#009688"),
    ]
    for xy, size, title, body, color in boxes:
        draw_box(ax, xy, size, title, body, color)

    arrow(ax, (0.155, 0.595), (0.189, 0.595))
    arrow(ax, (0.319, 0.595), (0.353, 0.595))
    arrow(ax, (0.483, 0.595), (0.517, 0.595))
    arrow(ax, (0.647, 0.595), (0.681, 0.595))
    arrow(ax, (0.811, 0.595), (0.845, 0.595))

    draw_box(
        ax,
        (0.07, 0.08),
        (0.40, 0.23),
        "Reward Terms",
        "position + velocity tracking\nphase lag + large error penalties\naction smoothness + jerk\njoint velocity + joint limit + orientation",
        "#3d91c9",
    )
    draw_box(
        ax,
        (0.53, 0.08),
        (0.40, 0.23),
        "Robustness Sources",
        "observation noise + action noise\none-step action delay\nunreachable target segments\ncombined mismatch evaluation",
        "#d67a22",
    )

    out = out_dir / "system_design.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out

def make_overview_figure(run_dir: Path, out_dir: Path, eval_dir_name: str) -> Path:
    eval_dir = final_eval_dir(run_dir, eval_dir_name)
    df = pd.read_csv(eval_dir / "tracking_log.csv")
    summary = pd.read_csv(eval_dir / "summary.csv")
    sample = first_clean_episode(df)

    sns.set_theme(style="whitegrid", context="talk")
    fig = plt.figure(figsize=(18, 12))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.95])

    ax3d = fig.add_subplot(grid[0, 0], projection="3d")
    ax3d.plot(sample["desired_x"], sample["desired_y"], sample["desired_z"], color=DESIRED, lw=3.0, label="desired")
    ax3d.plot(sample["ee_x"], sample["ee_y"], sample["ee_z"], color=ACTUAL, lw=3.0, label="actual")
    ax3d.set_title("3D Trajectory Tracking")
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.legend(loc="upper left")
    ax3d.view_init(elev=24, azim=-58)

    ax_err = fig.add_subplot(grid[0, 1])
    ax_err.plot(sample["time"], sample["tracking_error"] * 100.0, color="#d94f4f", lw=2.4)
    ax_err.axhline(float(summary.loc[summary["profile"] == "clean", "rmse_error"].iloc[0]) * 100.0, color="#333333", ls="--", lw=1.6, label="clean RMSE")
    ax_err.set_title("Tracking Error Over Time")
    ax_err.set_xlabel("time [s]")
    ax_err.set_ylabel("error [cm]")
    ax_err.legend()

    ax_act = fig.add_subplot(grid[1, 0])
    ax_act.plot(sample["time"], sample["action_delta"], color="#7c4dff", lw=2.4, label="action delta")
    ax_act.plot(sample["time"], sample["joint_velocity_norm"], color="#009688", lw=2.0, alpha=0.85, label="joint velocity")
    ax_act.set_title("Smoothness and Stability Signals")
    ax_act.set_xlabel("time [s]")
    ax_act.set_ylabel("magnitude")
    ax_act.legend()

    ax_rob = fig.add_subplot(grid[1, 1])
    order = ["clean", "obs_noise", "action_delay", "action_noise", "unreachable", "combined"]
    plot_summary = summary.copy()
    plot_summary["rmse_cm"] = plot_summary["rmse_error"] * 100.0
    sns.barplot(data=plot_summary, x="profile", y="rmse_cm", order=order, ax=ax_rob, color="#3d91c9")
    ax_rob.set_title("Robustness RMSE by Evaluation Profile")
    ax_rob.set_xlabel("")
    ax_rob.set_ylabel("RMSE [cm]")
    ax_rob.tick_params(axis="x", rotation=18)

    clean = summary[summary["profile"] == "clean"].iloc[0]
    combined = summary[summary["profile"] == "combined"].iloc[0]
    fig.suptitle(
        "Dynamic RL Trajectory Tracking Summary | "
        f"clean tracking RMSE {clean['rmse_error'] * 100:.2f} cm, "
        f"action delta {clean['mean_action_delta']:.4f}, "
        f"combined robustness RMSE {combined['rmse_error'] * 100:.2f} cm",
        fontsize=20,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = out_dir / "overview.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def draw_centered(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill=INK) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    draw.text((xy[0] - width // 2, xy[1]), text, font=font, fill=fill)


def make_card(size: tuple[int, int], title: str, lines: list[str], metrics: list[str] | None = None) -> Image.Image:
    image = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(image)
    title_font = load_font(48, bold=True)
    body_font = load_font(27)
    metric_font = load_font(31, bold=True)

    w, h = size
    draw.rectangle((0, 0, w, h), fill=BG)
    draw.rectangle((0, h - 10, w, h), fill=ACTUAL)
    draw.ellipse((w - 250, 70, w + 120, 440), fill="#1a2d36")
    draw.ellipse((-120, h - 330, 230, h + 40), fill="#172530")

    draw_centered(draw, (w // 2, 145), title, title_font)
    y = 235
    for line in lines:
        draw_centered(draw, (w // 2, y), line, body_font, fill=MUTED)
        y += 43

    if metrics:
        box_w = min(1030, w - 160)
        box_x = (w - box_w) // 2
        box_y = y + 35
        draw.rounded_rectangle((box_x, box_y, box_x + box_w, box_y + 150), radius=12, fill="#172129", outline="#324450", width=2)
        my = box_y + 26
        for metric in metrics:
            draw_centered(draw, (w // 2, my), metric, metric_font, fill=INK)
            my += 45

    return image


def overlay_frame(frame: np.ndarray, label: str, subtitle: str, idx: int, total: int) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    image = image.resize((1280, 720), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    title_font = load_font(30, bold=True)
    small_font = load_font(20)

    draw.rectangle((0, 0, 1280, 66), fill=(13, 19, 24))
    draw.text((28, 16), label, font=title_font, fill=INK)
    draw.text((410, 22), subtitle, font=small_font, fill=MUTED)

    bar_w = int(1280 * (idx + 1) / max(total, 1))
    draw.rectangle((0, 66, bar_w, 70), fill=ACTUAL)
    legend_x = 1056
    draw.ellipse((legend_x, 10, legend_x + 20, 30), fill=DESIRED)
    draw.text((legend_x + 27, 9), "desired", font=small_font, fill=MUTED)
    draw.ellipse((legend_x, 34, legend_x + 20, 54), fill=GENERATED)
    draw.text((legend_x + 27, 33), "generated", font=small_font, fill=MUTED)
    draw.ellipse((legend_x + 132, 22, legend_x + 152, 42), fill=ACTUAL)
    draw.text((legend_x + 159, 21), "actual", font=small_font, fill=MUTED)
    return np.asarray(image)


def write_repeated_card(writer, card: Image.Image, n_frames: int) -> None:
    frame = np.asarray(card)
    for _ in range(n_frames):
        writer.append_data(frame)


def write_video_segment(writer, source: Path, label: str, subtitle: str, seconds: float, fps: int) -> None:
    reader = imageio.get_reader(source)
    meta = reader.get_meta_data()
    source_fps = float(meta.get("fps") or 50.0)
    total_source_frames = reader.count_frames()
    total_out_frames = int(seconds * fps)
    start_time = 0.8
    for i in range(total_out_frames):
        source_t = start_time + i / fps
        source_idx = min(int(source_t * source_fps), total_source_frames - 1)
        frame = reader.get_data(source_idx)
        writer.append_data(overlay_frame(frame, label, subtitle, i, total_out_frames))
    reader.close()


def make_demo_video(run_dir: Path, out_dir: Path, eval_dir_name: str, fps: int, clip_seconds: float, title_seconds: float, metrics_seconds: float) -> Path:
    summary = pd.read_csv(final_eval_dir(run_dir, eval_dir_name) / "summary.csv")
    clean = summary[summary["profile"] == "clean"].iloc[0]
    combined = summary[summary["profile"] == "combined"].iloc[0]

    metrics = [
        f"Dynamic tracking RMSE (clean) {clean['rmse_error'] * 100:.2f} cm  |  Mean error {clean['mean_error'] * 100:.2f} cm",
        f"Smoothness: action delta {clean['mean_action_delta']:.4f}  |  joint velocity {clean['mean_joint_velocity']:.4f}",
        f"Robustness: combined RMSE {combined['rmse_error'] * 100:.2f} cm  |  success {combined['success_rate'] * 100:.2f}%",
    ]

    out = out_dir / "demo.mp4"
    with imageio.get_writer(out, fps=fps, codec="libx264", quality=8, macro_block_size=16) as writer:
        title = make_card(
            (1280, 720),
            "Complete RL System for Dynamic 3D Tracking",
            [
                "MuJoCo + official Franka Panda + Gymnasium-style environment",
                "Stable-Baselines3 PPO policy for continuous moving-target tracking",
                "Aligned-start trajectory generation removes artificial initial transients",
            ],
            metrics,
        )
        write_repeated_card(writer, title, int(title_seconds * fps))

        segments = [
            ("Circle trajectory", "smooth closed 3D reference with generated target and actual trace", run_dir / "videos" / "tracking_circle_20s.mp4"),
            ("Figure-8 trajectory", "crossing dynamic path: tracking accuracy plus smoothness", run_dir / "videos" / "tracking_figure8_20s.mp4"),
            ("Moving target", "non-closed target motion with robustness-oriented evaluation", run_dir / "videos" / "tracking_moving_20s.mp4"),
        ]
        for label, subtitle, path in segments:
            write_video_segment(writer, path, label, subtitle, clip_seconds, fps)

        end = make_card(
            (1280, 720),
            "Competition Evidence",
            [
                "Dynamic 3D tracking, not fixed-point reaching",
                "Multi-term reward balances precision, smoothness, joint safety, and orientation",
                "Robustness tested with clean, noisy, delayed, unreachable, and combined profiles",
            ],
            metrics,
        )
        write_repeated_card(writer, end, int(metrics_seconds * fps))

    return out
def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    system_design = make_system_design_figure(out_dir)
    overview = make_overview_figure(run_dir, out_dir, args.eval_dir_name)
    video = make_demo_video(run_dir, out_dir, args.eval_dir_name, fps=args.fps, clip_seconds=args.clip_seconds, title_seconds=args.title_seconds, metrics_seconds=args.metrics_seconds)

    for path in [system_design, overview, video]:
        print(path)


if __name__ == "__main__":
    main()

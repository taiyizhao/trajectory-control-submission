from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trajec_mujoco.utils.plotting import make_evaluation_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots from evaluation CSV files.")
    parser.add_argument("--eval-dir", type=Path, required=True)
    args = parser.parse_args()
    plots = make_evaluation_plots(args.eval_dir)
    print("Created plots:")
    for path in plots:
        print(f"  {path}")


if __name__ == "__main__":
    main()

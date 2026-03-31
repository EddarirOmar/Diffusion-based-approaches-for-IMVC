"""IMVCDC pipeline runner.

Examples:
  python run.py --stage 1 --data_name NoisyMNIST
  python run.py --stage 2 --data_name NoisyMNIST
  python run.py --stage 3 --data_name NoisyMNIST
  python run.py --stage all --data_name NoisyMNIST
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run_stage(script_name: str, data_name: str, output_dir: str | None = None) -> None:
    cmd = [sys.executable, script_name, "--data_name", data_name]
    if output_dir:
        cmd += ["--output_dir", output_dir]

    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Stage script failed: {script_name} (exit={result.returncode})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IMVCDC pipeline stages")
    parser.add_argument("--stage", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--data_name", required=True, help="Dataset name (e.g., NoisyMNIST, CUB, Synthetic3d)")
    parser.add_argument("--output_root", default=None, help="Optional root directory for stage outputs")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent

    if args.stage in ("1", "all"):
        out1 = str(root / "outputs" / f"stage1_rec_{args.data_name.lower()}") if args.output_root is None else str(Path(args.output_root) / "stage1")
        _run_stage(str(root / "run_stage1.py"), args.data_name, out1)

    if args.stage in ("2", "all"):
        out2 = str(root / "outputs" / f"stage2_dm_{args.data_name.lower()}") if args.output_root is None else str(Path(args.output_root) / "stage2")
        _run_stage(str(root / "run_stage2.py"), args.data_name, out2)

    if args.stage in ("3", "all"):
        out3 = str(root / "outputs" / f"stage3_clu_{args.data_name.lower()}") if args.output_root is None else str(Path(args.output_root) / "stage3")
        _run_stage(str(root / "run_stage3.py"), args.data_name, out3)

    print("[DONE] Requested stage(s) completed.")


if __name__ == "__main__":
    main()

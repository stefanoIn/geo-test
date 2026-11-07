"""Run the remaining TerraMind trials sequentially in separate processes."""

import subprocess
import sys
from pathlib import Path

CONFIGS = [
    "configs/terramind/tuning/s1_lr0.005_e20.yaml",
    "configs/terramind/tuning/s2_lr0.0001_e20.yaml",
    "configs/terramind/tuning/s2_lr0.0003_e20.yaml",
    "configs/terramind/tuning/s2_lr0.0005_e20.yaml",
    "configs/terramind/tuning/s2_lr0.001_e20.yaml",
    "configs/terramind/tuning/s2_lr0.005_e20.yaml",
    "configs/terramind/tuning/s1s2_lr0.0001_e20.yaml",
    "configs/terramind/tuning/s1s2_lr0.0003_e20.yaml",
    "configs/terramind/tuning/s1s2_lr0.0005_e20.yaml",
    "configs/terramind/tuning/s1s2_lr0.001_e20.yaml",
    "configs/terramind/tuning/s1s2_lr0.005_e20.yaml",
]

if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[2]
    for config in CONFIGS:
        if not (project_dir / config).is_file():
            raise FileNotFoundError(config)

    for index, config in enumerate(CONFIGS, start=1):
        print(f"\n[{index}/{len(CONFIGS)}] Starting {config}", flush=True)
        # Wait for each trial to exit, releasing its model and native-library state.
        # Use this environment's Python and enable crash tracebacks in every child.
        result = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-u", "run.py", "--path", config],
            cwd=project_dir,
        )
        if result.returncode != 0:
            raise SystemExit(
                f"Trial failed: {config}; exit code {result.returncode} "
                f"(0x{result.returncode & 0xFFFFFFFF:08X}). Remaining trials were not started."
            )
        print(f"Completed {config}", flush=True)

"""Train and evaluate the six tuned FCN experiments, one process at a time."""

import subprocess
import sys
from pathlib import Path

# Edit a YAML to change its settings. Each invocation creates a fresh run directory.
CONFIGS = [
    "configs/croma/fcn_tuned/s1.yaml",
    "configs/croma/fcn_tuned/s2.yaml",
    "configs/croma/fcn_tuned/s1s2.yaml",
    "configs/terramind/fcn_tuned/s1.yaml",
    "configs/terramind/fcn_tuned/s2.yaml",
    "configs/terramind/fcn_tuned/s1s2.yaml"
]

if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[1]
    for config in CONFIGS:
        if not (project_dir / config).is_file():
            raise FileNotFoundError(config)

    for index, config in enumerate(CONFIGS, start=1):
        print(f"\n[{index}/{len(CONFIGS)}] Starting {config}", flush=True)
        # Wait for training and test/Bolivia evaluation before launching the next model.
        # A fresh process avoids reinitializing TerraMind repeatedly in the same process.
        result = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-u", "run.py", "--path", config],
            cwd=project_dir,
        )
        if result.returncode != 0:
            raise SystemExit(
                f"Failed: {config}; exit code {result.returncode} "
                f"(0x{result.returncode & 0xFFFFFFFF:08X}). Remaining experiments were not started."
            )
        print(f"Completed {config}", flush=True)

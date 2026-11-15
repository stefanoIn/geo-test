import subprocess
import sys
from pathlib import Path

# YAML mode selects training or evaluation; these entries evaluate existing best checkpoints.
CONFIGS = [
    "configs/evaluation/croma_s1s2_linear_lr0.003.yaml",
    "configs/evaluation/terramind_s1s2_linear_lr0.005.yaml",
    "configs/evaluation/unet_s1s2_lr0.0003.yaml",
    "configs/evaluation/resnet18_unet_s1s2_lr0.001.yaml",
]

if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    for config in CONFIGS:
        if not (project_dir / config).is_file():
            raise FileNotFoundError(config)

    for index, config in enumerate(CONFIGS, start=1):
        print(f"\n[{index}/{len(CONFIGS)}] Starting {config}", flush=True)
        result = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-u", "run.py", "--path", config],
            cwd=project_dir,
        )
        if result.returncode != 0:
            raise SystemExit(
                f"Failed: {config}; exit code {result.returncode} "
                f"(0x{result.returncode & 0xFFFFFFFF:08X}). Remaining trials were not started."
            )
        print(f"Completed {config}", flush=True)

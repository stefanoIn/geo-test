import subprocess
import sys
from pathlib import Path

# Each YAML defines the full experiment. Re-running a trial creates a new run folder.
CONFIGS = [
    "configs/resnet_unet/resnet18_unet_s1_comparison_50e.yaml",
    "configs/resnet_unet/resnet18_unet_s2_comparison_50e.yaml",
    "configs/resnet_unet/resnet18_unet_s1s2_comparison_50e.yaml",
    "configs/resnet_unet/resnet34_unet_s1_comparison_50e.yaml",
    "configs/resnet_unet/resnet34_unet_s2_comparison_50e.yaml",
    "configs/resnet_unet/resnet34_unet_s1s2_comparison_50e.yaml",
    "configs/resnet_unet/resnet50_unet_s1_comparison_50e.yaml",
    "configs/resnet_unet/resnet50_unet_s2_comparison_50e.yaml",
    "configs/resnet_unet/resnet50_unet_s1s2_comparison_50e.yaml",
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

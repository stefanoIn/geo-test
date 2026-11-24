import subprocess
import sys
from pathlib import Path

# Train and validate the selected S1+S2 configurations with seeds 12 and 225.
CONFIGS = [
    "configs/unet/unet_s1s2_100_noshift/unet_s1s2_100_noshift_lr0.0003_seed12.yaml",
    "configs/unet/unet_s1s2_100_noshift/unet_s1s2_100_noshift_lr0.0003_seed225.yaml",
    "configs/resnet_unet/resnet18_unet_s1s2_100_noshift/resnet18_unet_s1s2_100_noshift_lr0.001_seed12.yaml",
    "configs/resnet_unet/resnet18_unet_s1s2_100_noshift/resnet18_unet_s1s2_100_noshift_lr0.001_seed225.yaml",
    "configs/terramind/linear/terramind_s1s2_linear_100_noshift/terramind_s1s2_linear_100_noshift_lr0.005_seed12.yaml",
    "configs/terramind/linear/terramind_s1s2_linear_100_noshift/terramind_s1s2_linear_100_noshift_lr0.005_seed225.yaml",
    "configs/croma/linear/croma_s1s2_linear_100_noshift/croma_s1s2_linear_100_noshift_lr0.003_seed12.yaml",
    "configs/croma/linear/croma_s1s2_linear_100_noshift/croma_s1s2_linear_100_noshift_lr0.003_seed225.yaml",
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

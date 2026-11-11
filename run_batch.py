import subprocess
import sys
from pathlib import Path

# Each YAML defines the full experiment. Re-running a trial creates a new run folder.
CONFIGS = [
      "configs/unet/01_unet_s1s2_full_pipeline_2e.yaml",
       "configs/unet/02_unet_s1_only_3e.yaml",
       "configs/unet/03_unet_s2_only_3e.yaml",
       "configs/unet/04_unet_s1s2_no_b10_4e.yaml",
       "configs/unet/05_unet_s1s2_two_sigma_8bit_4e.yaml",
       "configs/resnet_unet/01_resnet50_s1s2_full_pipeline_2e.yaml",
       "configs/resnet_unet/02_resnet18_s1s2_3e.yaml",
       "configs/resnet_unet/03_resnet34_s1s2_3e.yaml",
       "configs/resnet_unet/04_resnet50_s1s2_no_b10_4e.yaml",
       "configs/resnet_unet/05_resnet18_s1s2_sgd_4e.yaml",
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

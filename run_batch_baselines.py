"""Run all 10 GeoTest baseline integration YAMLs in a controlled sequence."""
import subprocess
import sys
from pathlib import Path

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
    root = Path(__file__).resolve().parent
    for config in CONFIGS:
        if not (root / config).is_file():
            raise FileNotFoundError(config)
    for idx, config in enumerate(CONFIGS, 1):
        print(f"\n[{idx}/{len(CONFIGS)}] {config}", flush=True)
        result = subprocess.run(
            [sys.executable, "-u", "run.py", "--path", config], cwd=root
        )
        if result.returncode != 0:
            raise SystemExit(f"FAILED: {config} (exit code {result.returncode})")
    print("\nAll baseline integration trials completed.", flush=True)

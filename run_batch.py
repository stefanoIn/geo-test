import subprocess
import sys
from pathlib import Path

# Each YAML defines the full experiment. Re-running a trial creates a new run folder.
CONFIGS = [
    "configs/terramind/linear_segmentation/terramind_s1_linear_segmentation_50e_all_hand_labels.yaml",
    "configs/croma/linear_segmentation/croma_s1_linear_segmentation_50e_all_hand_labels.yaml",
    "configs/terramind/linear_segmentation/terramind_s2_linear_segmentation_50e_all_hand_labels.yaml",  
    "configs/croma/linear_segmentation/croma_s2_linear_segmentation_50e_all_hand_labels.yaml",
        
]

if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    for config in CONFIGS:
        if not (project_dir / config).is_file():
            raise FileNotFoundError(config)

    for index, config in enumerate(CONFIGS, start=1):
        print(f"\n[{index}/{len(CONFIGS)}] Starting {config}", flush=True)
        # Wait for each process to exit before loading the next encoder.
        # run.py saves the best-mIoU checkpoint, summary and metric history per trial.
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

"""Regenerate run figures from saved metrics only; no model or imagery access."""
import argparse
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.plotting import save_training_plots, save_test_plots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--no-titles", action="store_true")
    parser.add_argument("--legend-loc", default="best")
    parser.add_argument("--legend-columns", type=int, default=1)
    args = parser.parse_args()
    root = args.run_dir
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    selected = summary.get("best_checkpoint")
    history = root / "metrics/history.json"
    if history.is_file():
        save_training_plots(root, config, json.loads(history.read_text(encoding="utf-8")),
                            selected_checkpoint=selected, show_titles=not args.no_titles,
                            legend_loc=args.legend_loc, legend_ncol=args.legend_columns)
    for path in sorted((root / "metrics").glob("*.json")):
        if path.name == "history.json" or path.stem.startswith("validation_"):
            continue
        results = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(results, dict) or not {"miou", "iou_per_class"} <= results.keys():
            continue
        names = results.get("class_names") or config.get("evaluation", {}).get("class_names")
        if names is None:
            names = [f"Class {i}" for i in range(len(results["iou_per_class"]))]
        if len(names) != len(results["iou_per_class"]):
            raise ValueError(f"{path}: class_names length disagrees with iou_per_class")
        epoch = results.get("metadata", {}).get("selected_epoch",
                    summary.get("selected_epoch", (selected or {}).get("epoch")))
        save_test_plots(root, config, path.stem, results, epoch,
                        class_names=list(names), show_titles=not args.no_titles)
    print(f"Figure exports: {root / 'plots'}")


if __name__ == "__main__":
    main()

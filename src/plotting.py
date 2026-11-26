"""Scientific figures; experiment summaries and metric definitions remain in utils."""
from contextlib import contextmanager
from pathlib import Path
import os
import warnings


@contextmanager
def scientific_style(path=None):
    """Scoped SciencePlots style with an optional external thesis stylesheet."""
    import matplotlib.style as style
    from cycler import cycler
    styles = ['default']
    try:
        import scienceplots  # Registers science and no-latex without activating them.
        styles.extend(['science', 'no-latex'])
    except ImportError:
        warnings.warn('SciencePlots unavailable; using the serif Matplotlib fallback.', stacklevel=2)
    styles.append({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
                   'mathtext.fontset': 'cm', 'text.usetex': False, 'font.size': 9,
                   'legend.fontsize': 7.5, 'legend.frameon': False,
                   'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                   'savefig.dpi': 300, 'pdf.fonttype': 42,
                   'figure.figsize': [7.2, 3.2], 'axes.formatter.use_mathtext': True,
                   'axes.formatter.limits': [-3, 4],
                   'axes.prop_cycle': (cycler(linestyle=['-', '--', ':', '-.']) *
                       cycler(marker=['', 'o', 's']) * cycler(color=[
                       '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#000000']))})
    sheet = Path(path or os.environ.get('THESIS_MPLSTYLE') or
                 Path(__file__).resolve().parents[2] / 'figures/style/thesis.mplstyle')
    if sheet.is_file():
        import matplotlib as mpl
        try:
            styles.append(mpl.rc_params_from_file(sheet, fail_on_error=True, use_default_template=False))
        except (OSError, ValueError) as exc:
            warnings.warn(f'Cannot read thesis style {sheet}: {exc}; using fallback.', stacklevel=2)
    else:
        warnings.warn(f'Thesis stylesheet not found at {sheet}; using fallback.', stacklevel=2)
    with style.context(styles):
        yield


def save_figure(figure, path, **kwargs):
    """Export vector PDF and 300 DPI PNG from the same figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('.pdf', '.png'):
        figure.savefig(path.with_suffix(extension), dpi=300, **kwargs)

def optional_plot(function):
    """Skip optional figure export when Matplotlib is not installed."""
    from functools import wraps
    from importlib.util import find_spec

    @wraps(function)
    def wrapped(*args, **kwargs):
        if find_spec("matplotlib") is None:
            warnings.warn("Matplotlib unavailable; skipping figure export.", stacklevel=2)
            return None
        return function(*args, **kwargs)
    return wrapped


# Save curves inside this run; use the same checkpoint selection as summary.json.
@optional_plot
@scientific_style()
def save_training_plots(run_dir, config, history, *, selected_checkpoint=..., show_titles=True,
                        figsize=(7.2, 3.2), legend_loc='best', legend_ncol=1):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if not history:
        return
    validated = [row for row in history if row.get("validation") is not None]
    if selected_checkpoint is ...:
        from utils import build_run_summary
        selected = build_run_summary(config, history)["best_checkpoint"]
    else:
        selected = selected_checkpoint
    plot_dir = Path(run_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    figure = Figure(figsize=figsize, layout="constrained")
    FigureCanvasAgg(figure)
    loss_ax, iou_ax = figure.subplots(1, 2)
    loss_ax.plot([row["epoch"] for row in history],
                 [row["train_loss"] for row in history], label="Training loss")
    loss_rows = [row for row in validated if row["validation"].get("loss") is not None]
    loss_ax.plot([row["epoch"] for row in loss_rows],
                 [row["validation"]["loss"] for row in loss_rows],
                 marker="o", label="Validation loss")
    iou_rows = [row for row in validated if row["validation"].get("miou") is not None]
    iou_ax.plot([row["epoch"] for row in iou_rows],
                [100 * row["validation"]["miou"] for row in iou_rows],
                marker="o", label="Validation mIoU")
    loss_ax.set(title="Training and validation loss", ylabel="Loss")
    iou_ax.set(title="Validation mIoU", ylabel="mIoU (%)", ylim=(0, 100))

    for axis in (loss_ax, iou_ax):
        if selected is not None:
            axis.axvline(selected["epoch"], color="tab:red", linestyle="--",
                         label=f"Selected checkpoint: epoch {selected['epoch']}")
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend(loc=legend_loc, ncol=legend_ncol)
    if show_titles:
        figure.suptitle(config["experiment"]["name"])
    else:
        for axis in (loss_ax, iou_ax):
            axis.set_title("")
    for extension in ("png", "pdf"):
        figure.savefig(plot_dir / f"training_curves.{extension}", dpi=300)
    figure.clear()


# Plot held-out scores without recomputing metrics or selecting on test data.
@optional_plot
@scientific_style()
def save_test_plots(run_dir, config, split, results, selected_epoch, *, class_names=None, show_titles=True, figsize=None):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    categories = results.get("aw_pw_fw", {}).get("summary", {})
    figure = Figure(figsize=(15 if categories else 5, 4.5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3 if categories else 1, squeeze=False)[0]

    def bars(axis, labels, values, title):
        heights = [100 * value if value is not None else 0 for value in values]
        rectangles = axis.bar(labels, heights)
        axis.bar_label(rectangles, labels=[f"{100 * value:.2f}%" if value is not None
                                          else "Undefined" for value in values], padding=3)
        axis.set(title=title if show_titles else "", ylabel="IoU (%)", ylim=(0, 110))
        axis.set_yticks(range(0, 101, 20))
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.25)

    if class_names is None:
        from utils import metric_class_names
        names = metric_class_names(config, results)
    else:
        names = class_names
    bars(axes[0], ["mIoU"] + names,
         [results["miou"]] + list(results["iou_per_class"]), "mIoU and per-class IoU")
    axes[0].tick_params(axis="x", labelrotation=45)
    figure.set_size_inches(*(figsize or ((10.8 if categories else 3.6), 3.6)))
    if categories:
        labels = [name for name in ("AW", "PW", "FW") if name in categories]
        bars(axes[1], labels, [categories[name]["mean_chip_iou"] for name in labels],
             "Mean per-chip Water IoU")
        bars(axes[2], labels, [categories[name]["total_iou"] for name in labels],
             "Pooled Water IoU by category")
    if show_titles:
        figure.suptitle(f"{config['experiment']['name']} | {split} | Selected epoch {selected_epoch}")
    plot_dir = Path(run_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(plot_dir / f"{split}_metrics.{extension}", dpi=300)
    figure.clear()



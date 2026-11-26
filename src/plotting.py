"""Scientific figures; experiment summaries and metric definitions remain in utils."""
from contextlib import contextmanager
from pathlib import Path
import os
import warnings


@contextmanager
def scientific_style(path=None):
    """Scoped native SciencePlots styles with an optional thesis stylesheet."""
    import matplotlib as mpl
    import matplotlib.style as style
    from cycler import cycler
    styles = ['default']
    try:
        import scienceplots  # Registers styles without activating them.
        styles.extend(['science', 'no-latex', 'vibrant'])
    except ImportError:
        warnings.warn('SciencePlots unavailable; using the serif Matplotlib fallback.', stacklevel=2)
        styles.append({'font.family': 'serif', 'mathtext.fontset': 'cm',
                       'text.usetex': False,
                       'axes.prop_cycle': cycler(color=[
                           '#0072B2', '#D55E00', '#009E73', '#CC79A7',
                           '#E69F00', '#56B4E9', '#000000'])})
    styles.append({'legend.frameon': False, 'savefig.dpi': 300,
                   'pdf.fonttype': 42, 'figure.figsize': [7.2, 3.2]})
    sheet = Path(path or os.environ.get('THESIS_MPLSTYLE') or
                 Path(__file__).resolve().parents[2] / 'figures/style/thesis.mplstyle')
    if sheet.is_file():
        try:
            styles.append(mpl.rc_params_from_file(sheet, fail_on_error=True, use_default_template=False))
        except (OSError, ValueError) as exc:
            warnings.warn(f'Cannot read thesis style {sheet}: {exc}; using fallback.', stacklevel=2)
    else:
        warnings.warn(f'Thesis stylesheet not found at {sheet}; using fallback.', stacklevel=2)
    with style.context(styles):
        # Preserve the active native palette; distinguish repeated colours by line style.
        colours = mpl.rcParams['axes.prop_cycle'].by_key()['color']
        mpl.rcParams['axes.prop_cycle'] = cycler(linestyle=['-', '--', ':', '-.']) * cycler(color=colours)
        yield


def compact_legend_layout(figure, *, w_pad=3.0, top=1.0):
    """Place existing legends just below axis labels using their rendered bounds."""
    from matplotlib.transforms import Bbox

    legends = [(legend, figure.axes) for legend in figure.legends]
    legends += [(axis.get_legend(), [axis]) for axis in figure.axes
                if axis.get_legend() is not None]
    for legend, _ in legends:
        legend.set_in_layout(False)
    figure.tight_layout(w_pad=w_pad, rect=(0, 0, 1, top))
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    gap = 4 * figure.dpi / 72
    for legend, axes in legends:
        bounds = [axis.get_tightbbox(renderer) for axis in axes]
        bottom = min(bound.y0 for bound in bounds if bound is not None)
        centre = (min(axis.bbox.x0 for axis in axes) +
                  max(axis.bbox.x1 for axis in axes)) / 2
        box = legend.get_window_extent(renderer)
        anchor = legend.get_bbox_to_anchor()
        shifted = Bbox.from_bounds(
            anchor.x0 + centre - (box.x0 + box.x1) / 2,
            anchor.y0 + bottom - gap - box.y1, anchor.width, anchor.height)
        legend.set_bbox_to_anchor(shifted.transformed(figure.transFigure.inverted()),
                                  transform=figure.transFigure)
    for legend, _ in legends:
        legend.set_in_layout(True)


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
                        figsize=(9.0, 3.2), legend_loc='lower center', legend_ncol=3,
                        title=None):
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

    figure = Figure(figsize=figsize)
    FigureCanvasAgg(figure)
    loss_ax, iou_ax = figure.subplots(1, 2)
    training_line, = loss_ax.plot(
        [row["epoch"] for row in history], [row["train_loss"] for row in history],
        color="#EE7733", linestyle="-", marker="", label="Training")
    loss_rows = [row for row in validated if row["validation"].get("loss") is not None]
    validation_line, = loss_ax.plot(
        [row["epoch"] for row in loss_rows],
        [row["validation"]["loss"] for row in loss_rows],
        color="#0077BB", linestyle="-", marker="", label="Validation")
    iou_rows = [row for row in validated if row["validation"].get("miou") is not None]
    iou_ax.plot([row["epoch"] for row in iou_rows],
                [100 * row["validation"]["miou"] for row in iou_rows],
                color=validation_line.get_color(), linestyle="-", marker="")
    loss_ax.set(ylabel="Loss")
    iou_ax.set(ylabel="mIoU (%)", ylim=(0, 100))
    if show_titles:
        loss_ax.set_title("(a) Training and validation loss")
        iou_ax.set_title("(b) Validation performance")
        if title:
            figure.suptitle(title)

    handles = [training_line, validation_line]
    for axis in (loss_ax, iou_ax):
        if selected is not None:
            checkpoint_line = axis.axvline(
                selected["epoch"], color="0.5", linestyle=":", linewidth=0.9,
                label=f"Selected checkpoint: epoch {selected['epoch']}")
            if axis is loss_ax:
                handles.append(checkpoint_line)
        axis.set_xlabel("Epoch")
        axis.minorticks_on()
        axis.grid(True, which="major", axis="both")
        axis.grid(False, which="minor")
    if selected is not None:
        for axis, rows, key, scale in ((loss_ax, loss_rows, "loss", 1),
                                      (iou_ax, iou_rows, "miou", 100)):
            chosen = next((row for row in rows if row["epoch"] == selected["epoch"]), None)
            if chosen is not None:
                axis.plot(chosen["epoch"], scale * chosen["validation"][key],
                          marker="o", markersize=4, markerfacecolor="none",
                          markeredgecolor=validation_line.get_color(),
                          linestyle="none", zorder=4)
    columns = max(1, min(legend_ncol, len(handles)))
    # Figure legends do not support loc='best'; retain compatibility with old callers.
    figure.legend(handles=handles, loc="lower center" if legend_loc == "best" else legend_loc,
                  bbox_to_anchor=(0.5, 0), ncol=columns)
    compact_legend_layout(figure, top=0.94 if title and show_titles else 1)
    save_figure(figure, plot_dir / "training_curves.png", bbox_inches="tight")
    figure.clear()


# Plot held-out scores without recomputing metrics or selecting on test data.
@optional_plot
@scientific_style()
def save_test_plots(run_dir, config, split, results, selected_epoch, *, class_names=None,
                    show_titles=True, figsize=None, legend_loc="lower center", legend_ncol=2):
    from textwrap import fill
    from matplotlib import rcParams
    from matplotlib.patches import Patch
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    categories = results.get("aw_pw_fw", {}).get("summary", {})
    if class_names is None:
        from utils import metric_class_names
        names = metric_class_names(config, results)
    else:
        names = list(class_names)
    class_values = list(results["iou_per_class"])
    if len(names) != len(class_values):
        raise ValueError("class_names must contain one name per evaluated class")

    # Preserve the saved category order for both aggregation methods.
    labels = list(categories)
    methods = [(key, label) for key, label in
               (("mean_chip_iou", "Mean per-chip IoU"), ("total_iou", "Pooled IoU"))
               if any(key in categories[name] for name in labels)]
    has_categories = bool(labels and methods)
    palette = list(dict.fromkeys(rcParams["axes.prop_cycle"].by_key()["color"]))
    method_colours = {"mean_chip_iou": palette[0], "total_iou": palette[1 % len(palette)]}
    class_width = max(3.8, .95 * len(names))
    category_width = max(5.2, 1.4 * len(labels))
    figure = Figure(figsize=figsize or
                    (class_width + (category_width if has_categories else 0), 3.3))
    FigureCanvasAgg(figure)
    axes = figure.subplots(
        1, 2 if has_categories else 1, sharey=True, squeeze=False,
        gridspec_kw={"width_ratios": [class_width, category_width] if has_categories else [1]}
    )[0]

    def percentage(value):
        return f"{100 * value:.2f}%" if value is not None else "Undefined"

    def bars(axis, positions, values, **kwargs):
        # Retain the existing explicit Undefined annotation for missing metrics.
        rectangles = axis.bar(positions, [100 * value if value is not None else 0
                                         for value in values], **kwargs)
        axis.bar_label(rectangles, labels=[percentage(value) for value in values],
                       padding=3, fontsize=8)
        return rectangles

    class_positions = [.9 * index for index in range(len(names))]
    class_limits = (-.45, .9 * max(0, len(names) - 1) + .45)
    class_bar_width = min(.48, .30 + .04 * len(names))
    if has_categories:
        width = min(.72, .56 + .02 * len(labels)) / len(methods)
        category_limits = (-.65, len(labels) - 1 + .65)
        # Match each class bar to the full category group in displayed width.
        class_bar_width = (width * len(methods) * category_width / class_width *
                              (class_limits[1] - class_limits[0]) /
                              (category_limits[1] - category_limits[0]))
    bars(axes[0], class_positions, class_values, width=class_bar_width,
         color=method_colours["total_iou"])
    axes[0].set_xlim(*class_limits)
    axes[0].set_xticks(class_positions,
                      [fill(str(name).replace("_", " ").replace("non ", "non-").strip().capitalize(),
                            width=16) for name in names])
    miou_text = f"mIoU: {percentage(results['miou'])}"
    if show_titles:
        axes[0].set_title("(a) Class-wise pooled IoU", pad=24)
    axes[0].annotate(miou_text, xy=(.5, 1), xycoords="axes fraction",
                     xytext=(0, 5), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("IoU (%)")

    if has_categories:
        axes[1].set_xlim(*category_limits)
        for index, (key, label) in enumerate(methods):
            positions = [position + (index - (len(methods) - 1) / 2) * width
                         for position in range(len(labels))]
            bars(axes[1], positions, [categories[name].get(key) for name in labels],
                 width=width, color=method_colours[key], label=label)
        axes[1].set_xticks(range(len(labels)),
                          [fill(str(name).replace("_", " "), width=16) for name in labels])
        if show_titles:
            axes[1].set_title("(b) Water IoU by category", pad=24)
    # Include pooled IoU even if only panel (a) contains that method.
    shown_methods = [(key, label) for key, label in
                     (("mean_chip_iou", "Mean per-chip IoU"), ("total_iou", "Pooled IoU"))
                     if key == "total_iou" or any(method[0] == key for method in methods)]
    handles = [Patch(facecolor=method_colours[key], label=label)
               for key, label in shown_methods]
    figure.legend(handles=handles,
                  loc="lower center" if legend_loc == "best" else legend_loc,
                  bbox_to_anchor=(.5, 0), ncol=max(1, min(legend_ncol, len(handles))))

    for axis in axes:
        axis.set_ylim(0, 110)
        axis.set_yticks(range(0, 101, 20))
        axis.set_axisbelow(True)
        axis.minorticks_off()
        axis.tick_params(axis="x", which="both", top=False, labeltop=False)
        axis.grid(False, axis="x", which="both")
        axis.grid(True, axis="y", which="major")
    compact_legend_layout(figure)
    plot_dir = Path(run_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_figure(figure, plot_dir / f"{split}_metrics.png", bbox_inches="tight")
    figure.clear()

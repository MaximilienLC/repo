"""Plotting and visualization functions for Experiment 3."""

import json
from pathlib import Path

import filelock
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatter, LogLocator

from config import RESULTS_DIR, SCRIPT_DIR
from utils import format_method_name

# Global plot figure for reuse
_PLOT_FIG = None
_PLOT_AXES = None


def save_results(env_name: str, method_name: str, data: dict) -> None:
    """Save results to JSON file with file locking."""
    file_path: Path = RESULTS_DIR / f"{env_name}_{method_name}.json"
    lock_path: Path = file_path.with_suffix(".lock")
    lock = filelock.FileLock(lock_path, timeout=10)

    with lock:
        with open(file_path, "w") as f:
            json.dump(data, f)


def load_all_results(env_name: str) -> dict[str, dict]:
    """Load all results for an environment with file locking."""
    results: dict[str, dict] = {}
    pattern: str = f"{env_name}_*.json"

    for file_path in RESULTS_DIR.glob(pattern):
        method_name: str = file_path.stem.replace(f"{env_name}_", "")
        lock_path: Path = file_path.with_suffix(".lock")
        lock = filelock.FileLock(lock_path, timeout=10)

        try:
            with lock:
                with open(file_path, "r") as f:
                    content: str = f.read()
                    if content.strip():  # Only load if file has content
                        results[method_name] = json.loads(content)
        except (json.JSONDecodeError, filelock.Timeout):
            # Skip files that are being written or corrupted
            continue

    return results


def update_plot(env_name: str, interactive: bool = False) -> None:
    """Update the real-time plot with current results."""
    global _PLOT_FIG, _PLOT_AXES

    results: dict[str, dict] = load_all_results(env_name)

    if not results:
        return

    if interactive:
        plt.ion()  # Interactive mode
        # Reuse existing figure or create new one
        if _PLOT_FIG is None or not plt.fignum_exists(_PLOT_FIG.number):
            _PLOT_FIG, _PLOT_AXES = plt.subplots(1, 3, figsize=(18, 6))
        fig = _PLOT_FIG
        axes = _PLOT_AXES
    else:
        # Non-interactive: create new figure each time
        plt.ioff()
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Parse method names to separate base method and CL variant
    # E.g., "SGD_with_cl" -> base_method="SGD", cl_variant="with_cl"
    parsed_methods: dict[str, tuple[str, str]] = {}
    for method_name in results.keys():
        if method_name.endswith("_with_cl"):
            base_method: str = method_name.replace("_with_cl", "")
            cl_variant: str = "with_cl"
        elif method_name.endswith("_no_cl"):
            base_method = method_name.replace("_no_cl", "")
            cl_variant = "no_cl"
        else:
            base_method = method_name
            cl_variant = "unknown"
        parsed_methods[method_name] = (base_method, cl_variant)

    # Create consistent color mapping for base methods
    unique_base_methods: list[str] = sorted(
        set(base for base, _ in parsed_methods.values())
    )
    color_map: dict[str, tuple] = {}
    colors_palette = plt.cm.tab10(np.linspace(0, 1, 10))
    for idx, base_method in enumerate(unique_base_methods):
        color_map[base_method] = colors_palette[idx % 10]

    # Define line styles for CL variants
    line_styles: dict[str, str] = {
        "with_cl": "-",  # solid
        "no_cl": "--",  # dashed
    }

    # Plot 1: Test CE Loss Curves
    ax1 = axes[0]
    ax1.clear()
    plotted_methods: set[str] = set()

    for method_name, data in results.items():
        base_method, cl_variant = parsed_methods[method_name]
        plotted_methods.add(method_name)

        if "test_loss" in data and data["test_loss"]:
            # Plot test loss vs runtime %
            if isinstance(data["test_loss"], list):
                # Downsample to exactly 100 points
                original_data: np.ndarray = np.array(data["test_loss"])
                if len(original_data) > 100:
                    x_original: np.ndarray = np.linspace(0, 100, len(original_data))
                    x_new: np.ndarray = np.linspace(0, 100, 100)
                    downsampled_data: np.ndarray = np.interp(
                        x_new, x_original, original_data
                    )
                    runtime_pct: np.ndarray = x_new
                else:
                    runtime_pct: np.ndarray = np.linspace(0, 100, len(original_data))
                    downsampled_data: np.ndarray = original_data

                ax1.plot(
                    runtime_pct,
                    downsampled_data,
                    color=color_map[base_method],
                    linestyle=line_styles.get(cl_variant, "-"),
                    alpha=0.8,
                    linewidth=2,
                )

    # Create custom legend
    sorted_methods: list[str] = sorted(plotted_methods)
    legend_handles: list[Line2D] = []
    legend_labels: list[str] = []

    for method_name in sorted_methods:
        base_method, cl_variant = parsed_methods[method_name]
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color_map[base_method],
                linestyle=line_styles.get(cl_variant, "-"),
                linewidth=2,
            )
        )
        legend_labels.append(format_method_name(method_name))

    ax1.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="best",
        fontsize=8,
    )
    ax1.set_xlabel("Runtime %")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Cross-Entropy Loss")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0]))
    ax1.yaxis.set_major_formatter(LogFormatter(base=10.0))
    ax1.grid(True, alpha=0.3)

    # Plot 2: Test Macro F1 Score Curves
    ax2 = axes[1]
    ax2.clear()
    plotted_methods_f1: set[str] = set()

    for method_name, data in results.items():
        base_method, cl_variant = parsed_methods[method_name]
        plotted_methods_f1.add(method_name)

        if "f1" in data and data["f1"]:
            # Downsample to exactly 100 points
            original_data: np.ndarray = np.array(data["f1"])
            if len(original_data) > 100:
                x_original: np.ndarray = np.linspace(0, 100, len(original_data))
                x_new: np.ndarray = np.linspace(0, 100, 100)
                downsampled_data: np.ndarray = np.interp(
                    x_new, x_original, original_data
                )
                runtime_pct: np.ndarray = x_new
            else:
                runtime_pct: np.ndarray = np.linspace(0, 100, len(original_data))
                downsampled_data: np.ndarray = original_data

            ax2.plot(
                runtime_pct,
                downsampled_data,
                color=color_map[base_method],
                linestyle=line_styles.get(cl_variant, "-"),
                alpha=0.8,
                linewidth=2,
            )

    # Create custom legend
    sorted_methods_f1: list[str] = sorted(plotted_methods_f1)
    legend_handles_f1: list[Line2D] = []
    legend_labels_f1: list[str] = []

    for method_name in sorted_methods_f1:
        base_method, cl_variant = parsed_methods[method_name]
        legend_handles_f1.append(
            Line2D(
                [0],
                [0],
                color=color_map[base_method],
                linestyle=line_styles.get(cl_variant, "-"),
                linewidth=2,
            )
        )
        legend_labels_f1.append(format_method_name(method_name))

    ax2.legend(
        handles=legend_handles_f1,
        labels=legend_labels_f1,
        loc="best",
        fontsize=8,
    )
    ax2.set_xlabel("Runtime %")
    ax2.set_ylabel("Macro F1 Score")
    ax2.set_title("Macro F1 Score")
    ax2.set_yscale("log")
    ax2.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0]))
    ax2.yaxis.set_major_formatter(LogFormatter(base=10.0))
    ax2.grid(True, alpha=0.3)

    # Plot 3: Final Performance Comparison (Grouped Bar Chart)
    ax3 = axes[2]
    ax3.clear()

    # Organize data by method
    method_final_f1: dict[str, float] = {}
    for method_name, data in results.items():
        if "f1" in data and data["f1"]:
            final_f1: float = data["f1"][-1]
            final_error: float = 1.0 - final_f1
            method_final_f1[method_name] = final_error

    if method_final_f1:
        # Sort methods by error (ascending)
        sorted_method_names: list[str] = sorted(
            method_final_f1.keys(), key=lambda m: method_final_f1[m]
        )

        # Prepare data for bar chart
        x_positions: np.ndarray = np.arange(len(sorted_method_names))
        errors: list[float] = [method_final_f1[m] for m in sorted_method_names]

        # Color bars by base method
        bar_colors: list[tuple] = []
        for method_name in sorted_method_names:
            base_method, cl_variant = parsed_methods[method_name]
            bar_colors.append(color_map[base_method])

        bars = ax3.bar(
            x_positions,
            errors,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.5,
        )

        # Add value labels on bars
        for bar, val in zip(bars, errors):
            if val > 0:
                # Format value in scientific notation
                if val >= 0.01:
                    exponent: int = int(np.floor(np.log10(val)))
                    mantissa: float = val / (10**exponent)
                    label: str = f"{mantissa:.2f}e{exponent}"
                else:
                    label = f"{val:.2e}"
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                )

        # Set x-axis labels
        ax3.set_xticks(x_positions)
        display_names: list[str] = [
            format_method_name(m) for m in sorted_method_names
        ]
        ax3.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)
        ax3.set_ylabel("Final Macro F1 Error")
        ax3.set_title("Final Macro F1 Error")
        ax3.set_yscale("log")
        ax3.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0]))
        ax3.yaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=False))
        ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path: Path = SCRIPT_DIR / f"{env_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")

    if interactive:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.01)
    else:
        plt.close(fig)

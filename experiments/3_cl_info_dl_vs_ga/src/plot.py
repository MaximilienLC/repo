"""Visualization functions for evaluation results - focuses on CL info impact."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

from src.config import PLOTS_DIR
from src.utils import format_method_name


def plot_cl_impact(eval_results: dict, save_path: Path | None = None) -> None:
    """Plot 2: Direct comparison showing impact of adding CL features.

    For each base method (SGD, GA), show side-by-side bars of with_cl vs no_cl.
    Shows percentage difference from human - closer to 0% is better.

    Args:
        eval_results: Results from evaluate_all_methods
        save_path: Optional path to save plot
    """
    env_name: str = eval_results["env_name"]
    model_stats: dict = eval_results["model_stats"]

    if not model_stats:
        print("No model results to plot")
        return

    # Group methods by base method
    base_methods: dict[str, dict] = {}
    for method_name, stats in model_stats.items():
        # Extract base method (SGD, adaptive_ga_CE)
        if "_with_cl" in method_name:
            base: str = method_name.replace("_with_cl", "")
            if base not in base_methods:
                base_methods[base] = {}
            base_methods[base]["with_cl"] = stats
        elif "_no_cl" in method_name:
            base = method_name.replace("_no_cl", "")
            if base not in base_methods:
                base_methods[base] = {}
            base_methods[base]["no_cl"] = stats

    if not base_methods:
        print("No paired methods found for CL comparison")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    base_names: list[str] = sorted(base_methods.keys())
    x_pos: np.ndarray = np.arange(len(base_names))
    width: float = 0.35

    # Extract data - use mean_pct_diff instead of mean_return
    no_cl_means: list[float] = []
    no_cl_stds: list[float] = []
    with_cl_means: list[float] = []
    with_cl_stds: list[float] = []

    for base in base_names:
        no_cl: dict = base_methods[base].get("no_cl", {})
        with_cl: dict = base_methods[base].get("with_cl", {})

        no_cl_means.append(no_cl.get("mean_pct_diff", 0.0))
        no_cl_stds.append(no_cl.get("std_pct_diff", 0.0))
        with_cl_means.append(with_cl.get("mean_pct_diff", 0.0))
        with_cl_stds.append(with_cl.get("std_pct_diff", 0.0))

    # Set up color palette from tab10
    colors_palette = plt.cm.tab10(np.linspace(0, 1, 10))

    # Color mapping for base methods: SGD -> blue (0), GA -> brown (5)
    base_method_colors: dict[str, tuple] = {
        "SGD": colors_palette[0],  # Blue
        "adaptive_ga_CE": colors_palette[5],  # Brown
    }

    # Build colors for each base method
    colors_no_cl: list = []
    colors_with_cl: list = []
    for base in base_names:
        color = base_method_colors.get(
            base, colors_palette[7]
        )  # Gray fallback
        colors_no_cl.append(color)
        colors_with_cl.append(color)

    # Plot bars
    bars1 = ax.bar(
        x_pos - width / 2,
        no_cl_means,
        width,
        yerr=no_cl_stds,
        label="Without CL Info",
        color=colors_no_cl,
        alpha=0.8,
        capsize=5,
        edgecolor="black",
        linewidth=1.5,
    )
    bars2 = ax.bar(
        x_pos + width / 2,
        with_cl_means,
        width,
        yerr=with_cl_stds,
        label="With CL Info",
        color=colors_with_cl,
        alpha=0.8,
        capsize=5,
        edgecolor="black",
        linewidth=1.5,
    )

    # Apply hatch patterns: with_cl gets hatching, no_cl is solid
    for bar in bars2:
        bar.set_hatch("///")

    # Add perfect match baseline (0%)
    ax.axhline(
        y=0,
        color="#2ecc71",
        linestyle="--",
        linewidth=2,
        label="Perfect Match (0%)",
    )

    # Add value labels and improvement indicators
    for i, (base, no_cl_mean, with_cl_mean) in enumerate(
        zip(base_names, no_cl_means, with_cl_means)
    ):
        # Value labels with appropriate positioning for positive/negative values
        for mean_val, x_offset in [
            (no_cl_mean, -width / 2),
            (with_cl_mean, width / 2),
        ]:
            if mean_val >= 0:
                ax.text(
                    x_pos[i] + x_offset,
                    mean_val,
                    f"{mean_val:+.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
            else:
                ax.text(
                    x_pos[i] + x_offset,
                    mean_val,
                    f"{mean_val:+.1f}%",
                    ha="center",
                    va="top",
                    fontsize=10,
                )

        # Improvement indicator: getting closer to 0% is better
        abs_no_cl: float = abs(no_cl_mean)
        abs_with_cl: float = abs(with_cl_mean)
        improvement_to_zero: float = (
            abs_no_cl - abs_with_cl
        )  # Positive = improvement

        # Arrow showing change
        if (
            abs(no_cl_mean - with_cl_mean) > 0.1
        ):  # Only show if meaningful difference
            ax.annotate(
                "",
                xy=(x_pos[i] + width / 2, with_cl_mean),
                xytext=(x_pos[i] - width / 2, no_cl_mean),
                arrowprops=dict(
                    arrowstyle="->", color="black", lw=1.5, alpha=0.5
                ),
            )

            # Label with improvement metric (reduction in absolute difference)
            # Position at bottom of plot
            symbol: str = "✓" if improvement_to_zero > 0 else "✗"
            ax.text(
                x_pos[i],
                ax.get_ylim()[0],
                f"{symbol} {abs(improvement_to_zero):.1f}pp",
                ha="center",
                va="top",
                fontsize=10,
                color="green" if improvement_to_zero > 0 else "red",
            )

    # Formatting
    ax.set_xticks([])  # Remove x-axis ticks and labels
    ax.set_ylabel("% Difference from Human (per episode)", fontsize=13)
    ax.set_title(
        f"End-of-optimization impact of providing CL information - {env_name.capitalize()}",
        fontsize=14,
    )

    # Custom legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor="#2ecc71",
            edgecolor="black",
            alpha=0.3,
            label="Perfect Match",
        ),
        Patch(facecolor=colors_palette[0], edgecolor="black", label="SGD"),
        Patch(
            facecolor=colors_palette[0],
            edgecolor="black",
            hatch="///",
            label="SGD w/ CL info",
        ),
        Patch(
            facecolor=colors_palette[5], edgecolor="black", label="Adaptive GA"
        ),
        Patch(
            facecolor=colors_palette[5],
            edgecolor="black",
            hatch="///",
            label="Adaptive GA w/ CL info",
        ),
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc="lower center")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  CL impact plot saved to {save_path}")
    else:
        default_path: Path = PLOTS_DIR / f"cl_impact_{env_name}.png"
        plt.savefig(default_path, dpi=150, bbox_inches="tight")
        print(f"  CL impact plot saved to {default_path}")

    plt.close(fig)


def create_comparison_table(eval_results: dict) -> None:
    """Print formatted table comparing all methods.

    Args:
        eval_results: Results from evaluate_all_methods
    """
    env_name: str = eval_results["env_name"]
    human_stats: dict = eval_results["human_stats"]
    model_stats: dict = eval_results["model_stats"]
    comparisons: dict = eval_results["comparisons"]

    print(f"\n{'='*100}")
    print(f"EVALUATION RESULTS: {env_name.upper()}")
    print(f"{'='*100}")

    # Human baseline
    print(f"\nHuman Baseline ({human_stats['num_episodes']} episodes):")
    print(
        f"  Mean Return: {human_stats['mean_return']:8.2f} ± {human_stats['std_return']:.2f}"
    )
    print(f"  Median:      {human_stats['median_return']:8.2f}")
    print(
        f"  Range:       [{human_stats['min_return']:.2f}, {human_stats['max_return']:.2f}]"
    )
    print(
        f"  Q25-Q75:     [{human_stats['q25_return']:.2f}, {human_stats['q75_return']:.2f}]"
    )
    print(f"  Mean Length: {human_stats['mean_length']:8.1f} steps")

    if not model_stats:
        print("\nNo model results available.")
        return

    # Table header
    print(f"\n{'-'*110}")
    print(
        f"{'Method':<25} | {'Mean % Diff':>12} | {'Std % Diff':>12} | "
        f"{'Abs Diff':>10} | {'p-value':>10} | {'Sig':>5}"
    )
    print(f"{'-'*110}")

    # Sort methods by absolute mean percentage difference (closest to 0 = best)
    sorted_methods: list[str] = sorted(
        model_stats.keys(),
        key=lambda m: abs(model_stats[m]["mean_pct_diff"]),
    )

    for method_name in sorted_methods:
        stats: dict = model_stats[method_name]
        comp: dict = comparisons[method_name]

        display_name: str = format_method_name(method_name)
        mean_pct_diff: float = stats["mean_pct_diff"]
        std_pct_diff: float = stats["std_pct_diff"]
        abs_pct_diff: float = abs(mean_pct_diff)
        pvalue: float = comp["u_pvalue"]
        sig: str = (
            "***"
            if comp["significant_001"]
            else ("**" if comp["significant_005"] else "")
        )

        print(
            f"{display_name:<25} | {mean_pct_diff:>+11.2f}% | {std_pct_diff:>11.2f}% | "
            f"{abs_pct_diff:>9.2f}% | {pvalue:>10.4f} | {sig:>5}"
        )

    print(f"{'-'*100}")
    print("\n** = p < 0.05, *** = p < 0.01 (Mann-Whitney U test vs Human)")

    # CL Impact Analysis
    print(f"\n{'='*100}")
    print("CL INFORMATION IMPACT ANALYSIS")
    print(f"{'='*100}")

    # Group by base method
    base_methods: dict[str, dict] = {}
    for method_name, stats in model_stats.items():
        if "_with_cl" in method_name:
            base: str = method_name.replace("_with_cl", "")
            if base not in base_methods:
                base_methods[base] = {}
            base_methods[base]["with_cl"] = stats
        elif "_no_cl" in method_name:
            base = method_name.replace("_no_cl", "")
            if base not in base_methods:
                base_methods[base] = {}
            base_methods[base]["no_cl"] = stats

    if base_methods:
        print(
            f"\n{'Base Method':<20} | {'No CL % Diff':>13} | {'With CL % Diff':>15} | {'Improvement':>15} | {'Better?':>10}"
        )
        print(f"{'-'*100}")

        for base in sorted(base_methods.keys()):
            if (
                "no_cl" in base_methods[base]
                and "with_cl" in base_methods[base]
            ):
                no_cl_pct_diff: float = base_methods[base]["no_cl"][
                    "mean_pct_diff"
                ]
                with_cl_pct_diff: float = base_methods[base]["with_cl"][
                    "mean_pct_diff"
                ]

                # Improvement: reduction in absolute deviation from 0
                abs_improvement: float = abs(no_cl_pct_diff) - abs(
                    with_cl_pct_diff
                )

                better: str = "✓ Yes" if abs_improvement > 0 else "✗ No"

                print(
                    f"{format_method_name(base):<20} | {no_cl_pct_diff:>+12.2f}% | {with_cl_pct_diff:>+14.2f}% | "
                    f"{abs_improvement:>+7.2f}pp closer | {better:>10}"
                )

        print(f"{'-'*90}")

    print(f"\n{'='*100}\n")


def plot_progression_over_time(
    env_name: str, subject: str = "sub01", save_path: Path | None = None
) -> None:
    """Plot progression of mean % difference from human over optimization time.

    Loads checkpoints and plots how each method's similarity to human behavior
    evolves during optimization. X-axis is runtime percentage (0-100%).

    Args:
        env_name: Environment name
        subject: Subject identifier (sub01, sub02)
        save_path: Optional path to save plot
    """
    import torch
    from src.config import RESULTS_DIR

    # Find all checkpoint files for this environment and subject
    checkpoint_pattern: str = f"{env_name}_*_{subject}_checkpoint.pt"
    checkpoint_files: list[Path] = list(RESULTS_DIR.glob(checkpoint_pattern))

    if not checkpoint_files:
        print(f"  No checkpoint files found matching {checkpoint_pattern}")
        return

    # Load progression history from each checkpoint
    progression_data: dict[str, list[dict]] = {}

    for checkpoint_file in checkpoint_files:
        # Extract method name from filename: env_name_METHOD_subject_checkpoint.pt
        filename_parts: list[str] = checkpoint_file.stem.split("_")
        # Remove env_name, subject, and "checkpoint" to get method name
        # Format: env_method1_method2_..._subject_checkpoint
        method_name: str = "_".join(
            filename_parts[1:-2]
        )  # Everything between env and subject

        try:
            checkpoint: dict = torch.load(
                checkpoint_file, weights_only=False, map_location="cpu"
            )
            prog_history: list[dict] = checkpoint.get(
                "progression_history", []
            )
            optim_time: float | None = checkpoint.get("optim_time")

            if prog_history and optim_time is not None:
                # Calculate runtime_pct for each entry
                for entry in prog_history:
                    entry["runtime_pct"] = (
                        entry["elapsed_time"] / optim_time
                    ) * 100.0

                progression_data[method_name] = prog_history
                print(
                    f"  Loaded {len(prog_history)} progression points for {method_name}"
                )
            elif not prog_history:
                print(
                    f"  No progression history in checkpoint for {method_name}"
                )
            elif optim_time is None:
                print(
                    f"  No optim_time in checkpoint for {method_name}, skipping"
                )
        except Exception as e:
            print(f"  Error loading {checkpoint_file}: {e}")

    if not progression_data:
        print("  No progression data found in any checkpoints")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Set up color palette from tab10
    colors_palette = plt.cm.tab10(np.linspace(0, 1, 10))

    # Color mapping for base methods: SGD -> blue (0), GA -> brown (5)
    base_method_colors: dict[str, tuple] = {
        "SGD": colors_palette[0],  # Blue
        "adaptive_ga_CE": colors_palette[5],  # Brown
    }

    # Collect method info for custom legend
    method_info: list[tuple[str, tuple, bool | None]] = []

    # Plot each method
    for method_name in sorted(progression_data.keys()):
        prog_history: list[dict] = progression_data[method_name]

        # Extract data
        runtime_pcts: list[float] = [
            entry["runtime_pct"] for entry in prog_history
        ]
        mean_pct_diffs: list[float] = [
            entry["mean_pct_diff"] for entry in prog_history
        ]
        std_pct_diffs: list[float] = [
            entry.get("std_pct_diff", 0.0) for entry in prog_history
        ]

        # Downsample to max 50 points (including first and last)
        max_points: int = 50
        if len(runtime_pcts) > max_points:
            # Get evenly-spaced indices including first (0) and last (-1)
            indices: np.ndarray = np.linspace(
                0, len(runtime_pcts) - 1, max_points
            )
            indices = np.round(indices).astype(int)
            indices = np.unique(indices)  # Ensure uniqueness

            # Subsample the data
            runtime_pcts = [runtime_pcts[i] for i in indices]
            mean_pct_diffs = [mean_pct_diffs[i] for i in indices]
            std_pct_diffs = [std_pct_diffs[i] for i in indices]

        # Convert to numpy arrays for arithmetic
        runtime_pcts_arr: np.ndarray = np.array(runtime_pcts)
        mean_pct_diffs_arr: np.ndarray = np.array(mean_pct_diffs)
        std_pct_diffs_arr: np.ndarray = np.array(std_pct_diffs)

        # Extract base method (remove _with_cl or _no_cl suffix)
        if "_with_cl" in method_name:
            base_method: str = method_name.replace("_with_cl", "")
            has_cl: bool = True
        elif "_no_cl" in method_name:
            base_method = method_name.replace("_no_cl", "")
            has_cl = False
        else:
            base_method = method_name
            has_cl = None

        # Get color for this base method (default to gray if unknown)
        color = base_method_colors.get(
            base_method, colors_palette[7]
        )  # Gray fallback

        # Store method info for legend
        label: str = format_method_name(method_name)
        method_info.append((label, color, has_cl))

        # Plot std bounds as thin lines
        if has_cl:
            # For with_cl: black solid line first, then colored dashed line on top
            # Upper bound
            ax.plot(
                runtime_pcts_arr,
                mean_pct_diffs_arr + std_pct_diffs_arr,
                color="black",
                linestyle="-",
                linewidth=1.0,
                alpha=0.5,
            )
            ax.plot(
                runtime_pcts_arr,
                mean_pct_diffs_arr + std_pct_diffs_arr,
                color=color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.5,
            )
            # Lower bound
            ax.plot(
                runtime_pcts_arr,
                mean_pct_diffs_arr - std_pct_diffs_arr,
                color="black",
                linestyle="-",
                linewidth=1.0,
                alpha=0.5,
            )
            ax.plot(
                runtime_pcts_arr,
                mean_pct_diffs_arr - std_pct_diffs_arr,
                color=color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.5,
            )
        else:
            # For no_cl: solid colored lines
            # Upper bound
            ax.plot(
                runtime_pcts_arr,
                mean_pct_diffs_arr + std_pct_diffs_arr,
                color=color,
                linestyle="-",
                linewidth=1.0,
                alpha=0.5,
            )
            # Lower bound
            ax.plot(
                runtime_pcts_arr,
                mean_pct_diffs_arr - std_pct_diffs_arr,
                color=color,
                linestyle="-",
                linewidth=1.0,
                alpha=0.5,
            )

        if has_cl:
            # For with_cl: plot black solid line first, then dashed colored line on top
            # Black underlay (no label)
            ax.plot(
                runtime_pcts,
                mean_pct_diffs,
                marker="o",
                markersize=6,
                linewidth=2.5,
                color="black",
                linestyle="-",
                alpha=0.8,
            )
            # Colored dashed overlay (no label - will create custom legend)
            ax.plot(
                runtime_pcts,
                mean_pct_diffs,
                marker="o",
                markersize=6,
                linewidth=2.5,
                color=color,
                linestyle="--",
                alpha=0.8,
            )
        else:
            # For no_cl: solid colored line (no label - will create custom legend)
            ax.plot(
                runtime_pcts,
                mean_pct_diffs,
                marker="o",
                markersize=6,
                linewidth=2.5,
                color=color,
                linestyle="-",
                alpha=0.8,
            )

    # Perfect match line
    ax.axhline(y=0, color="green", linestyle="--", linewidth=2, alpha=0.7)

    # Create custom legend with styling that reflects CL pattern
    from matplotlib.lines import Line2D

    legend_handles: list[Line2D] = []
    legend_labels: list[str] = []

    # Add perfect match line first
    perfect_match_handle = Line2D(
        [], [], color="green", linestyle="--", linewidth=2, alpha=0.7
    )
    legend_handles.append(perfect_match_handle)
    legend_labels.append("Perfect Match")

    # Add SGD methods
    sgd_handle = Line2D(
        [],
        [],
        marker="o",
        markersize=6,
        linewidth=2.5,
        color=colors_palette[0],
        linestyle="-",
        alpha=0.8,
    )
    legend_handles.append(sgd_handle)
    legend_labels.append("SGD")

    sgd_cl_handle = Line2D(
        [],
        [],
        marker="o",
        markersize=6,
        linewidth=2.5,
        color=colors_palette[0],
        linestyle="--",
        alpha=0.8,
    )
    legend_handles.append(sgd_cl_handle)
    legend_labels.append("SGD w/ CL info")

    # Add Adaptive GA methods
    ga_handle = Line2D(
        [],
        [],
        marker="o",
        markersize=6,
        linewidth=2.5,
        color=colors_palette[5],
        linestyle="-",
        alpha=0.8,
    )
    legend_handles.append(ga_handle)
    legend_labels.append("Adaptive GA")

    ga_cl_handle = Line2D(
        [],
        [],
        marker="o",
        markersize=6,
        linewidth=2.5,
        color=colors_palette[5],
        linestyle="--",
        alpha=0.8,
    )
    legend_handles.append(ga_cl_handle)
    legend_labels.append("Adaptive GA w/ CL info")

    # Formatting
    ax.set_xlabel("Runtime %", fontsize=13)
    ax.set_ylabel("% Difference from Human (per episode)", fontsize=13)
    ax.set_title(
        f"Behavioural difference over the course of optimization - {env_name.capitalize()}",
        fontsize=14,
    )
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(
        handles=legend_handles, labels=legend_labels, fontsize=11, loc="best"
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Progression plot saved to {save_path}")
    else:
        default_path: Path = PLOTS_DIR / f"behav_progress_{env_name}.png"
        plt.savefig(default_path, dpi=150, bbox_inches="tight")
        print(f"  Progression plot saved to {default_path}")

    plt.close(fig)


def plot_optimization_progress(
    env_name: str, subject: str = "sub01", save_path: Path | None = None
) -> None:
    """Plot optimization progress curves showing test loss and F1 over time.

    Creates 3 subplots similar to experiment 2:
    1. Test CE Loss curves over optimization (CE methods only)
    2. Test F1 Score curves over optimization (all methods)
    3. Final performance comparison (grouped bar chart)

    Args:
        env_name: Environment name
        subject: Subject identifier (sub01, sub02)
        save_path: Optional path to save plot
    """
    import json
    from matplotlib.ticker import LogLocator, LogFormatter
    from src.config import RESULTS_DIR

    # Load all JSON result files for this environment and subject
    result_pattern: str = f"{env_name}_*_{subject}.json"
    result_files: list[Path] = list(RESULTS_DIR.glob(result_pattern))

    if not result_files:
        print(f"  No result files found matching {result_pattern}")
        return

    # Load results from JSON files
    results: dict[str, dict] = {}
    for result_file in result_files:
        # Extract method name from filename: env_name_METHOD_subject.json
        filename_parts: list[str] = result_file.stem.split("_")
        # Remove env_name and subject to get method name
        method_name: str = "_".join(
            filename_parts[1:-1]
        )  # Everything between env and subject

        try:
            with open(result_file, "r") as f:
                content: str = f.read()
                if content.strip():
                    results[method_name] = json.loads(content)
                    print(f"  Loaded results for {method_name}")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"  Error loading {result_file}: {e}")

    if not results:
        print("  No valid result data found")
        return

    # Parse method names to identify base method and CL status
    parsed_methods: dict[str, tuple[str, bool | None]] = {}
    for method_name in results.keys():
        if "_with_cl" in method_name:
            base_method: str = method_name.replace("_with_cl", "")
            has_cl: bool = True
        elif "_no_cl" in method_name:
            base_method = method_name.replace("_no_cl", "")
            has_cl = False
        else:
            base_method = method_name
            has_cl = None
        parsed_methods[method_name] = (base_method, has_cl)

    # Create color mapping for base methods
    unique_base_methods: list[str] = sorted(
        set(base for base, _ in parsed_methods.values())
    )
    colors_palette = plt.cm.tab10(np.linspace(0, 1, 10))
    color_map: dict[str, tuple] = {
        "SGD": colors_palette[0],  # Blue
        "adaptive_ga_CE": colors_palette[5],  # Brown
    }
    # Add any other methods with default colors
    for idx, base_method in enumerate(unique_base_methods):
        if base_method not in color_map:
            color_map[base_method] = colors_palette[(idx + 2) % 10]

    # Line styles based on CL status
    line_styles: dict[bool | None, str] = {
        False: "-",  # solid for no_cl
        True: "--",  # dashed for with_cl
        None: "-",  # solid for undefined
    }

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Subplot 1: Test CE Loss Curves (CE-optimizing methods only)
    ax1 = axes[0]
    ce_methods: set[str] = set()

    for method_name, data in results.items():
        base_method, has_cl = parsed_methods[method_name]
        # Only show CE-optimizing methods (exclude F1-optimizing if any exist)
        if "_F1" in base_method:
            continue
        ce_methods.add(base_method)

        if "test_loss" in data and data["test_loss"]:
            # Downsample to exactly 100 points
            original_data: np.ndarray = np.array(data["test_loss"])
            if len(original_data) > 100:
                # Interpolate to get exactly 100 evenly-spaced points
                x_original: np.ndarray = np.linspace(
                    0, 100, len(original_data)
                )
                x_new: np.ndarray = np.linspace(0, 100, 100)
                downsampled_data: np.ndarray = np.interp(
                    x_new, x_original, original_data
                )
                runtime_pct: np.ndarray = x_new
            else:
                runtime_pct: np.ndarray = np.linspace(
                    0, 100, len(original_data)
                )
                downsampled_data: np.ndarray = original_data

            if has_cl:
                # For with_cl: plot black solid line first, then dashed colored line on top
                # Black underlay
                ax1.plot(
                    runtime_pct,
                    downsampled_data,
                    color="black",
                    linestyle="-",
                    alpha=0.8,
                    linewidth=2.5,
                )
                # Colored dashed overlay
                ax1.plot(
                    runtime_pct,
                    downsampled_data,
                    color=color_map[base_method],
                    linestyle="--",
                    alpha=0.8,
                    linewidth=2.5,
                )
            else:
                # For no_cl: solid colored line
                ax1.plot(
                    runtime_pct,
                    downsampled_data,
                    color=color_map[base_method],
                    linestyle="-",
                    alpha=0.8,
                    linewidth=2.5,
                )

    ax1.set_xlabel("Runtime %", fontsize=12)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax1.set_title("Cross-Entropy Loss", fontsize=13)
    ax1.set_yscale("log")
    ax1.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0]))
    ax1.yaxis.set_major_formatter(LogFormatter(base=10.0))
    ax1.grid(True, alpha=0.3)
    # Add line style explanation

    # Subplot 2: Test Macro F1 Score Curves (all methods)
    ax2 = axes[1]
    all_methods: set[str] = set()

    for method_name, data in results.items():
        base_method, has_cl = parsed_methods[method_name]
        all_methods.add(base_method)

        if "f1" in data and data["f1"]:
            # Downsample to exactly 100 points
            original_data: np.ndarray = np.array(data["f1"])
            if len(original_data) > 100:
                x_original: np.ndarray = np.linspace(
                    0, 100, len(original_data)
                )
                x_new: np.ndarray = np.linspace(0, 100, 100)
                downsampled_data: np.ndarray = np.interp(
                    x_new, x_original, original_data
                )
                runtime_pct: np.ndarray = x_new
            else:
                runtime_pct: np.ndarray = np.linspace(
                    0, 100, len(original_data)
                )
                downsampled_data: np.ndarray = original_data

            if has_cl:
                # For with_cl: plot black solid line first, then dashed colored line on top
                # Black underlay
                ax2.plot(
                    runtime_pct,
                    downsampled_data,
                    color="black",
                    linestyle="-",
                    alpha=0.8,
                    linewidth=2.5,
                )
                # Colored dashed overlay
                ax2.plot(
                    runtime_pct,
                    downsampled_data,
                    color=color_map[base_method],
                    linestyle="--",
                    alpha=0.8,
                    linewidth=2.5,
                )
            else:
                # For no_cl: solid colored line
                ax2.plot(
                    runtime_pct,
                    downsampled_data,
                    color=color_map[base_method],
                    linestyle="-",
                    alpha=0.8,
                    linewidth=2.5,
                )

    ax2.set_xlabel("Runtime %", fontsize=12)
    ax2.set_ylabel("Macro F1 Score", fontsize=12)
    ax2.set_title("Macro F1 Score", fontsize=13)
    ax2.set_yscale("log")
    ax2.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0]))
    ax2.yaxis.set_major_formatter(LogFormatter(base=10.0))
    ax2.grid(True, alpha=0.3)
    # Add line style explanation

    # Subplot 3: Final Performance Comparison (Grouped Bar Chart)
    ax3 = axes[2]

    # Organize data by base method and CL status
    method_data: dict[str, dict[str, float]] = {}
    for method_name, data in results.items():
        base_method, has_cl = parsed_methods[method_name]
        if "f1" in data and data["f1"]:
            final_f1: float = data["f1"][-1]
            final_error: float = 1.0 - final_f1  # Compute error instead of F1
            if base_method not in method_data:
                method_data[base_method] = {}
            cl_label: str = "with_cl" if has_cl else "no_cl"
            method_data[base_method][cl_label] = final_error

    if method_data:
        # Sort methods by best error (min across all CL statuses)
        sorted_base_methods: list[str] = sorted(
            method_data.keys(),
            key=lambda m: min(method_data[m].values()),
        )

        bar_width: float = 0.35
        x_positions: np.ndarray = np.arange(len(sorted_base_methods))

        # Extract data for no_cl and with_cl
        no_cl_errors: list[float] = [
            method_data[m].get("no_cl", 0.0) for m in sorted_base_methods
        ]
        with_cl_errors: list[float] = [
            method_data[m].get("with_cl", 0.0) for m in sorted_base_methods
        ]

        # Bar colors
        bar_colors_no_cl: list[tuple] = [
            (*color_map[m][:3], 0.8) for m in sorted_base_methods
        ]
        bar_colors_with_cl: list[tuple] = [
            (*color_map[m][:3], 0.8) for m in sorted_base_methods
        ]

        # Plot bars
        bars1 = ax3.bar(
            x_positions - bar_width / 2,
            no_cl_errors,
            bar_width,
            label="No CL",
            color=bar_colors_no_cl,
            edgecolor="black",
            linewidth=1.0,
        )
        bars2 = ax3.bar(
            x_positions + bar_width / 2,
            with_cl_errors,
            bar_width,
            label="With CL",
            color=bar_colors_with_cl,
            edgecolor="black",
            linewidth=1.0,
        )

        # Apply hatch pattern to with_cl bars
        for bar in bars2:
            bar.set_hatch("///")

        # Add value labels on bars
        for bars, errors in [(bars1, no_cl_errors), (bars2, with_cl_errors)]:
            for bar, val in zip(bars, errors):
                if val > 0:
                    # Format value in scientific notation
                    if val >= 0.01:
                        exponent: int = int(np.floor(np.log10(val)))
                        mantissa: float = val / (10**exponent)
                        label: str = f"{mantissa:.2f}e{exponent}"
                    else:
                        label: str = f"{val:.2e}"
                    ax3.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        label,
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        # Remove x-axis ticks and labels
        ax3.set_xticks([])
        ax3.set_ylabel("Final Macro F1 Error", fontsize=12)
        ax3.set_title("Final Macro F1 Error", fontsize=13)
        ax3.set_yscale("log")
        ax3.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0]))
        ax3.yaxis.set_major_formatter(
            LogFormatter(base=10.0, labelOnlyBase=False)
        )

        # Custom legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=colors_palette[0], edgecolor="black", label="SGD"),
            Patch(
                facecolor=colors_palette[0],
                edgecolor="black",
                hatch="///",
                label="SGD w/ CL info",
            ),
            Patch(
                facecolor=colors_palette[5],
                edgecolor="black",
                label="Adaptive GA",
            ),
            Patch(
                facecolor=colors_palette[5],
                edgecolor="black",
                hatch="///",
                label="Adaptive GA w/ CL info",
            ),
        ]
        ax3.legend(handles=legend_elements, loc="best", fontsize=10)
        ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Optimization progress plot saved to {save_path}")
    else:
        default_path: Path = PLOTS_DIR / f"loss_progress_{env_name}.png"
        plt.savefig(default_path, dpi=150, bbox_inches="tight")
        print(f"  Optimization progress plot saved to {default_path}")

    plt.close(fig)


def create_all_plots(eval_results: dict) -> None:
    """Create all evaluation plots.

    Args:
        eval_results: Results from evaluate_all_methods
    """
    print("\nGenerating evaluation plots...")

    env_name: str = eval_results["env_name"]
    subject: str = eval_results.get("subject", "sub01")

    plot_cl_impact(eval_results)
    plot_progression_over_time(env_name, subject)
    plot_optimization_progress(env_name, subject)

    print("All plots generated successfully!")

"""Main script for Experiment 3: Continual Learning Information Ablation."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor

from src.plot import (
    create_all_plots,
    create_comparison_table,
)
from src import config
from src.config import (
    ENV_CONFIGS,
    RESULTS_DIR,
    SCRIPT_DIR,
    ExperimentConfig,
    set_device,
)
from src.data import load_human_data
from src.optim import deeplearn, neuroevolve
from src.utils import set_random_seeds


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""

    def default(self, obj):
        """Convert numpy types to native Python types."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


def get_all_methods() -> list[tuple[str, dict]]:
    """Get all method configurations."""
    return [
        ("SGD", {"type": "dl"}),
        ("adaptive_ga_CE", {"type": "ne"}),
    ]


def run_single_method(
    env_name: str,
    method_name: str,
    method_config: dict,
    use_cl_info: bool,
    optim_obs: Float[Tensor, "optim_size input_size"],
    optim_act: Int[Tensor, " optim_size"],
    test_obs: Float[Tensor, "test_size input_size"],
    test_act: Int[Tensor, " test_size"],
    input_size: int,
    output_size: int,
    exp_config: ExperimentConfig,
    subject: str = "sub01",
) -> None:
    """Run a single optimization method."""
    # Append CL variant to method name
    cl_suffix: str = "with_cl" if use_cl_info else "no_cl"
    method_name_full: str = f"{method_name}_{cl_suffix}"

    print(f"\n{'='*60}")
    print(f"Running {method_name_full} for {env_name} ({subject}'s data)")
    print(f"{'='*60}")
    print(f"Optim size: {optim_obs.shape[0]}, Test size: {test_obs.shape[0]}")
    print(f"Input size: {input_size}, Output size: {output_size}")

    if method_config["type"] == "dl":
        deeplearn(
            optim_obs,
            optim_act,
            test_obs,
            test_act,
            input_size,
            output_size,
            exp_config,
            env_name,
            method_name_full,
            subject,
        )
    else:
        neuroevolve(
            optim_obs,
            optim_act,
            test_obs,
            test_act,
            input_size,
            output_size,
            exp_config,
            env_name,
            method_name_full,
            subject,
        )


def main() -> None:
    """Main function to run Experiment 3."""
    parser = argparse.ArgumentParser(
        description="Experiment 3: Continual Learning Information Ablation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cartpole", "mountaincar", "acrobot", "lunarlander"],
        required=True,
        help="Environment to use",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["SGD", "adaptive_ga_CE"],
        help="Method to run. Use --list-methods to see all options.",
    )
    parser.add_argument(
        "--use-cl-info",
        action="store_true",
        help="Include continual learning (session/run) information as input features",
    )
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="List all available methods",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot mode: generate all plots from saved checkpoint data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to use (default: 0)",
    )
    parser.add_argument(
        "--subject",
        type=str,
        choices=["sub01", "sub02"],
        default="sub01",
        help="Subject whose data to use (default: sub01)",
    )

    args = parser.parse_args()

    # Set global DEVICE based on --gpu argument
    set_device(args.gpu)
    print(f"Using device: {config.DEVICE}")

    all_methods: list[tuple[str, dict]] = get_all_methods()
    method_dict: dict[str, dict] = {name: cfg for name, cfg in all_methods}

    if args.list_methods:
        print("Available methods:")
        for name, _ in all_methods:
            print(f"  - {name}")
        return

    # Setup environment
    env_config: dict = ENV_CONFIGS[args.dataset]
    env_name: str = args.dataset
    obs_dim: int = env_config["obs_dim"]
    action_dim: int = env_config["action_dim"]

    # Plot mode
    if args.plot:
        print(f"\n{'='*60}")
        print(f"PLOT MODE: {env_name.upper()} ({args.subject.upper()}'S DATA)")
        print(f"{'='*60}")

        # Load human data
        from src.evaluation import compare_returns
        import numpy as np

        print(f"\nLoading human data...")
        _, _, _, _, metadata = load_human_data(
            env_name, use_cl_info=False, subject=args.subject, holdout_pct=0.1
        )

        # Load episode details from metadata
        metadata_path: Path = (
            RESULTS_DIR / f"metadata_{env_name}_{args.subject}.json"
        )
        if not metadata_path.exists():
            print(f"Error: Metadata file not found: {metadata_path}")
            print(f"Please run optimization first to generate metadata.")
            return

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Extract human episode details
        test_episode_info: list[dict] = metadata["test_episode_info"]
        human_returns: list[float] = [ep["return"] for ep in test_episode_info]

        # Compute human stats
        human_returns_array: np.ndarray = np.array(human_returns)
        human_lengths: list[int] = [
            ep["num_steps"] for ep in test_episode_info
        ]
        human_lengths_array: np.ndarray = np.array(human_lengths)

        human_stats: dict = {
            "returns": human_returns,
            "lengths": human_lengths,
            "mean_return": float(np.mean(human_returns_array)),
            "std_return": float(np.std(human_returns_array)),
            "median_return": float(np.median(human_returns_array)),
            "min_return": float(np.min(human_returns_array)),
            "max_return": float(np.max(human_returns_array)),
            "q25_return": float(np.percentile(human_returns_array, 25)),
            "q75_return": float(np.percentile(human_returns_array, 75)),
            "mean_length": float(np.mean(human_lengths_array)),
            "num_episodes": len(test_episode_info),
        }

        print(f"  Loaded {human_stats['num_episodes']} human test episodes")
        print(
            f"  Human mean return: {human_stats['mean_return']:.2f} ± {human_stats['std_return']:.2f}"
        )

        # Find all checkpoints for this subject and environment
        checkpoint_pattern: str = f"{env_name}_*_{args.subject}_checkpoint.pt"
        checkpoint_files: list[Path] = list(
            RESULTS_DIR.glob(checkpoint_pattern)
        )

        if not checkpoint_files:
            print(
                f"\nError: No checkpoints found for {env_name} ({args.subject})"
            )
            print(f"  Pattern: {checkpoint_pattern}")
            print(f"  Directory: {RESULTS_DIR}")
            return

        print(f"\nFound {len(checkpoint_files)} checkpoint(s)")

        # Build eval_results structure from checkpoints
        model_stats: dict[str, dict] = {}
        comparisons: dict[str, dict] = {}

        for checkpoint_path in checkpoint_files:
            # Extract method name from filename
            method_name: str = checkpoint_path.stem.replace(
                f"{env_name}_", ""
            ).replace(f"_{args.subject}_checkpoint", "")

            print(f"\n  Loading {method_name}...")

            # Load checkpoint
            checkpoint: dict = torch.load(
                checkpoint_path, weights_only=False, map_location="cpu"
            )

            # Get final progression entry
            progression_history: list[dict] = checkpoint.get(
                "progression_history", []
            )

            if not progression_history:
                print(
                    f"    Warning: No progression history found in checkpoint"
                )
                continue

            final_entry: dict = progression_history[-1]

            # Extract model returns from final entry
            model_returns: list[float] = final_entry.get("model_returns", [])

            if not model_returns:
                print(
                    f"    Warning: No model_returns found in final progression entry"
                )
                continue

            # Compute stats
            model_returns_array: np.ndarray = np.array(model_returns)
            model_lengths: list[int] = [0] * len(
                model_returns
            )  # Not tracked, placeholder

            # Compute percentage differences
            pct_differences: list[float] = []
            for model_ret, human_ret in zip(model_returns, human_returns):
                if human_ret != 0:
                    pct_diff: float = (
                        (model_ret - human_ret) / abs(human_ret)
                    ) * 100
                else:
                    pct_diff = model_ret * 100
                pct_differences.append(pct_diff)

            pct_diff_array: np.ndarray = np.array(pct_differences)

            stats: dict = {
                "returns": model_returns,
                "lengths": model_lengths,
                "natural_terminations": [False]
                * len(model_returns),  # Not tracked
                "pct_differences": pct_differences,
                "mean_return": float(np.mean(model_returns_array)),
                "std_return": float(np.std(model_returns_array)),
                "median_return": float(np.median(model_returns_array)),
                "min_return": float(np.min(model_returns_array)),
                "max_return": float(np.max(model_returns_array)),
                "q25_return": float(np.percentile(model_returns_array, 25)),
                "q75_return": float(np.percentile(model_returns_array, 75)),
                "mean_length": 0.0,  # Not tracked
                "success_rate": 0.0,  # Not tracked
                "mean_pct_diff": final_entry["mean_pct_diff"],
                "std_pct_diff": final_entry["std_pct_diff"],
                "median_pct_diff": float(np.median(pct_diff_array)),
            }

            model_stats[method_name] = stats

            # Compute comparison
            comparison: dict = compare_returns(human_stats, stats, method_name)
            comparisons[method_name] = comparison

            print(
                f"    Mean % diff: {stats['mean_pct_diff']:+.2f}% ± {stats['std_pct_diff']:.2f}%"
            )

        # Build eval_results dict
        eval_results: dict = {
            "env_name": env_name,
            "subject": args.subject,
            "human_stats": human_stats,
            "model_stats": model_stats,
            "comparisons": comparisons,
            "success_threshold": 0.0,  # Not used in plots
            "num_eval_episodes": len(test_episode_info),
            "metadata": metadata,
        }

        # Create all visualizations
        print(f"\n{'='*60}")
        print("GENERATING PLOTS")
        print(f"{'='*60}")
        create_all_plots(eval_results)

        # Print comparison table
        create_comparison_table(eval_results)

        print(f"\n{'='*60}")
        print("PLOTTING COMPLETE")
        print(f"{'='*60}")
        return

    # Check method
    if not args.method:
        print(
            "Error: --method is required unless using --list-methods or --plot"
        )
        return

    if args.method not in method_dict:
        print(f"Error: Unknown method '{args.method}'")
        print("Use --list-methods to see available options")
        return

    exp_config: ExperimentConfig = ExperimentConfig(seed=args.seed)

    # Set random seeds for reproducibility
    set_random_seeds(exp_config.seed)
    print(f"Random seed: {exp_config.seed}")
    print(f"Subject: {args.subject}")
    print(f"Continual learning info: {args.use_cl_info}")

    # Load data
    print(f"\nLoading {env_config['name']} human behavior data...")
    optim_obs, optim_act, test_obs, test_act, metadata = load_human_data(
        env_name, args.use_cl_info, args.subject, exp_config.holdout_pct
    )

    # Save metadata for potential evaluation use
    metadata_path: Path = (
        RESULTS_DIR / f"metadata_{env_name}_{args.subject}.json"
    )
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, cls=NumpyEncoder)
    print(f"  Metadata saved to {metadata_path}")

    # Determine actual input size (may include CL features)
    input_size: int = optim_obs.shape[1]
    output_size: int = action_dim

    # Run single method
    run_single_method(
        env_name,
        args.method,
        method_dict[args.method],
        args.use_cl_info,
        optim_obs,
        optim_act,
        test_obs,
        test_act,
        input_size,
        output_size,
        exp_config,
        args.subject,
    )

    print("\n" + "=" * 60)
    print(f"{args.method} Complete!")
    print("=" * 60)
    print(f"Results saved to {RESULTS_DIR}/")
    print(
        f"To evaluate optimized models, run: --dataset {env_name} --evaluate"
    )


if __name__ == "__main__":
    exit_code: int = 0
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        # Ensure cleanup even on error
        plt.close("all")
        torch.cuda.empty_cache()
        sys.exit(exit_code)

"""Main script for Experiment 3: Continual Learning Information Ablation."""

import argparse
import sys

import matplotlib.pyplot as plt
import torch
from jaxtyping import Float, Int
from torch import Tensor

from config import (
    DEVICE,
    ENV_CONFIGS,
    RESULTS_DIR,
    SCRIPT_DIR,
    ExperimentConfig,
    set_device,
)
from data import load_human_data
from plotting import update_plot
from training import train_deep_learning, train_neuroevolution
from utils import set_random_seeds


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
    train_obs: Float[Tensor, "train_size input_size"],
    train_act: Int[Tensor, " train_size"],
    test_obs: Float[Tensor, "test_size input_size"],
    test_act: Int[Tensor, " test_size"],
    input_size: int,
    output_size: int,
    config: ExperimentConfig,
) -> None:
    """Run a single optimization method."""
    # Append CL variant to method name
    cl_suffix: str = "with_cl" if use_cl_info else "no_cl"
    method_name_full: str = f"{method_name}_{cl_suffix}"

    print(f"\n{'='*60}")
    print(f"Running {method_name_full} for {env_name}")
    print(f"{'='*60}")
    print(f"Train size: {train_obs.shape[0]}, Test size: {test_obs.shape[0]}")
    print(f"Input size: {input_size}, Output size: {output_size}")

    if method_config["type"] == "dl":
        train_deep_learning(
            train_obs,
            train_act,
            test_obs,
            test_act,
            input_size,
            output_size,
            config,
            env_name,
            method_name_full,
        )
    else:
        train_neuroevolution(
            train_obs,
            train_act,
            test_obs,
            test_act,
            input_size,
            output_size,
            config,
            env_name,
            method_name_full,
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
        "--plot-only",
        action="store_true",
        help="Only update the plot with existing results",
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

    args = parser.parse_args()

    # Set global DEVICE based on --gpu argument
    set_device(args.gpu)
    print(f"Using device: {DEVICE}")

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

    # Plot-only mode
    if args.plot_only:
        print(f"Updating plot for {env_name}...")
        update_plot(env_name, interactive=True)
        plt.ioff()
        plt.show()
        return

    # Check method
    if not args.method:
        print("Error: --method is required unless using --list-methods or --plot-only")
        return

    if args.method not in method_dict:
        print(f"Error: Unknown method '{args.method}'")
        print("Use --list-methods to see available options")
        return

    config: ExperimentConfig = ExperimentConfig(seed=args.seed)

    # Set random seeds for reproducibility
    set_random_seeds(config.seed)
    print(f"Random seed: {config.seed}")
    print(f"Continual learning info: {args.use_cl_info}")

    # Load data
    print(f"\nLoading {env_config['name']} human behavior data...")
    train_obs, train_act, test_obs, test_act = load_human_data(
        env_name, args.use_cl_info
    )

    # Determine actual input size (may include CL features)
    input_size: int = train_obs.shape[1]
    output_size: int = action_dim

    # Run single method
    run_single_method(
        env_name,
        args.method,
        method_dict[args.method],
        args.use_cl_info,
        train_obs,
        train_act,
        test_obs,
        test_act,
        input_size,
        output_size,
        config,
    )

    print("\n" + "=" * 60)
    print(f"{args.method} Complete!")
    print("=" * 60)
    print(f"Results saved to {RESULTS_DIR}/")
    plot_path = SCRIPT_DIR / f"{env_name}.png"
    print(f"Plot saved to {plot_path}")


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

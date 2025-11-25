import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# List of files to process
filenames = [
    "sub01_data_acrobot.json",
    "sub01_data_cartpole.json",
    "sub01_data_lunarlander.json",
    "sub01_data_mountaincar.json",
    "sub02_data_acrobot.json",
    "sub02_data_cartpole.json",
    "sub02_data_lunarlander.json",
    "sub02_data_mountaincar.json",
]


def parse_analyze_and_plot(filenames):
    # Setup for printing the table
    print(
        f"{'File Name':<30} | {'Total Steps':<12} | {'Sessions':<8} | {'Runs per Session'}"
    )
    print("-" * 100)

    # Setup for plotting (4x2 grid)
    fig, axes = plt.subplots(4, 2, figsize=(20, 24))
    axes = axes.flatten()

    for idx, filename in enumerate(filenames):
        try:
            with open(filename, "r") as f:
                data = json.load(f)

            episodes_data = []
            total_steps = 0

            if isinstance(data, list):
                for ep in data:
                    # Count observations/steps (State-Action Pairs)
                    if "steps" in ep and isinstance(ep["steps"], list):
                        step_count = sum(
                            1 for s in ep["steps"] if "observation" in s
                        )
                        total_steps += step_count

                        if "timestamp" in ep:
                            ts = datetime.fromisoformat(ep["timestamp"])
                            total_return = sum(
                                step.get("reward", 0) for step in ep["steps"]
                            )
                            length = len(ep["steps"])

                            episodes_data.append(
                                {
                                    "timestamp": ts,
                                    "return": total_return,
                                    "length": length,
                                }
                            )

            # Sort episodes chronologically
            episodes_data.sort(key=lambda x: x["timestamp"])

            # --- Session Analysis ---
            sessions_counts = []
            session_boundaries = []  # Indices for plotting vertical lines

            if episodes_data:
                current_session_count = 1
                last_time = episodes_data[0]["timestamp"]

                for i in range(1, len(episodes_data)):
                    curr_time = episodes_data[i]["timestamp"]
                    # Check for 30-minute gap
                    if (curr_time - last_time) > timedelta(minutes=30):
                        sessions_counts.append(current_session_count)
                        current_session_count = 1
                        session_boundaries.append(
                            i
                        )  # Mark start of new session
                    else:
                        current_session_count += 1
                    last_time = curr_time
                # Append the final session
                sessions_counts.append(current_session_count)
            else:
                sessions_counts = [0]

            # Print stats to terminal
            print(
                f"{filename:<30} | {total_steps:<12} | {len(sessions_counts):<8} | {sessions_counts}"
            )

            # --- Plotting ---
            returns = [x["return"] for x in episodes_data]
            lengths = [x["length"] for x in episodes_data]
            indices = range(len(episodes_data))

            ax1 = axes[idx]

            # Plot 1: Total Return (Left Y-Axis)
            color_ret = "tab:blue"
            ax1.set_xlabel("Episode Number")
            ax1.set_ylabel("Total Return", color=color_ret, fontweight="bold")
            ax1.scatter(
                indices,
                returns,
                color=color_ret,
                alpha=0.6,
                s=15,
                label="Return",
            )
            ax1.tick_params(axis="y", labelcolor=color_ret)

            # Plot 2: Episode Length (Right Y-Axis)
            ax2 = ax1.twinx()
            color_len = "tab:orange"
            ax2.set_ylabel(
                "Episode Length", color=color_len, fontweight="bold"
            )
            ax2.scatter(
                indices,
                lengths,
                color=color_len,
                alpha=0.5,
                marker="x",
                s=15,
                label="Length",
            )
            ax2.tick_params(axis="y", labelcolor=color_len)

            # Add Session Delimiters
            for boundary in session_boundaries:
                ax1.axvline(
                    x=boundary - 0.5,
                    color="gray",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                )

            ax1.set_title(f"File: {filename}", fontsize=12, fontweight="bold")
            ax1.grid(True, which="both", linestyle=":", alpha=0.4)

        except Exception as e:
            print(f"{filename:<30} | Error: {str(e)}")
            axes[idx].text(
                0.5, 0.5, f"Error processing file:\n{e}", ha="center"
            )

    plt.tight_layout()
    output_filename = "training_sessions_plot.png"
    plt.savefig(output_filename, dpi=150)
    print(f"\nPlot saved to {output_filename}")
    plt.show()


if __name__ == "__main__":
    parse_analyze_and_plot(filenames)

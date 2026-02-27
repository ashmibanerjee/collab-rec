import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from experiments.helpers import get_pop_level
from dotenv import load_dotenv
from matplotlib.lines import Line2D
from typing import List

load_dotenv()
from analysis.plot_utils import save_plots, set_paper_style

PAPER_LOCATION = os.getenv("PAPER_LOCATION")

from constants import AGENT_COLORS, POP_LEVEL_COLORS


def _create_agent_legend(agent_names: List[str], ax=None, position='auto'):
    """
    Create a clean legend showing only agent roles with correct colors.

    Args:
        agent_names: List of agent names to include in legend
        ax: Matplotlib axes object (optional)
        position: 'auto' for automatic placement, 'upper_right' for explicit positioning

    Returns:
        Legend object
    """
    legend_handles = [
        Line2D(
            [0], [0],
            color=AGENT_COLORS[agent],
            lw=3.0,  # Match the line width in plots
            marker="o",
            markersize=10,  # Match the marker size in plots
            linestyle="-",
            label=agent.title(),
        )
        for agent in agent_names if agent in AGENT_COLORS
    ]

    if position == 'upper_right':
        legend = ax.legend(
            handles=legend_handles,
            title="Agent Role",
            loc="center right",
            bbox_to_anchor=(1.0, 0.60),  # Upper right, above pop level legend
            frameon=False
        )
    else:
        legend = ax.legend(
            handles=legend_handles,
            title="Agent Role",
            loc="best",
            frameon=False
        )
    return legend


def _create_pop_level_legend(pop_levels: List[str], markers: List[str] = None, ax=None, position='bottom'):
    """
    Create a legend showing popularity levels with different markers as a horizontal row.

    Args:
        pop_levels: List of popularity levels to include in legend
        markers: List of marker styles corresponding to pop_levels
        ax: Matplotlib axes object (optional)
        position: 'bottom' for below plot, 'side' for placed inline near the agent legend on the right,
                  'inline' for centered inline placement (used to match screenshot)

    Returns:
        Legend object
    """
    if markers is None:
        markers = ["o", "s", "^"]

    # Map pop levels to markers
    pop_marker_map = dict(zip(["low", "medium", "high"], markers))

    legend_handles = [
        Line2D(
            [0], [0],
            color="gray",
            lw=0,
            marker=pop_marker_map.get(pop, "o"),
            markersize=10,  # Increased from 8 to match plot settings
            linestyle="",
            label=pop.capitalize()
        )
        for pop in pop_levels if pop in pop_marker_map
    ]

    # Position legend based on parameter
    if position == 'side':
        # Place the popularity-legend inline near the center-left area to match screenshot
        legend = ax.legend(
            handles=legend_handles,
            title="Popularity Level",
            loc="center",
            bbox_to_anchor=(0.45, 0.62),  # adjusted to center-left, slightly above mid-height
            ncol=len(legend_handles),  # Horizontal layout
            frameon=True,
            columnspacing=1.5,
            edgecolor='black',
            fancybox=False
        )
    elif position == 'inline':
        # Alternate inline option (same as 'side' but explicit)
        legend = ax.legend(
            handles=legend_handles,
            title="Popularity Level",
            loc="center",
            bbox_to_anchor=(0.35, 0.57),
            ncol=len(legend_handles),
            frameon=True,
            columnspacing=1.5,
            edgecolor='black',
            fancybox=False
        )
    else:
        # Create horizontal legend below the plot
        legend = ax.legend(
            handles=legend_handles,
            title="Popularity Level",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=len(legend_handles),
            frameon=True,
            columnspacing=1.5,
            edgecolor='black',
            fancybox=False
        )
    legend.get_frame().set_linewidth(1.0)
    return legend


def _save_plot_with_title(file_name: str, title: str, subfolder: str,
                          paper_location: str = None, copy_to_paper: bool = True,
                          agent_legend=None, pop_legend=None, ax=None) -> None:
    """
    Save plot in both PDF (without title) and PNG (with title) formats.

    Args:
        file_name: Base filename for saving
        title: Title to add to PNG version
        subfolder: Subfolder to save plots in
        paper_location: Location to copy paper plots to
        copy_to_paper: Whether to copy PDF to paper location
        agent_legend: Agent legend object (to hide for PDF if needed)
        pop_legend: Pop level legend object (to hide for PDF if needed)
        ax: Matplotlib axes object
    """
    # Save PDF without title
    save_plots(file_name, subfolder=subfolder,
               extensions=["pdf"],
               paper_location=paper_location,
               copy_to_paper=copy_to_paper)

    # Restore legends for PNG if they were hidden
    if agent_legend is not None and ax is not None:
        agent_legend.set_visible(True)
    if pop_legend is not None:
        pop_legend.set_visible(True)

    # Save PNG with title
    plt.title(title)
    save_plots(file_name, subfolder=subfolder,
               extensions=["png"],
               copy_to_paper=False)


def plot_agent_behavior(df: pd.DataFrame, model_name: str, rounds: int=10):
    """
    Plots agent behavior over rounds.
    Creates separate plots for each rejection strategy, showing all popularity levels combined.
    """
    df["pop_level"] = df["query_id"].apply(get_pop_level)
    set_paper_style()

    rejection_strategies = sorted(df["rejection_strategy"].unique())

    for strategy in rejection_strategies:
        strategy_df = df[df["rejection_strategy"] == strategy]

        # Aggregate by round, agent, and pop_level
        avg_scores = (
            strategy_df
            .groupby(["round_nr", "agent_name", "pop_level"], as_index=False)
            .agg(avg_score=("success_score", "mean"))
        )

        # Agent names (used by legends and std shading)
        agent_names = sorted(avg_scores["agent_name"].unique())

        # Create figure
        plt.figure(figsize=(8, 5))  # Increased from (7, 4.5) for better legibility
        ax = sns.lineplot(
            data=avg_scores,
            x="round_nr",
            y="avg_score",
            hue="agent_name",
            style="pop_level",
            style_order=["low", "medium", "high"],
            markers=["o", "s", "^"],  # Different markers for each pop level
            dashes=False,
            palette=AGENT_COLORS,
        )

        plt.xlabel("Round Number")
        plt.ylabel("Avg Success Score")
        ax.legend_.remove()

        # Mark convergence point at round 5
        ax.axvline(x=5, color='black', linestyle='--', linewidth=1.2)
        # Add label for convergence
        ylim = ax.get_ylim()
        ax.text(5 + 0.1, ylim[1] - 0.02 * (ylim[1] - ylim[0]), '', rotation=90,
                va='top', ha='left', fontsize=12)

        # Determine legend visibility based on model
        show_agent_legend_in_pdf = model_name.lower() in ["gemini", "olmo", "olmo-7b"]
        show_pop_legend_in_pdf = model_name.lower() in ["gpt", "gemma-4b"]

        # Create agent legend (positioned on right for gemini/olmo)
        if show_agent_legend_in_pdf:
            agent_legend = _create_agent_legend(agent_names, ax=ax, position='upper_right')
        else:
            agent_legend = _create_agent_legend(agent_names, ax=ax, position='auto')

        # Add the agent legend back to the plot (so it's present for PNGs)
        ax.add_artist(agent_legend)

        # Create pop-level legend: inline for selected PDF models, below otherwise
        pop_levels = sorted(avg_scores["pop_level"].unique())
        if show_pop_legend_in_pdf:
            pop_legend = _create_pop_level_legend(pop_levels, markers=["o", "s", "^"], ax=ax, position='side')
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)
        else:
            pop_legend = _create_pop_level_legend(pop_levels, markers=["o", "s", "^"], ax=ax, position='bottom')
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.25)

        # Before saving PDF: only show the requested legends for PDF (others hidden)
        agent_legend.set_visible(bool(show_agent_legend_in_pdf))
        pop_legend.set_visible(bool(show_pop_legend_in_pdf))

        # Save plot
        if rounds == 20:
            file_name = f"{model_name}_success_{strategy}_20_rounds"
        else:
            file_name = f"{model_name}_success_{strategy}_all_pop"
        _save_plot_with_title(
            file_name=file_name,
            title=f"{model_name.title()} Agent Success Over Rounds ({strategy.title()} Strategy)",
            subfolder="agent_success",
            paper_location=PAPER_LOCATION,
            copy_to_paper=True,
            agent_legend=agent_legend,
            pop_legend=pop_legend,
            ax=ax
        )
        plt.close()


def _compute_across_models_stats(models: List[str]) -> dict:
    """Compute mean and std of success_score across models for each round and agent.

    Returns a dict keyed by rejection_strategy with DataFrame columns: round_nr, agent_name, mean_score, std_score
    """
    frames = []
    for model in models:
        for strat in ["aggressive", "majority"]:
            file_path = f"../data/collab-rec-2026/analysis/{model}_mami_{strat}_scores_with_relevance.csv"
            try:
                d = pd.read_csv(file_path)
            except FileNotFoundError:
                continue
            # Ensure columns exist
            if {'round_nr', 'agent_name', 'success_score'}.issubset(d.columns):
                d = d[['round_nr', 'agent_name', 'success_score']].copy()
                d['model'] = model
                d['rejection_strategy'] = strat
                frames.append(d)
    if not frames:
        return {}
    df_all = pd.concat(frames, ignore_index=True)

    stats = {}
    for strat in df_all['rejection_strategy'].unique():
        df_strat = df_all[df_all['rejection_strategy'] == strat]
        grouped = (
            df_strat
            .groupby(['round_nr', 'agent_name'], as_index=False)
            .agg(mean_score=('success_score', 'mean'), std_score=('success_score', 'std'))
        )
        stats[strat] = grouped
    return stats


def main(model_name: str, rounds: int = 20):
    """
    Main function to compute scores and generate plots.

    Args:
        model_name: Model name (e.g., "gemini")
        rounds: Number of rounds to use for MAMI scores (20 for all rounds, 10 for 10 rounds)
    """
    if rounds == 10:
        aggressive_file = f"../data/collab-rec-2026/analysis/{model_name}_mami_aggressive_scores_with_relevance.csv"
        majority_file = f"../data/collab-rec-2026/analysis/{model_name}_mami_majority_scores_with_relevance.csv"
    else:
        aggressive_file = f"../data/collab-rec-2026/analysis/{model_name}_mami_aggressive_20_rounds_scores.csv"
        majority_file = f"../data/collab-rec-2026/analysis/{model_name}_mami_majority_20_rounds_scores.csv"
    df_aggressive = pd.DataFrame()
    df_majority = pd.DataFrame()
    try:
        df_aggressive = pd.read_csv(aggressive_file)
        df_aggressive['rejection_strategy'] = 'aggressive'
        print(f"Loaded {len(df_aggressive)} rows from {aggressive_file}.")
    except FileNotFoundError:
        print(f"No scores found for {model_name} at {aggressive_file}. Skipping plot generation.")

    try:
        df_majority = pd.read_csv(majority_file)
        df_majority['rejection_strategy'] = 'majority'
        print(f"Loaded {len(df_majority)} rows from {majority_file}.")
    except FileNotFoundError:
        print(f"No scores found for {model_name} at {majority_file}. Skipping plot generation.")

    df = pd.concat([df_aggressive, df_majority], ignore_index=True)

    set_paper_style()
    # Plot all metrics
    print(f"\nGenerating plots...")
    plot_agent_behavior(df, model_name=model_name, rounds=rounds)
    print("Plots saved successfully!")


# MODELS = ["claude", "gemini", "gpt", "gemma-12b", "olmo-7b", "gemma-4b"]
MODELS = ["claude", "gemma-4b", "olmo-7b"]  # models with 20 rounds data
# MODELS=["smol-3b"]
if __name__ == "__main__":
    for model in MODELS:
        # Generate plots split by popularity level
        main(model_name=model, rounds=20)

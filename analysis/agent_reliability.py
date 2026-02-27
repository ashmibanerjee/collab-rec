import os

import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from matplotlib import pyplot as plt

from analysis.agent_behavior_plots import _create_agent_legend
from analysis.plot_utils import set_paper_style, save_plots
from constants import AGENT_COLORS
from experiments.helpers import get_pop_level

load_dotenv()


def _plot_metric_by_agent_and_round(df: pd.DataFrame, metric_col: str, metric_name: str,
                                    ylabel: str, model_name: str = "gemini"):
    """
    Helper function to plot a metric by agent and round with consistent styling.
    Plots BOTH rejection strategies in a single plot per model: aggressive as solid, majority as dashed.
    Only plots the following agents for the main lines: popularity, sustainability, personalization.
    Additionally plots the moderator's success (averaged across strategies) as a dotted purple line.
    """
    df["pop_level"] = df["query_id"].apply(get_pop_level)
    set_paper_style()

    plot_df = df.copy()

    # For reliability and hallucination, add synthetic round 1 data
    if metric_col == "reliability_score":
        round_1_data = df[df['round_nr'] == 1].copy()
        round_1_data['reliability_score'] = 1.0
        plot_df = pd.concat([round_1_data, df[df['round_nr'] > 1]], ignore_index=True)
    elif metric_col == "hallucination_rate":
        round_1_data = df[df['round_nr'] == 1].copy()
        round_1_data['hallucination_rate'] = 0.0
        plot_df = pd.concat([round_1_data, df[df['round_nr'] > 1]], ignore_index=True)

    # Aggregate across pop_levels, keep rejection_strategy so both appear on same plot
    avg_scores = (
        plot_df
        .groupby(["round_nr", "agent_name", "rejection_strategy"], as_index=False)
        .agg(avg_metric=(metric_col, "mean"))
    )

    # Which agents to plot as main lines
    target_agents = ["popularity", "sustainability", "personalization"]
    avg_scores_sub = avg_scores[avg_scores['agent_name'].isin(target_agents)].copy()

    # Create figure
    plt.figure(figsize=(8, 5))
    # aggressive: solid, majority: dashed
    style_order = ["aggressive", "majority"]
    dashes = [(None, None), (2, 2)]  # aggressive solid, majority dashed

    ax = sns.lineplot(
        data=avg_scores_sub,
        x="round_nr",
        y="avg_metric",
        hue="agent_name",
        style="rejection_strategy",
        style_order=style_order,
        dashes=dashes,
        markers=True,
        marker="o",
        palette=AGENT_COLORS,
    )

    plt.xlabel("Round Number")
    plt.ylabel(ylabel)
    ax.legend_.remove()
    # Convergence vertical line at round 4 (black dashed)
    ax.axvline(x=3, color='black', linestyle='--', linewidth=1.5)

    # Agent legend handling: show in PDF only for specific models
    show_agent_legend_in_pdf = model_name.lower() in ["gemini", "olmo-7b"]

    # Build legend handles using only the three main agents + moderator (if present)
    agent_names_for_legend = [a for a in target_agents if a in avg_scores['agent_name'].unique()]
    # if not mod_df.empty:
    #     agent_names_for_legend.append('moderator')

    agent_legend = _create_agent_legend(agent_names_for_legend, ax=ax,
                                        position='lower_right' if show_agent_legend_in_pdf else 'auto')
    ax.add_artist(agent_legend)

    # Before saving PDF: set visibility per rules
    agent_legend.set_visible(bool(show_agent_legend_in_pdf))

    # Save PDF (without title)
    file_name = f"{model_name}_{metric_col}_combined"
    save_plots(file_name, subfolder="agent_reliability", extensions=["pdf"], copy_to_paper=True,
               paper_location=os.getenv("PAPER_LOCATION"))

    # Restore legend for PNG and save with title
    agent_legend.set_visible(True)
    plt.title(f"{model_name.title()} {metric_name} Over Rounds (Both Strategies)")
    save_plots(file_name, subfolder="agent_reliability", extensions=["png"], copy_to_paper=False)

    plt.close()


def plot_agent_metrics(df: pd.DataFrame, model_name: str):
    """
    Plot reliability scores and hallucination rates per agent, per round.
    Creates combined plots showing all popularity levels together.

    Args:
        df: DataFrame with columns including agent_name, round_nr, reliability_score, hallucination_rate
        model_name: Model name for filename
    """
    _plot_metric_by_agent_and_round(
        df=df,
        metric_col="reliability_score",
        metric_name="Agent Reliability Score",
        ylabel="Average Reliability Score",
        model_name=model_name
    )


def main(model_name: str):
    """
    Main function to compute scores and generate plots.

    Args:
        model_name: Model name (e.g., "gemini")
    """
    aggressive_file = f"../data/collab-rec-2026/analysis/{model_name}_mami_aggressive_scores_with_relevance.csv"
    majority_file = f"../data/collab-rec-2026/analysis/{model_name}_mami_majority_scores_with_relevance.csv"

    df_aggressive = pd.read_csv(aggressive_file)
    df_majority = pd.read_csv(majority_file)

    df_aggressive['rejection_strategy'] = 'aggressive'
    df_majority['rejection_strategy'] = 'majority'

    df = pd.concat([df_aggressive, df_majority], ignore_index=True)

    set_paper_style()
    # Plot all metrics
    print(f"\nGenerating plots...")
    plot_agent_metrics(df, model_name=model_name)
    # plot_agent_metrics(df=df, model_name=model_name)
    print("Plots saved successfully!")


MODELS = ["claude", "gemini", "gpt", "gemma-12b", "olmo-7b", "gemma-4b"]
# MODELS=["smol-3b"]
if __name__ == "__main__":
    for model in MODELS:
        # Generate plots split by popularity level
        main(model_name=model)

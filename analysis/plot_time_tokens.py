"""
Plot time and token complexity metrics for the moderator agent across rounds for all models.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from dotenv import load_dotenv

from analysis.plot_utils import save_plots, set_paper_style
load_dotenv()
# List of models to include in the plot
MODELS = ["claude", "gemini", "gpt", "gemma-12b", "olmo-7b", "gemma-4b"]
REJECTION_STRATEGIES = ["aggressive", "majority"]


def load_all_model_data(rejection_strategy: str):
    """
    Load data for all models for a given rejection strategy.
    """
    all_data = []
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "collab-rec-2026" / "analysis"

    for model in MODELS:
        # Prefer scores_with_relevance if available, otherwise just scores
        csv_path = data_dir / f"{model}_mami_{rejection_strategy}_scores_with_relevance.csv"
        if not csv_path.exists():
            csv_path = data_dir / f"{model}_mami_{rejection_strategy}_scores.csv"

        if csv_path.exists():
            print(f"Loading data for {model} from: {csv_path.name}")
            df = pd.read_csv(csv_path)
            # Filter for moderator only
            moderator_df = df[df['agent_name'] == 'moderator'].copy()
            moderator_df['model_name'] = model
            all_data.append(moderator_df)
        else:
            print(f"Warning: No data found for model {model} and strategy {rejection_strategy}")

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


def plot_combined_complexity(df_aggressive: pd.DataFrame, df_majority: pd.DataFrame, metric: str):
    """
    Plot average metric per round for all models combining both rejection strategies.
    Solid lines for aggressive, dashed lines for majority.

    Args:
        df_aggressive: DataFrame containing all models' moderator data for aggressive strategy
        df_majority: DataFrame containing all models' moderator data for majority strategy
        metric: 'time_taken' or 'total_token_count'
    """
    set_paper_style()
    if df_aggressive.empty and df_majority.empty:
        print(f"No data to plot for {metric}")
        return

    plt.figure(figsize=(8, 6))

    # Use a distinct color palette and markers for legibility
    markers = ['o', 's', '^', 'D', 'v', 'p']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, model in enumerate(MODELS):
        # Plot aggressive strategy (solid line)
        if not df_aggressive.empty:
            agg_df_aggressive = (
                df_aggressive[df_aggressive['model_name'] == model]
                .groupby(['round_nr'], as_index=False)
                .agg(avg_metric=(metric, 'mean'))
                .sort_values('round_nr')
            )
            if not agg_df_aggressive.empty:
                plt.plot(
                    agg_df_aggressive['round_nr'],
                    agg_df_aggressive['avg_metric'],
                    marker=markers[i % len(markers)],
                    color=colors[i % len(colors)],
                    label=model.upper(),
                    linewidth=3,
                    markersize=10,
                    markeredgecolor='white',
                    markeredgewidth=1,
                    linestyle='-'
                )

        # Plot majority strategy (dashed line)
        if not df_majority.empty:
            agg_df_majority = (
                df_majority[df_majority['model_name'] == model]
                .groupby(['round_nr'], as_index=False)
                .agg(avg_metric=(metric, 'mean'))
                .sort_values('round_nr')
            )
            if not agg_df_majority.empty:
                plt.plot(
                    agg_df_majority['round_nr'],
                    agg_df_majority['avg_metric'],
                    marker=markers[i % len(markers)],
                    color=colors[i % len(colors)],
                    linewidth=3,
                    markersize=10,
                    markeredgecolor='white',
                    markeredgewidth=1,
                    linestyle='--'
                )

    ylabel = "Avg. Time (s)" if metric == 'time_taken' else "Avg. Token Count"
    
    plt.xlabel("Round Number")
    plt.ylabel(ylabel)
    plt.xticks(range(1, 11))  # Focus on 1-10 as requested
    plt.xlim(0.8, 10.2)
    
    # Adjust y-axis to start from 0 for fair comparison
    plt.ylim(bottom=0)

    # Legend only shows model names, not rejection strategies
    plt.legend(loc='best', frameon=True, framealpha=0.9, edgecolor='gray')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save plots
    paper_loc = os.getenv("PAPER_LOCATION")
    copy_to_paper = paper_loc is not None
    
    file_prefix = "avg_time" if metric == 'time_taken' else "avg_tokens"
    file_name = f"{file_prefix}_combined_all_models"

    save_plots(
        file_name, 
        subfolder="complexity", 
        copy_to_paper=copy_to_paper, 
        paper_location=paper_loc
    )
    plt.close()


def main():
    """
    Generate combined complexity plots for all models combining both strategies.
    """
    print("\nLoading data for both strategies...")

    # Load data for both strategies
    df_aggressive = load_all_model_data("aggressive")
    df_aggressive = df_aggressive.loc[df_aggressive["agent_name"]=="moderator"]

    df_majority = load_all_model_data("majority")
    df_majority = df_majority.loc[df_majority["agent_name"]=="moderator"]

    if df_aggressive.empty and df_majority.empty:
        print("Error: No data found for either strategy")
        return

    # Check required columns
    required_cols = ['round_nr', 'time_taken', 'total_token_count']

    for df, name in [(df_aggressive, "aggressive"), (df_majority, "majority")]:
        if not df.empty:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Error: Missing columns {missing_cols} in {name} data")
                return

    print("Generating combined plots...")
    # Generate 2 plots: one for time, one for tokens
    plot_combined_complexity(df_aggressive, df_majority, 'time_taken')
    plot_combined_complexity(df_aggressive, df_majority, 'total_token_count')

    print("\nAll plots generated and saved.")


if __name__ == "__main__":
    main()

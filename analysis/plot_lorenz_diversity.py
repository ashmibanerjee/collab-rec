import os

import pandas as pd
from dotenv import load_dotenv
from matplotlib import pyplot as plt
import numpy as np

from analysis.plot_city_distri import make_plot_configs, get_recommendations_for_config, prepare_city_plot_data
from constants import CITIES
from analysis.plot_utils import save_plots, set_paper_style

load_dotenv()

PAPER_LOCATION = os.getenv("PAPER_LOCATION")


def plot_lorenz_curve(model_name, method_data_dict, rejection_strategy):
    """
    Plot Lorenz curve for city distribution across multiple methods.

    Args:
        model_name: Name of the model
        method_data_dict: Dictionary mapping method names to their city count DataFrames
    """
    set_paper_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    # Define colors for each method
    colors = {
        'SASI': '#1f77b4',
        'MASI': '#ff7f0e',
        'MAMI': '#2ca02c'
    }

    # Define line styles
    linestyles = {
        'SASI': '--',
        'MASI': '-.',
        'MAMI': '-'
    }

    for method, df in method_data_dict.items():
        if df.empty:
            print(f"Warning: Empty dataframe for {model_name} - {method}")
            continue

        # Sort cities by count (descending)
        sorted_df = df.sort_values('count', ascending=False).copy()

        # Calculate cumulative sums
        cumulative_recommendations = sorted_df['count'].cumsum()
        total_recommendations = sorted_df['count'].sum()

        # Calculate cumulative percentages
        cumulative_pct_recommendations = cumulative_recommendations / total_recommendations * 100
        cumulative_pct_cities = np.arange(1, len(sorted_df) + 1) / len(sorted_df) * 100

        # Plot the Lorenz curve
        ax.plot(
            cumulative_pct_cities,
            cumulative_pct_recommendations,
            label=method,
            color=colors.get(method, '#333333'),
            linestyle=linestyles.get(method, '-'),
            linewidth=3.5,
            marker='o',
            markersize=0,  # No markers for cleaner look
            markevery=10
        )

    # Plot the line of perfect equality (diagonal)
    ax.plot([0, 100], [0, 100], 'k--', linewidth=2.5, alpha=0.5, label='Perfect Equality')

    # Set labels and title
    ax.set_xlabel('Cumulative % of Cities', fontsize=28, fontweight='bold')
    ax.set_ylabel('Recommendations (%)', fontsize=26, fontweight='bold')

    # Set axis limits
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    if model_name in ["gpt", "gemma-4b"]:
        legend = ax.legend(
            loc='best'
        )
        legend.set_title('')

    # Set tick parameters
    ax.tick_params(axis='both', labelsize=24)

    plt.tight_layout()

    # Save the plot
    file_name = f'{model_name}_lorenz_curve_{rejection_strategy}'
    save_plots(
        file_name=file_name,
        subfolder="lorenz",
        extensions=["pdf"],
        copy_to_paper=True,
        paper_location=PAPER_LOCATION
    )
    ax.legend()
    ax.set_title(f'Lorenz Curve: {model_name.upper()}', fontsize=30, fontweight='bold', pad=20)
    save_plots(
        file_name=file_name,
        subfolder="lorenz",
        extensions=["png"],
        copy_to_paper=False
    )
    plt.close()


def plot_heatmap(model_name, method_data_dict):
    """
    Create a grid heatmap showing recommendation frequency for all cities across methods.

    Args:
        model_name: Name of the model
        method_data_dict: Dictionary mapping method names to their city count DataFrames
    """
    from matplotlib.colors import LogNorm

    set_paper_style()

    # Get all 200 cities from constants
    all_cities = CITIES

    # Load KB to get popularity information
    kb = pd.read_csv("../data/collab-rec-2026/input-data/kb/merged_listing.csv")
    kb_cities = kb.drop_duplicates(subset=['city'])[['city', 'popularity']].copy()

    # Create a single figure with 3 subplots (one for each method)
    fig, axes = plt.subplots(1, 3, figsize=(24, 10))

    methods = ['SASI', 'MASI', 'MAMI']

    for idx, method in enumerate(methods):
        ax = axes[idx]
        df = method_data_dict.get(method, pd.DataFrame())

        # Create a complete city dataframe with all 200 cities
        city_counts = pd.DataFrame({'city': all_cities})

        # Merge with actual counts (cities not in df will have NaN, which we'll fill with 0)
        if not df.empty:
            city_counts = city_counts.merge(
                df[['city', 'count']],
                on='city',
                how='left'
            )
        else:
            city_counts['count'] = 0

        city_counts['count'] = city_counts['count'].fillna(0).astype(int)

        # Merge with popularity
        city_counts = city_counts.merge(kb_cities, on='city', how='left')

        # Sort by popularity (High -> Medium -> Low) then by count descending
        popularity_order = {'High': 0, 'Medium': 1, 'Low': 2}
        city_counts['pop_rank'] = city_counts['popularity'].map(popularity_order).fillna(3)
        city_counts = city_counts.sort_values(['pop_rank', 'count'], ascending=[True, False])

        # Create a 10x20 grid (200 cities)
        n_rows = 20
        n_cols = 10

        # Reshape counts into grid
        counts_array = city_counts['count'].values
        heatmap_data = counts_array.reshape(n_rows, n_cols)

        # Use log scale for better visualization if there's a large range
        max_count = counts_array.max()
        min_count = counts_array[counts_array > 0].min() if (counts_array > 0).any() else 1

        # Create heatmap
        if max_count > 0:
            # Use log scale if range is large
            if max_count / max(min_count, 1) > 100:
                im = ax.imshow(
                    heatmap_data,
                    cmap='YlOrRd',
                    aspect='auto',
                    norm=LogNorm(vmin=max(min_count, 0.1), vmax=max_count)
                )
            else:
                im = ax.imshow(
                    heatmap_data,
                    cmap='YlOrRd',
                    aspect='auto',
                    vmin=0,
                    vmax=max_count
                )

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Recommendation Count', fontsize=20, fontweight='bold')
            cbar.ax.tick_params(labelsize=18)
        else:
            # Empty heatmap
            im = ax.imshow(
                heatmap_data,
                cmap='YlOrRd',
                aspect='auto',
                vmin=0,
                vmax=1
            )

        # Set title and labels
        ax.set_title(f'{method}', fontsize=28, fontweight='bold', pad=15)
        ax.set_xlabel('City Grid Column', fontsize=22, fontweight='bold')
        ax.set_ylabel('City Grid Row', fontsize=22, fontweight='bold')

        # Adjust ticks to be less dense
        ax.set_xticks(np.arange(0, n_cols, 2))
        ax.set_xticklabels(np.arange(0, n_cols, 2))
        ax.set_yticks(np.arange(0, n_rows, 4))
        ax.set_yticklabels(np.arange(0, n_rows, 4))
        ax.tick_params(axis='both', labelsize=18)

        # Add statistics text
        total_recs = city_counts['count'].sum()
        non_zero_cities = (city_counts['count'] > 0).sum()
        stats_text = f'Total: {int(total_recs)}\nCities: {non_zero_cities}/200'
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=18,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

    plt.suptitle(
        f'City Recommendation Heatmap: {model_name.upper()}',
        fontsize=32,
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout(rect=(0, 0, 1, 0.96))

    # Save the plot
    file_name = f'{model_name}_city_heatmap'
    save_plots(
        file_name=file_name,
        subfolder="heatmap",
        extensions=["pdf", "png"],
        copy_to_paper=False
    )
    plt.close()

    print(f"Created heatmap for {model_name}")


def make_plot_configs_with_rejection(model_name, rejection_strategy):
    """
    Create plot configurations for a specific rejection strategy.

    Args:
        model_name: Name of the model
        rejection_strategy: 'aggressive' or 'majority'

    Returns:
        List of configuration dictionaries
    """
    base = f"../data/collab-rec-2026/llm-results/{model_name}"

    configs = [
        {
            "label": "SASI",
            "file": f"{base}/sasi/{model_name}_sasi.json",
            "method": "sasi",
            "rounds": 10,
        },
        {
            "label": "MASI",
            "file": f"{base}/mami/{model_name}_{rejection_strategy}_10_rounds_fewshot.json",
            "method": "masi",
            "rounds": 1,
        },
        {
            "label": "MAMI",
            "file": f"{base}/mami/{model_name}_{rejection_strategy}_10_rounds_fewshot.json",
            "method": "mami",
            "rounds": 10,
        },
        {
            "label": "MAMI_early",
            "type": "early_stopping",
            "file": f"{base}/mami/{model_name}_{rejection_strategy}_10_rounds_fewshot.json",
            "earlystopping_csv": f"../data/collab-rec-2026/analysis/{model_name}_mami_{rejection_strategy}_earlystopping_scores.csv",
        },
    ]

    return configs


def calculate_method_coverage(config, kb, random_cities):
    """
    Calculate coverage for a single method configuration.

    Args:
        config: Configuration dictionary
        kb: Knowledge base dataframe
        random_cities: List of cities to consider

    Returns:
        Tuple of (unique_cities, total_recommendations, coverage_pct, avg_recs_per_city)
    """
    recommendations = get_recommendations_for_config(config)
    cities_to_plot = prepare_city_plot_data(recommendations, kb, random_cities)

    unique_cities = len(cities_to_plot)
    total_recommendations = int(cities_to_plot['count'].sum())
    coverage_pct = (unique_cities / 200) * 100
    avg_recs_per_city = total_recommendations / unique_cities if unique_cities > 0 else 0

    return unique_cities, total_recommendations, coverage_pct, avg_recs_per_city


def calculate_all_coverage(models, rejection_strategies):
    """
    Calculate coverage statistics for all models and rejection strategies.

    Args:
        models: List of model names
        rejection_strategies: List of rejection strategies

    Returns:
        Dictionary with coverage statistics organized by model and rejection strategy
    """
    random_cities = CITIES
    kb = pd.read_csv("../data/collab-rec-2026/input-data/kb/merged_listing.csv")

    all_stats = {}

    for model in models:
        all_stats[model] = {}

        for rej_str in rejection_strategies:
            print(f"Processing {model} - {rej_str}...")

            configs = make_plot_configs_with_rejection(model, rej_str)
            method_stats = {}

            for config in configs:
                label = config["label"]
                try:
                    unique, total, cov_pct, avg_per_city = calculate_method_coverage(
                        config, kb, random_cities
                    )
                    method_stats[label] = {
                        'unique_cities': unique,
                        'total_recommendations': total,
                        'coverage_pct': cov_pct,
                        'avg_recs_per_city': avg_per_city
                    }
                    print(f"  {label}: {unique} cities, {total} recs, {cov_pct:.1f}% coverage, {avg_per_city:.1f} avg/city")
                except Exception as e:
                    print(f"  Warning: Could not process {label} for {model}-{rej_str}: {e}")
                    method_stats[label] = None

            all_stats[model][rej_str] = method_stats

    return all_stats


def save_consolidated_coverage(all_stats, models, rejection_strategies):
    """
    Save consolidated coverage statistics to a single text file.

    Args:
        all_stats: Dictionary with all coverage statistics
        models: List of model names
        rejection_strategies: List of rejection strategies
    """
    output_dir = "../data/collab-rec-2026/analysis"
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"{output_dir}/coverage.txt"

    with open(output_file, 'w') as f:
        f.write("City Coverage Statistics Across Models and Rejection Strategies\n")
        f.write("=" * 120 + "\n\n")
        f.write("Format: coverage % (avg recs/city)\n\n")

        # Table header
        header = f"{'Model':<15} | {'Rej_Str':<10} | {'SASI':<20} | {'MASI':<20} | {'MAMI_early':<20} | {'MAMI':<20}"
        f.write(header + "\n")
        f.write("-" * 120 + "\n")

        # Table rows
        for model in models:
            for rej_str in rejection_strategies:
                if model not in all_stats or rej_str not in all_stats[model]:
                    continue

                stats = all_stats[model][rej_str]
                row_parts = [f"{model:<15}", f"{rej_str:<10}"]

                for method in ['SASI', 'MASI', 'MAMI_early', 'MAMI']:
                    if method in stats and stats[method] is not None:
                        cov = stats[method]['coverage_pct']
                        avg = stats[method]['avg_recs_per_city']
                        cell = f"{cov:>5.1f}% ({avg:>5.1f})"
                        row_parts.append(f"{cell:<20}")
                    else:
                        row_parts.append(f"{'N/A':<20}")

                f.write(" | ".join(row_parts) + "\n")

        f.write("=" * 120 + "\n\n")

        # Additional summary statistics
        f.write("Summary Statistics:\n")
        f.write("-" * 120 + "\n\n")

        for model in models:
            f.write(f"\n{model.upper()}:\n")
            for rej_str in rejection_strategies:
                if model not in all_stats or rej_str not in all_stats[model]:
                    continue

                f.write(f"  {rej_str}:\n")
                stats = all_stats[model][rej_str]

                for method in ['SASI', 'MASI', 'MAMI_early', 'MAMI']:
                    if method in stats and stats[method] is not None:
                        s = stats[method]
                        f.write(f"    {method:<12}: {s['unique_cities']:>3} cities ({s['coverage_pct']:>5.1f}%), "
                               f"{s['total_recommendations']:>5} total recs, {s['avg_recs_per_city']:>5.1f} avg/city\n")

    print(f"\nSaved consolidated coverage statistics to {output_file}")


def compute_city_dist(model_name, rejection_strategy):
    """
    Compute city distributions for all methods and create Lorenz curve and heatmap plots.

    Args:
        model_name: Name of the model to analyze
        rejection_strategy: Rejection strategy to use ('aggressive' or 'majority')
    """
    random_cities = CITIES
    kb = pd.read_csv("../data/collab-rec-2026/input-data/kb/merged_listing.csv")

    plot_configs = make_plot_configs_with_rejection(model_name, rejection_strategy)

    # Filter to only SASI, MASI, and MAMI (not early stopping)
    relevant_configs = [
        config for config in plot_configs
        if config["label"] in ["SASI", "MASI", "MAMI"]
    ]

    # Collect data for each method
    method_data_dict = {}
    for config in relevant_configs:
        recommendations = get_recommendations_for_config(config)
        cities_to_plot = prepare_city_plot_data(recommendations, kb, random_cities)
        method_data_dict[config["label"]] = cities_to_plot
        print(
            f"{model_name} - {config['label']}: {len(cities_to_plot)} unique cities, {cities_to_plot['count'].sum()} total recommendations")

    # Create single Lorenz curve plot with all three methods
    plot_lorenz_curve(model_name, method_data_dict, rejection_strategy)
    #
    # # Create heatmap plot with all three methods
    # plot_heatmap(model_name, method_data_dict)


MODELS = ["claude", "gemini", "gpt", "gemma-12b", "olmo-7b", "gemma-4b"]
REJECTION_STRATEGIES = ["aggressive", "majority"]

if __name__ == "__main__":
    # Generate plots for all models (using aggressive strategy)
    print("Generating plots for all models...")
    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Processing {model}...")
        print('='*60)
        compute_city_dist(model, rejection_strategy='aggressive')
        compute_city_dist(model, rejection_strategy='majority')

    # Generate consolidated coverage statistics
    # print("\n" + "="*60)
    # print("Calculating coverage statistics for all models and rejection strategies...")
    # print("="*60)
    # all_stats = calculate_all_coverage(MODELS, REJECTION_STRATEGIES)
    # save_consolidated_coverage(all_stats, MODELS, REJECTION_STRATEGIES)

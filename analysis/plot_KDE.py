import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import os
from dotenv import load_dotenv

from analysis.plot_utils import save_plots, set_paper_style

load_dotenv()


def prepare_data(mami_agg_df: pd.DataFrame | None,
                 mami_maj_df: pd.DataFrame | None,
                 sasi_df: pd.DataFrame | None,
                 city_pop_map: dict[str, int]):
    """Prepare data from MAMI (aggressive & majority), MASI (round 1), and SASI dataframes."""

    dfs_to_combine = []

    # Extract MASI (Round 1) and MAMI (Round 10) from MAMI data - for both rejection strategies
    if mami_agg_df is not None:
        # MASI aggressive (Round 1)
        masi_agg = mami_agg_df.loc[(mami_agg_df["agent_name"] == "moderator") &
                                   (mami_agg_df["round_nr"] == 1)].copy()
        masi_agg['Method'] = 'MASI'
        masi_agg['rejection_strategy'] = 'aggressive'
        dfs_to_combine.append(masi_agg)

        # MAMI aggressive (Round 10)
        mami_agg_r10 = mami_agg_df.loc[(mami_agg_df["agent_name"] == "moderator") &
                                       (mami_agg_df["round_nr"] == 10)].copy()
        mami_agg_r10['Method'] = 'MAMI'
        mami_agg_r10['rejection_strategy'] = 'aggressive'
        dfs_to_combine.append(mami_agg_r10)

    if mami_maj_df is not None:
        # MASI majority (Round 1)
        masi_maj = mami_maj_df.loc[(mami_maj_df["agent_name"] == "moderator") &
                                   (mami_maj_df["round_nr"] == 1)].copy()
        masi_maj['Method'] = 'MASI'
        masi_maj['rejection_strategy'] = 'majority'
        dfs_to_combine.append(masi_maj)

        # MAMI majority (Round 10)
        mami_maj_r10 = mami_maj_df.loc[(mami_maj_df["agent_name"] == "moderator") &
                                       (mami_maj_df["round_nr"] == 10)].copy()
        mami_maj_r10['Method'] = 'MAMI'
        mami_maj_r10['rejection_strategy'] = 'majority'
        dfs_to_combine.append(mami_maj_r10)

    if sasi_df is not None and len(sasi_df) > 0:
        sasi_df = sasi_df.copy()

        # Handle different SASI file formats
        # Some models use 'final_offer' instead of 'candidates'
        if 'final_offer' in sasi_df.columns and 'candidates' not in sasi_df.columns:
            sasi_df['candidates'] = sasi_df['final_offer']
            print("  Note: Using 'final_offer' as 'candidates' for SASI data")

        # Check if candidates column exists
        if 'candidates' not in sasi_df.columns:
            print("  Warning: SASI data missing 'candidates' column, skipping SASI")
            sasi_df = None
        else:
            sasi_df['Method'] = 'SASI'
            sasi_df['rejection_strategy'] = 'aggressive'  # SASI only has one "strategy"
            dfs_to_combine.append(sasi_df)
    else:
        print("  Warning: No SASI data available for this model")

    # Safeguard: if no df available, return empty dataframe with expected columns
    if len(dfs_to_combine) == 0:
        print("  Warning: No data available to prepare. Returning empty dataframe.")
        empty = pd.DataFrame(columns=['Method', 'rejection_strategy', 'candidates', 'pop_value'])
        return empty

    # Combine all dataframes
    combined_df = pd.concat(dfs_to_combine, ignore_index=True)

    # Convert candidates string to list and explode
    combined_df['candidates'] = combined_df['candidates'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df_exploded = combined_df.explode('candidates')

    # Map city names to popularity values
    df_exploded['pop_value'] = df_exploded['candidates'].map(city_pop_map)

    # Drop rows with missing popularity values
    df_exploded = df_exploded.dropna(subset=['pop_value'])

    return df_exploded


def plot_kde_by_model(df_viz, model_name, rejection_strategy: str | None = None):
    """Create KDE plot for a single model with SASI, MASI, and MAMI.

    Args:
        df_viz: Combined dataframe with columns ['Method', 'rejection_strategy', 'pop_value'].
        model_name: Name of the model (used for titles and filenames).
        rejection_strategy: If None, plot both 'aggressive' (solid) and 'majority' (dashed) lines.
                           If specified, plot only this strategy (always solid).
    """
    set_paper_style()
    fig, ax = plt.subplots(figsize=(16, 10))

    methods = ['SASI', 'MASI', 'MAMI']
    colors = ['#e74c3c', '#2754F5', '#2ecc71']  # Red, Blue, Green
    linestyles_default = {'aggressive': '-', 'majority': '--'}

    plotted_methods = set()

    for method, color in zip(methods, colors):
        # Special-case SASI: it doesn't have a rejection strategy; plot it once regardless
        if method == 'SASI':
            strategies_to_plot = [None]
        else:
            strategies_to_plot = [rejection_strategy] if rejection_strategy is not None else ['aggressive', 'majority']

        for strat in strategies_to_plot:
            # For SASI (strat is None) we ignore the rejection_strategy filter
            if method == 'SASI':
                data = df_viz[df_viz['Method'] == 'SASI']
            else:
                data = df_viz[(df_viz['Method'] == method) &
                              (df_viz['rejection_strategy'] == strat)]

            if len(data) == 0:
                continue

            # Choose linestyle: solid if a single strategy was requested; otherwise per-default
            if rejection_strategy is not None:
                linestyle = '-'
            else:
                # strat can be 'aggressive' or 'majority' here
                linestyle = linestyles_default.get(str(strat), '-')

            # Only add label once per method
            label = method if method not in plotted_methods else None
            if label:
                plotted_methods.add(method)

            sns.kdeplot(
                data=data,
                x='pop_value',
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=8,
                ax=ax,
                fill=True,
                alpha=0.2 if linestyle == '-' else 0.15
            )

    # Axes labels and ticks
    ax.set_xlabel('City Popularity Score', fontsize=36, fontweight='bold')
    ax.set_ylabel('Density', fontsize=36, fontweight='bold')
    ax.tick_params(axis='x', labelsize=36)
    ax.tick_params(axis='y', labelsize=36)

    # Custom legend with only method names and colors
    handles = [plt.Line2D([0], [0], color=color, linewidth=3, label=method)
               for method, color in zip(methods, colors) if method in plotted_methods]
    ax.legend(handles=handles, loc='best', fontsize=30, frameon=True,
              framealpha=0.9, edgecolor='black')

    plt.tight_layout()

    # Save plots
    suffix = 'comparison' if rejection_strategy is None else rejection_strategy
    output_file = f'{model_name}_kde_{suffix}'

    save_plots(output_file, extensions=["pdf"], subfolder="kde_plots", copy_to_paper=True,
               paper_location=os.getenv("PAPER_LOCATION"))
    ax.set_title(f'{model_name.upper()}', fontsize=18, fontweight='bold', pad=20)
    save_plots(output_file, extensions=["png"], subfolder="kde_plots", copy_to_paper=False)

    print(f"Saved plot: {output_file}")
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)


def load_csv_if_exists(path):
    """Helper to load a CSV if it exists, else return None."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"    Loaded {os.path.basename(path)} with shape {df.shape}")
        return df
    else:
        print(f"    Warning: File not found: {os.path.basename(path)}")
        return None


def get_sasi_path(model_name):
    """Return SASI file path for a model, trying alternative suffix if needed."""
    sasi_path = f"../data/collab-rec-2026/analysis/{model_name}_sasi.csv"
    if not os.path.exists(sasi_path):
        sasi_path_alt = f"../data/collab-rec-2026/analysis/{model_name}_sasi_scores_with_relevance.csv"
        if os.path.exists(sasi_path_alt):
            sasi_path = sasi_path_alt
            print(f"  Using SASI file: {os.path.basename(sasi_path)}")
        else:
            print(f"  Warning: No SASI file found for {model_name}")
            sasi_path = None
    return sasi_path


def plot_kde_for_models(models=None, rejection_strategies=None):
    """
    Generate KDE plots for given models and rejection strategies.

    Args:
        models (list[str]): List of model names.
        rejection_strategies (list[str] or None): If None, load both 'aggressive' and 'majority' MAMI.
    """
    if models is None:
        models = ['gemini', 'gemma-12b', 'gemma-4b', 'gpt', 'olmo-7b', 'claude', 'smol-3b']

    # Load city popularity mapping
    kb_path = "../data/collab-rec-2026/input-data/kb/merged_listing.csv"
    print(f"Loading knowledge base from: {kb_path}")
    kb = pd.read_csv(kb_path)
    city_pop_map = dict(zip(kb['city'], kb['weighted_pop_score']))
    print(f"Loaded {len(city_pop_map)} cities with popularity scores\n")

    successful_plots = []
    failed_plots = []

    for model_name in models:
        print(f"{'=' * 60}\nProcessing model: {model_name}\n{'=' * 60}")
        try:
            mami_data = {}
            # Determine which rejection strategies to load
            strategies = rejection_strategies or ['aggressive', 'majority']
            for strat in strategies:
                path = f"../data/collab-rec-2026/analysis/{model_name}_mami_{strat}_scores.csv"
                if not os.path.exists(path):
                    path = f"../data/collab-rec-2026/analysis/{model_name}_mami_{strat}_scores_with_relevance.csv"
                mami_data[strat] = load_csv_if_exists(path)

            # Ensure at least one MAMI file was loaded
            if all(v is None for v in mami_data.values()):
                raise FileNotFoundError(f"No MAMI data found for {model_name} (strategies: {strategies})")

            # Load SASI if available
            sasi_path = get_sasi_path(model_name)
            sasi_df = load_csv_if_exists(sasi_path) if sasi_path else None

            # Prepare combined dataframe
            print(f"  Preparing combined data...")
            # If only one strategy is provided, pass it; else combine all MAMI dfs
            df_viz = prepare_data(
                mami_data.get('aggressive'),
                mami_data.get('majority'),
                sasi_df,
                city_pop_map
            )
            print(f"    Combined shape: {df_viz.shape}")
            print(f"    Methods available: {df_viz['Method'].unique().tolist()}")

            # Create and save plot
            print(f"  Creating plot...")
            # determine requested single strategy (None if both)
            requested = rejection_strategies[0] if rejection_strategies else None
            plot_kde_by_model(df_viz, model_name, rejection_strategy=requested)
            successful_plots.append(model_name)
            print(f"  ✓ Success!\n")

        except FileNotFoundError as e:
            error_msg = f"Missing data file - {e}"
            print(f"  ✗ Skipping {model_name}: {error_msg}\n")
            failed_plots.append((model_name, error_msg))
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            print(f"  ✗ Error processing {model_name}: {error_msg}\n")
            failed_plots.append((model_name, error_msg))

    # Summary
    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    print(f"Successfully generated {len(successful_plots)} plots:")
    for model in successful_plots:
        print(f"  ✓ {model}")

    if failed_plots:
        print(f"\nFailed to generate {len(failed_plots)} plots:")
        for model, error in failed_plots:
            print(f"  ✗ {model}: {error}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    # plot_kde_for_models()  # default: all models, both MAMI strategies
    plot_kde_for_models(rejection_strategies=['aggressive'])  # only aggressive
    plot_kde_for_models(rejection_strategies=['majority'])

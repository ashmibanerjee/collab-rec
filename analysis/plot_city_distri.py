import json
from collections import Counter

import numpy as np
from dotenv import load_dotenv

load_dotenv()
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from matplotlib.ticker import PercentFormatter
from constants import MAX_ROUNDS, CITIES
from analysis.diversity import get_recommendations_combined, _normalize_city_list
from analysis.plot_utils import save_plots, set_paper_style


def get_city_iata():
    city_iata = pd.read_csv("../data/collab-rec-2026/input-data/city_iata_codes.csv")
    return city_iata


def get_city_counts(cities, kb):
    city_iata = get_city_iata()
    city_iata.drop_duplicates(subset=["city"], inplace=True)
    city_counts = Counter(city for city_list in cities for city in city_list)

    # Convert to DataFrame (optional)
    city_counts_df = pd.DataFrame.from_dict(city_counts, orient='index', columns=['count']).reset_index()
    city_counts_df.columns = ['city', 'count']

    city_counts_df = city_counts_df.merge(kb[["city", "popularity"]])
    popularity_order = ['Low', 'Medium', 'High']

    # Convert to ordered categorical
    city_counts_df['popularity'] = pd.Categorical(
        city_counts_df['popularity'],
        categories=popularity_order,
        ordered=True)

    new_df = pd.merge(left=city_counts_df, right=city_iata, how='left')

    return new_df


def load_json_data(file_path):
    with open(file_path, "r") as handle:
        return json.load(handle)


def build_moderator_recommendations_map(data):
    recommendations_by_query_round = {}
    for query in data:
        query_id = query.get("query_id")
        for response in query.get("response", []):
            if response.get("agent_role") != "moderator":
                continue
            round_nr = response.get("round_number")
            if round_nr is None:
                continue
            candidates = response.get("candidates", [])
            if not candidates:
                candidates = response.get("cities", [])
            recommendations_by_query_round[(query_id, int(round_nr))] = _normalize_city_list(candidates)
    return recommendations_by_query_round


def parse_recommendations_cell(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


def get_earlystopping_recommendations(results_file, earlystopping_csv, fallback_round=MAX_ROUNDS):
    if not os.path.exists(earlystopping_csv):
        print(f"Missing early-stopping CSV at {earlystopping_csv}")
        return []
    data = load_json_data(results_file)
    es_df = pd.read_csv(earlystopping_csv)
    es_df = es_df.set_index("query_id")
    recommendations_by_query_round = build_moderator_recommendations_map(data)

    recommendations = []
    for query in data:
        query_id = query.get("query_id")
        row = es_df.loc[query_id] if query_id in es_df.index else None
        early_flag = bool(row["earlystopping_flag"]) if row is not None else False
        early_round = None if row is None else row.get("earlystopping_round")
        early_recs = None if row is None else parse_recommendations_cell(row.get("earlystopping_recommendations"))

        if early_flag:
            if early_recs:
                recommendations.append(_normalize_city_list(early_recs))
                continue
            if early_round is not None and not pd.isna(early_round):
                recs = recommendations_by_query_round.get((query_id, int(early_round)), [])
            else:
                recs = []
            recommendations.append(_normalize_city_list(recs))
        else:
            recs = recommendations_by_query_round.get((query_id, fallback_round), [])
            recommendations.append(_normalize_city_list(recs))

    return recommendations


def get_recommendations_for_config(config):
    if config.get("type") == "early_stopping":
        return get_earlystopping_recommendations(
            results_file=config["file"],
            earlystopping_csv=config["earlystopping_csv"],
            fallback_round=config.get("fallback_round", MAX_ROUNDS),
        )
    data = load_json_data(config["file"])
    return get_recommendations_combined(
        data,
        method=config["method"],
        rounds=config["rounds"],
    )


def prepare_city_plot_data(recommendations, kb, cities):
    city_counts = get_city_counts(recommendations, kb)
    cities_to_plot = city_counts[city_counts["city"].isin(cities)].drop_duplicates().copy()
    total_count = cities_to_plot["count"].sum()
    if total_count > 0:
        cities_to_plot["count_frac"] = cities_to_plot["count"] / total_count
    else:
        cities_to_plot["count_frac"] = 0.0
    return cities_to_plot


def plot_city_histogram(model_name, df, method, y_max=None, xtick_step=1):
    set_paper_style()
    fig, axes = plt.subplots(figsize=(16, 10))
    color_palette = sns.color_palette()
    palette = {
        'High': color_palette[0],
        'Medium': color_palette[1],
        'Low': color_palette[2]
    }
    df_plot = (
        df.sort_values("count", ascending=False)
        .reset_index()  # index (city) becomes a column
    )
    sns.lineplot(
        data=df_plot,
        x='index',
        y='count_frac',
        hue='popularity',
        hue_order=['High', 'Medium', 'Low'],
        palette=palette,
        ax=axes,
        # edgecolor="none",
        linewidth=0.3
    )

    if y_max is not None:
        axes.set_ylim(0, y_max)

    axes.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))

    # ---- X ticks as 0–200 ----
    max_idx = 200
    step = 20
    ticks = np.arange(0, max_idx + 1, step)

    axes.set_xticks(ticks)
    axes.set_xticklabels(
        ticks,
        rotation=45,
        ha="right",
        fontsize=22
    )

    axes.tick_params(axis='y', labelsize=36)

    for label in axes.get_xticklabels():
        label.set_ha("right")
    if xtick_step and xtick_step > 1:
        ticks = axes.get_xticks()
        axes.set_xticks(ticks[::xtick_step])

    # Remove the legend
    if method != "MASI":
        axes.legend_.remove()
    else:
        legend = axes.get_legend()
        legend.set_title("Popularity Levels", prop={'size': 30, 'weight': 'bold'})
        # legend.set_bbox_to_anchor((1, 1))
        legend.set_loc('best')
        for text in legend.get_texts():
            text.set_fontsize(28)
            # text.set_fontweight('bold')
        for line in legend.get_lines():
            line.set_linewidth(5.0)
        # for handle in legend.legend_handles:
        #     handle.set_linewidth(3)  # line width
        # For scatter markers (like in sns.scatterplot), use set_sizes or set_markerfacecolor etc.

    plt.tight_layout()
    file_name = f'{model_name}_citydist_{method}'
    os.makedirs(f'../../plots/pdf/city_dist/', exist_ok=True)
    os.makedirs(f'../../plots/png/city_dist/', exist_ok=True)
    save_plots(file_name=file_name, subfolder="city_dist",
               extensions=["pdf", "png"],
               copy_to_paper=True,
               paper_location=os.getenv("PAPER_LOCATION"))


def make_plot_configs(model_name):
    base = f"../data/collab-rec-2026/llm-results/{model_name}"

    return [
        {
            "label": "SASI",
            "file": f"{base}/sasi/{model_name}_sasi.json",
            "method": "sasi",
            "rounds": 10,
        },
        {
            "label": "MASI",
            "file": f"{base}/mami/{model_name}_aggressive_10_rounds_fewshot.json",
            "method": "masi",
            "rounds": 1,
        },
        {
            "label": "MAMI",
            "file": f"{base}/mami/{model_name}_aggressive_10_rounds_fewshot.json",
            "method": "mami",
            "rounds": 10,
        },
        {
            "label": "MAMI (early)",
            "type": "early_stopping",
            "file": f"{base}/mami/{model_name}_aggressive_10_rounds_fewshot.json",
            "earlystopping_csv": f"../data/collab-rec-2026/analysis/{model_name}_mami_aggressive_earlystopping_scores.csv",
        },
    ]


def compute_city_dist(model_name):
    random_cities = CITIES
    kb = pd.read_csv("../data/collab-rec-2026/input-data/kb/merged_listing.csv")

    plot_configs = make_plot_configs(model_name)

    plotted = []
    for config in plot_configs:
        recommendations = get_recommendations_for_config(config)
        cities_to_plot = prepare_city_plot_data(recommendations, kb, random_cities)
        plotted.append((config["label"], cities_to_plot))

    max_count_frac = 0.0
    for _, df in plotted:
        if not df.empty:
            max_count_frac = max(max_count_frac, df["count_frac"].max())
    y_max = min(1.0, max_count_frac * 1.05) if max_count_frac > 0 else 1.0

    for label, df in plotted:
        plot_city_histogram(model_name, df, method=label, y_max=y_max, xtick_step=1)


MODELS = ["claude", "gemini", "gpt", "gemma-12b", "olmo-7b", "gemma-4b"]
# MODELS = ["smol-3b"]
if __name__ == "__main__":
    for model in MODELS:
        compute_city_dist(model)

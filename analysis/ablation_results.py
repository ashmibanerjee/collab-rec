import pandas as pd
from pathlib import Path

from analysis.diversity import compute_diversity
from analysis.get_scores import get_scores

ABLATION_COMPONENTS = ["success", "reliability", "hallucination", "rank"]
MODELS = ["gemini", "olmo-7b"]
## THis is buggy, use the notebook
ablation_label_map = {
    None: "Full (default)",
    "success": "w/o $r$ (success)",
    "reliability": "w/o $d$ (reliability)",
    "hallucination": "w/o $h$ (hallucination)",
    "rank": "w/o rank"
}


def preprocess_file(model_name, rejection_strategy, rounds, ablated_component):

    file_to_save = f'../data/collab-rec-2026/analysis/ablation/{model_name}_{rejection_strategy}_{rounds}_rounds_ablated_{ablated_component}.csv'
    if not Path(file_to_save).exists():
        df = get_scores(model_name=model_name,
                        output_file=file_to_save,
                        rejection_strategy=rejection_strategy,
                        rounds=rounds,
                        ablated_component=ablated_component)
        print(df)
    else:
        df = pd.read_csv(file_to_save)
        print(f"File {file_to_save} already exists. Loaded existing data with {len(df)} records.")

    # results for early stopping
    # Get the round number where moderator achieves max success_score (rounds 2-5) for each query
    moderator_df = df[(df["agent_name"] == "moderator") & (df["round_nr"].between(2, 5))]
    best_rounds = moderator_df.loc[moderator_df.groupby("query_id")["success_score"].idxmax()]
    best_round_mapping = best_rounds.set_index("query_id")["round_nr"].to_dict()

    # Filter df to keep only rows at the best round for each query
    df_early_stopping = df[df.apply(lambda row: row["round_nr"] == best_round_mapping.get(row["query_id"]), axis=1)]
    df_early_stopping = df_early_stopping.loc[df_early_stopping["agent_name"] == "moderator"]
    if model == "gemini":
        round_nr = 10
    else:
        round_nr = 5
    dump_ablation_results(model_name, rejection_strategy, ablated_component, df_early_stopping, with_early_stopping=True, round_nr=round_nr)
    # results for round 10
    df_round_10 = df[(df["round_nr"] == 5) & (df["agent_name"] == "moderator")]

    dump_ablation_results(model_name, rejection_strategy, ablated_component, df_round_10, round_nr=round_nr)


def dump_ablation_results(model_name: str, rejection_strategy: str,
                          ablated_component: str,
                          df: pd.DataFrame,
                          round_nr: int,
                          with_early_stopping: bool = False) -> None:
    json_file_path = f"../data/collab-rec-2026/llm-results/{model_name}/mami/ablated/{model_name}_{rejection_strategy}_{round_nr}_rounds_ablated_{ablated_component}.json"
    print("=" * 50)
    avg_moderator_scores = df.groupby("query_id")["success_score"].mean()
    gini, entropy = compute_diversity(json_file_path, method="mami", round_nr=10)
    avg_hallucination_scores = df.groupby("query_id")["hallucination_rate"].mean()
    avg_reliability_scores = df.groupby("query_id")["reliability_score"].mean()

    if with_early_stopping:
        results_file = f"../data/collab-rec-2026/analysis/ablation/results_es.txt"
    else:
        results_file = f"../data/collab-rec-2026/analysis/ablation/results.txt"
    setting_label = ablation_label_map.get(ablated_component, "Full (default)")

    row = (
        f"& {model_name} "
        f"& {setting_label} "
        f"& {avg_moderator_scores.mean():.2f} "
        f"& {gini:.2f} "
        f"& {entropy:.2f} "
        f"& {avg_hallucination_scores.mean():.2f} "
        f"& {avg_reliability_scores.mean():.2f} "
        f"\\\\\n"
    )

    with open(results_file, "a") as f:
        # If first row for a model, write model name with multirow
        if setting_label == "Full (default)":
            f.write(f"\\multirow{{5}}{{*}}{{{model_name}}} {row}")
        else:
            f.write(row)


if __name__ == "__main__":
    # this is buggy use the notebook
    for model in MODELS:
        for component in ABLATION_COMPONENTS:
            print(f"Model: {model}, component: {component}")
            if model == "gemini":
                round_nr = 10
            else:
                round_nr = 5
            preprocess_file(model_name=model, rejection_strategy="aggressive", rounds=round_nr, ablated_component=component)

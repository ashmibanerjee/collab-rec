import json
import os
import pandas as pd


def compute_earlystopping_success_rate(model_name: str = "claude", rejection_strategy: str = "aggressive", rounds: int = 10):
    mami_scores_csv = f"../data/collab-rec-2026/analysis/{model_name}_mami_{rejection_strategy}_{rounds}_rounds_scores.csv"
    mami_json_file = f"../data/collab-rec-2026/llm-results/{model_name}/mami/{model_name}_{rejection_strategy}_{rounds}_rounds_fewshot.json"
    if not os.path.exists(mami_scores_csv):
        print(f"Missing MAMI scores CSV at {mami_scores_csv}")
        return pd.DataFrame()

    mami_df = pd.read_csv(mami_scores_csv)
    if "agent_name" in mami_df.columns:
        mami_df = mami_df[mami_df["agent_name"] == "moderator"]

    required_cols = {"query_id", "round_nr", "success_score"}
    if not required_cols.issubset(set(mami_df.columns)):
        print(f"Missing required columns in {mami_scores_csv}: {required_cols - set(mami_df.columns)}")
        return pd.DataFrame()

    recommendations_by_query_round = {}
    if os.path.exists(mami_json_file):
        with open(mami_json_file, "r") as file:
            queries = json.load(file)
        for query in queries:
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
                recommendations_by_query_round[(query_id, int(round_nr))] = candidates
    else:
        print(f"Missing MAMI results JSON at {mami_json_file}")

    records = []
    for query_id, group in mami_df.groupby("query_id"):
        round1 = group[group["round_nr"] == 1]
        if round1.empty:
            continue
        round1_score = round1["success_score"].iloc[0]
        later = group[(group["round_nr"] >= 2) & (group["round_nr"] <= 10)]
        if later.empty:
            max_score = round1_score
            early_round = None
        else:
            max_score = later["success_score"].max()
            best_rounds = later[later["success_score"] == max_score]["round_nr"]
            early_round = int(best_rounds.min()) if len(best_rounds) else None

        early_flag = bool(max_score > round1_score)
        early_recommendations = None
        if early_flag and early_round is not None:
            early_recommendations = recommendations_by_query_round.get((query_id, int(early_round)))
            if early_recommendations is not None:
                early_recommendations = json.dumps(early_recommendations)
        records.append({
            "query_id": query_id,
            "success_score_round1": round1_score,
            "max_success_score_rounds_2_10": max_score,
            "earlystopping_flag": early_flag,
            "earlystopping_round": early_round,
            "earlystopping_recommendations": early_recommendations
        })

    es_df = pd.DataFrame(records)
    if not es_df.empty:
        early_rate = es_df["earlystopping_flag"].mean()
        avg_round = es_df.loc[es_df["earlystopping_flag"], "earlystopping_round"].mean()
        print(f"Early-stopping success rate: {early_rate:.3f} ({es_df['earlystopping_flag'].sum()}/{len(es_df)})")
        print(f"Avg early-stopping round (when improved): {avg_round:.2f}")
    return es_df


def get_score_distri_with_earlystopping(model_name: str = "claude", rejection_strategy: str = "aggressive"):
    def _score_list(df, config_ids):
        if df is None or df.empty:
            return []
        if "query_id" not in df.columns:
            return []
        if "overall_relevance" in df.columns:
            score_col = "overall_relevance"
        elif "mod_succ_relevance_score" in df.columns:
            score_col = "mod_succ_relevance_score"
        elif "success_score" in df.columns:
            score_col = "success_score"
        else:
            return []
        if config_ids:
            df = df[df["query_id"].isin(config_ids)]
        df = df.sort_values("query_id")
        return df[score_col].tolist()

    mami_scores_csv = f"../data/collab-rec-2026/analysis/{model_name}_mami_{rejection_strategy}_scores.csv"
    sasi_scores_csv = f"../data/collab-rec-2026/analysis/{model_name}_sasi_scores_with_relevance.csv"

    if not os.path.exists(mami_scores_csv):
        print(f"Missing MAMI scores CSV at {mami_scores_csv}")
        return [], [], []

    full_mami_df = pd.read_csv(mami_scores_csv)
    if "agent_name" in full_mami_df.columns:
        full_mami_df = full_mami_df[full_mami_df["agent_name"] == "moderator"]
    if "round_nr" in full_mami_df.columns:
        masi_df = full_mami_df[full_mami_df["round_nr"] == 1]
    else:
        masi_df = full_mami_df

    config_ids = []
    if "query_id" in full_mami_df.columns:
        config_ids = full_mami_df["query_id"].unique().tolist()

    sasi_df = pd.DataFrame()
    if os.path.exists(sasi_scores_csv):
        sasi_df = pd.read_csv(sasi_scores_csv)

    sasi_scores = _score_list(sasi_df, config_ids)
    masi_scores = _score_list(masi_df, config_ids)

    es_df = compute_earlystopping_success_rate(model_name=model_name, rejection_strategy=rejection_strategy)
    if es_df.empty:
        return sasi_scores, masi_scores, []

    mami_lookup = {}
    for _, row in full_mami_df.iterrows():
        key = (row["query_id"], int(row["round_nr"]))
        mami_lookup[key] = row["success_score"]

    es_df = es_df.set_index("query_id")
    mami_early_scores = []
    for query_id in sorted(config_ids):
        early_round = es_df["earlystopping_round"].get(query_id)
        if pd.isna(early_round):
            early_round = 1
        early_round = int(early_round)
        score = mami_lookup.get((query_id, early_round))
        if score is not None:
            mami_early_scores.append(score)

    return sasi_scores, masi_scores, mami_early_scores


def main(model_name: str = "claude", rejection_strategy: str = "aggressive", rounds: int = 10):
    df = compute_earlystopping_success_rate(model_name=model_name, rejection_strategy=rejection_strategy, rounds=rounds)
    df.to_csv(f"../data/collab-rec-2026/analysis/{model_name}_mami_{rejection_strategy}_earlystopping_scores_{rounds}_rounds.csv",
              index=False)
    print("Saved scores to CSV.")


if __name__ == "__main__":
    # print("Model: Gemma-12b")
    # main(model_name="olmo-7b", rejection_strategy="aggressive", rounds=20)
    # print("Model: Gemma-4b")
    main(model_name="gemma-4b", rejection_strategy="majority", rounds=10)
    # print("model: gpt")
    # main(model_name="gpt", rejection_strategy="majority")
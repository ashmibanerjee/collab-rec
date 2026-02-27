import os

from typing import List, Dict

import numpy as np
from analysis.early_stopping import get_score_distri_with_earlystopping
from analysis.helpers import load_json, get_agent_responses
from analysis.sig_tests import perform_significance_tests, perform_multicomparison, perform_nonparametric_tests, test
from constants import MAX_ROUNDS
import pandas as pd

from k_base.context_retrieval import ContextRetrieval


def calculate_match_with_filters(offer: List[str], filters: Dict[str, str]):
    retriever = ContextRetrieval()
    total_rel_score = 0
    for city in offer:
        matched_filters = retriever.match_city_with_filters(city, filters)

        unmatched = [key for key in filters.keys() if key not in matched_filters.keys()]
        rel_score = len(matched_filters) / len(filters)
        # print(f"\n \ncity: {city} matched  filters: {matched_filters}, unmatched: {unmatched}, rel_score: {rel_score}")
        total_rel_score += rel_score
    return total_rel_score / len(offer)


def run_relevance_mami_masi(model_name: str = "gemini",
                            rejection_strategy: str = "majority",
                            method: str = "mami",
                            ):
    file_name = f"../data/collab-rec-2026/llm-results/{model_name}/mami/{model_name}_{rejection_strategy}_10_rounds_fewshot.json"
    outputs = load_json(file_name)

    round_nr = MAX_ROUNDS if method == "mami" else 1
    scores_outputs = []
    for output in outputs:
        try:
            moderator_responses = get_agent_responses(output["response"], agent_role="moderator",
                                                      roundnr=round_nr)  # moderator response for the N round
            relevance_score = calculate_match_with_filters(moderator_responses[0]["candidates"],
                                                           output["query_details"]["filters"])
            score_output = {"exp_config": output["exp_config"], "query_id": output["query_id"],
                            "final_offer": moderator_responses[0]["candidates"],
                            "round_nr": moderator_responses[0]["round_number"],
                            "mod_succ_relevance_score": relevance_score}
            scores_outputs.append(score_output)
        except IndexError:
            print(f"No moderator response for {output['query_id']}")
    scores_df = pd.DataFrame(scores_outputs)
    print(
        f'Average Relevance Score: {scores_df["mod_succ_relevance_score"].mean()}, for {len(scores_outputs)} queries using {output["exp_config"]}')
    scores_df.to_csv(
        f"../data/collab-rec-2026/analysis/{model_name}_moderator_success_{rejection_strategy}_{method}.csv",
        index=False)
    print("Saved scores to CSV.")
    return scores_df["mod_succ_relevance_score"].tolist()


def calc_relevance_sasi(model_name: str = "gemini"):
    file_name = f"../data/collab-rec-2026/llm-results/{model_name}/sasi/{model_name}_sasi.json"
    outputs = load_json(file_name)
    scores_outputs = []
    for output in outputs:
        responses = output.get("response") or []
        if not responses:
            continue
        response = responses[0]
        candidates = response.get("candidates", [])
        if not candidates:
            candidates = response.get("cities", [])
        if not candidates:
            continue
        relevance_score = calculate_match_with_filters(candidates, output["query_details"]["filters"])
        score_output = {"exp_config": output["exp_config"], "query_id": output["query_id"],
                        "final_offer": candidates,
                        "mod_succ_relevance_score": relevance_score}
        scores_outputs.append(score_output)
    scores_df = pd.DataFrame(scores_outputs)
    print(
        f'Average Relevance Score: {scores_df["mod_succ_relevance_score"].mean()}, for {len(scores_outputs)} queries using {output["exp_config"]}')
    scores_df.to_csv(f"../data/collab-rec-2026/analysis/{model_name}_sasi.csv", index=False)
    return scores_df["mod_succ_relevance_score"].tolist()


def main(model_name: str = "gemini", rejection_strategy: str = "aggressive"):
    print("=" * 80)
    print("Computing Sig Tests for Relevance")
    print("=" * 80)
    sig_test_analysis_earlystopping(model_name, rejection_strategy)
    print("✓ All relevance computations completed!")
    print("=" * 80)


def get_score_distri(model_name: str = "gemini", rejection_strategy: str = "majority"):
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

    config_ids = []
    sasi_df = pd.DataFrame()
    mami_df = pd.DataFrame()
    masi_df = pd.DataFrame()

    mami_scores_csv = f"../data/collab-rec-2026/analysis/{model_name}_mami_{rejection_strategy}_scores.csv"
    sasi_scores_csv = f"../data/collab-rec-2026/analysis/{model_name}_sasi_scores_with_relevance.csv"

    if os.path.exists(mami_scores_csv):
        full_mami_df = pd.read_csv(mami_scores_csv)
        if "round_nr" in full_mami_df.columns:
            mami_df = full_mami_df[(full_mami_df["round_nr"] == 10) & (full_mami_df["agent_name"] == "moderator")]
            masi_df = full_mami_df[(full_mami_df["round_nr"] == 1) & (full_mami_df["agent_name"] == "moderator")]
        else:
            mami_df = full_mami_df
            masi_df = full_mami_df
        if "query_id" in mami_df.columns:
            config_ids = mami_df["query_id"].unique().tolist()

    if os.path.exists(sasi_scores_csv):
        sasi_df = pd.read_csv(sasi_scores_csv)

    return _score_list(sasi_df, config_ids), _score_list(masi_df, config_ids), _score_list(mami_df, config_ids)


def sig_test_analysis(model_name, rejection_strategy):
    sasi, masi, mami = get_score_distri(model_name=model_name, rejection_strategy=rejection_strategy)
    print(f"SASI, MASI ({rejection_strategy}), MAMI ({rejection_strategy})")
    perform_multicomparison(sasi=sasi, masi=masi, mami=mami)
    perform_significance_tests(sasi=sasi, masi=masi, mami=mami)
    print(
        f"Avg sasi: {np.mean(sasi)}({len(sasi)} queries), masi: {np.mean(masi)}({len(masi)} queries), mami: {np.mean(mami)}({len(mami)} queries)")


def dump_latex_sigtest_results(result, model_name, rejection_strategy):
    if result is None:
        return

    # result[0] is the summary table
    # We need to extract data from it. 
    # The table has columns: group1, group2, stat, pval, pval_corr, reject

    table_data = result[0].data
    # headers are in table_data[0]
    # rows are in table_data[1:]

    comparisons = {}
    for row in table_data[1:]:
        g1, g2, stat, pval, pval_corr, reject = row
        # Normalize group names to lowercase for easier lookup
        g1, g2 = g1.lower(), g2.lower()
        pair = tuple(sorted((g1, g2)))

        # Format: stat (p-value | corrected_p-value) (Reject H0)
        # Reject H0 is usually a boolean in the table, we want to show it as text
        reject_str = "Reject H0" if reject == True or str(reject).lower() == 'true' else "Fail to Reject"

        # Ensure stat and pvals are formatted nicely
        try:
            stat_val = float(stat)
            pval_val = float(pval)
            pval_corr_val = float(pval_corr)
            formatted_val = f"{stat_val:.4f} ({pval_val:.4f} | {pval_corr_val:.4f}) ({reject_str})"
        except ValueError:
            formatted_val = f"{stat} ({pval} | {pval_corr}) ({reject_str})"

        comparisons[pair] = formatted_val

    # Pairs we want: MAMI vs MASI, MAMI vs SASI, MASI vs SASI
    mami_masi = comparisons.get(tuple(sorted(('mami', 'masi'))), "N/A")
    mami_sasi = comparisons.get(tuple(sorted(('mami', 'sasi'))), "N/A")
    masi_sasi = comparisons.get(tuple(sorted(('masi', 'sasi'))), "N/A")

    output_path = "../data/collab-rec-2026/analysis/sig_test_latex_table.txt"

    with open(output_path, "a") as f:
        f.write(f"\nModel: {model_name}, Strategy: {rejection_strategy}\n")
        f.write("MAMI vs MASI | MAMI vs SASI | MASI vs SASI\n")
        f.write(f"{mami_masi} | {mami_sasi} | {masi_sasi}\n")
        f.write("-" * 100 + "\n")
    print(f"Dumped significance results for {model_name} ({rejection_strategy}) to {output_path}")


def sig_test_analysis_earlystopping(model_name, rejection_strategy):
    print(f"model_name: {model_name}, rejection_strategy: {rejection_strategy}")
    sasi, masi, mami_early = get_score_distri_with_earlystopping(model_name=model_name,
                                                                 rejection_strategy=rejection_strategy)
    print(f"SASI, MASI ({rejection_strategy}), MAMI (early-stopping)")
    print(f"Sample sizes: sasi={len(sasi)}, masi={len(masi)}, mami_early={len(mami_early)}")
    if len(mami_early) and len(masi):
        improved = sum(1 for a, b in zip(mami_early, masi) if a > b)
        ties = sum(1 for a, b in zip(mami_early, masi) if a == b)
        print(f"MAMI early vs MASI round1: improved={improved}, ties={ties}")
    result = perform_multicomparison(sasi=sasi, masi=masi, mami=mami_early)
    dump_latex_sigtest_results(result, model_name, rejection_strategy)
    perform_significance_tests(sasi=sasi, masi=masi, mami=mami_early)
    perform_nonparametric_tests(mami_early, masi, sasi)
    test(mami_early, masi)
    print(
        f"Avg sasi: {np.mean(sasi)}({len(sasi)} queries), masi: {np.mean(masi)}({len(masi)} queries), mami (ES): {np.mean(mami_early)}({len(mami_early)} queries)")


REJECTION_STRATEGIES = ["aggressive", "majority"]
MODELS = ["claude", "gemini", "gpt", "gemma-12b", "gemma-4b", "olmo-7b", "smol-3b"]
if __name__ == "__main__":
    for model_name in MODELS:
        for rejection_strategy in REJECTION_STRATEGIES:
            main(model_name, rejection_strategy)

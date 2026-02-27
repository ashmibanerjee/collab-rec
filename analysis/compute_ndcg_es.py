from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
import os

from analysis.helpers import get_agent_responses
from analysis.early_stopping import compute_earlystopping_success_rate
from experiments.helpers import load_queries
from k_base.context_retrieval import ContextRetrieval


def compute_graded_relevance(recommended: List[str], retriever: None | ContextRetrieval, filters: Dict[str, str]) -> \
        Dict[str, float]:
    scores = {}
    if retriever is None:
        retriever = ContextRetrieval()
    for city in recommended:
        matched_filters = retriever.match_city_with_filters(city, filters)
        rel_score = len(matched_filters) / len(filters)
        scores[city] = rel_score
    return scores


def compute_dcg(recommended: List[str], relevance_scores: Dict[str, float], k: int = 10) -> float:
    dcg = 0.0
    for i in range(min(k, len(recommended))):
        city = recommended[i]
        rel_score = relevance_scores.get(city, 0.0)
        dcg += (2 ** rel_score - 1) / np.log2(i + 2)  # i + 2 because i starts at 0
    return dcg


def compute_idcg(relevance_scores: Dict[str, float], k: int = 10) -> float:
    # ideal relevance score for each city is 1.0
    ideal_relevance_scores = [1.0] * len(relevance_scores)
    idcg = 0.0
    for i in range(min(k, len(relevance_scores))):
        rel_score = ideal_relevance_scores[i]
        idcg += (2 ** rel_score - 1) / np.log2(i + 2)
    return idcg


def compute_ndcg(relevance_scores: Dict[str, float], k: int = 10) -> float:
    recommended_cities = list(relevance_scores.keys())
    dcg = compute_dcg(recommended_cities, relevance_scores, k)
    idcg = compute_idcg(relevance_scores, k)
    return dcg / idcg if idcg > 0 else 0.0


def compute_ndcg_for_method(model_name: str, rejection_strategy: str | None, round_nr: int | None, method: str, k: int = 10) -> Tuple[float, int]:
    """
    Compute NDCG for a given method.

    Args:
        model_name: Name of the model
        rejection_strategy: Rejection strategy (for MAMI/MASI)
        round_nr: Round number to evaluate (for MAMI/MASI)
        method: Method name ('sasi', 'masi', 'mami', 'mearly')
        k: Top-k for NDCG computation

    Returns:
        Tuple of (average_ndcg, error_count)
    """
    if method == "sasi":
        file_to_load = f'../data/collab-rec-2026/llm-results/{model_name}/{method}/{model_name}_sasi.json'
    else:
        file_to_load = f'../data/collab-rec-2026/llm-results/{model_name}/mami/{model_name}_{rejection_strategy}_10_rounds_fewshot.json'

    if not os.path.exists(file_to_load):
        print(f"  Warning: File not found: {file_to_load}")
        return 0.0, 0

    data = load_queries(file_to_load)

    # For early stopping, we need to get the early stopping dataframe
    early_stopping_df = None
    if method == "mearly":
        # Try to read existing early stopping CSV file
        es_csv_path = f'../data/collab-rec-2026/analysis/{model_name}_mami_{rejection_strategy}_earlystopping_scores.csv'

        if os.path.exists(es_csv_path):
            early_stopping_df = pd.read_csv(es_csv_path)
        else:
            print(f"  Warning: No early stopping data at {es_csv_path}")
            return 0.0, 0

        if early_stopping_df is None or early_stopping_df.empty:
            print(f"  Warning: Empty early stopping data for {model_name} with {rejection_strategy}")
            return 0.0, 0
        early_stopping_df = early_stopping_df.set_index("query_id")

    relevance_scores = {}
    error_count = 0
    retriever = ContextRetrieval()

    for item in data:
        query_id = item["query_id"]
        try:
            query_details = item.get("query_details", item.get("query", {}))
            filters = query_details["filters"]

            if method == "sasi":
                try:
                    response = item["response"][0]
                    recommended_cities = response.get("candidates", response.get("cities", []))
                except (IndexError, KeyError) as e:
                    recommended_cities = []
                    error_count += 1
            elif method == "mearly":
                # Use early stopping round for this query
                if query_id not in early_stopping_df.index:
                    error_count += 1
                    continue

                early_round = early_stopping_df.loc[query_id, "earlystopping_round"]
                if pd.isna(early_round):
                    early_round = 1
                early_round = int(early_round)

                try:
                    recommended_cities = get_agent_responses(
                        data=item["response"],
                        agent_role="moderator",
                        roundnr=early_round,
                        all_responses=False
                    )[0]["candidates"]
                except (IndexError, KeyError):
                    recommended_cities = []
                    error_count += 1
            else:
                # MASI or MAMI
                try:
                    recommended_cities = get_agent_responses(
                        data=item["response"],
                        agent_role="moderator",
                        roundnr=round_nr,
                        all_responses=False
                    )[0]["candidates"]
                except (IndexError, KeyError):
                    recommended_cities = []
                    error_count += 1

            if recommended_cities:
                relevance_scores[query_id] = compute_graded_relevance(
                    recommended=recommended_cities,
                    filters=filters,
                    retriever=retriever
                )
        except Exception as e:
            print(f"Error processing query_id {query_id}: {e}")
            error_count += 1

    if not relevance_scores:
        return 0.0, error_count

    ndcg_scores = {query_id: compute_ndcg(scores, k) for query_id, scores in relevance_scores.items()}
    avg_ndcg = np.mean(list(ndcg_scores.values()))

    return avg_ndcg, error_count


def main(model_name: str, rejection_strategy: str | None, round_nr: int | None, method: str):
    """Legacy main function for backward compatibility."""
    avg_ndcg, error_count = compute_ndcg_for_method(model_name, rejection_strategy, round_nr, method, k=10)

    if method == "sasi":
        print(f"Average NDCG@10 for {model_name} SASI: {avg_ndcg:.4f}")
    elif method == "masi":
        print(f"Average NDCG@10 for {model_name} MASI ({rejection_strategy}): {avg_ndcg:.4f}")
    elif method == "mearly":
        print(f"Average NDCG@10 for {model_name} MAMI Early Stopping ({rejection_strategy}): {avg_ndcg:.4f}")
    else:
        print(f"Average NDCG@10 for {model_name} MAMI ({rejection_strategy}): {avg_ndcg:.4f}")

    if error_count > 0:
        print(f"\tNumber of queries with errors in extracting recommendations: {error_count}")


def compute_all_ndcg_results(model_names: List[str], rejection_strategies: List[str], output_file: str = "../data/collab-rec-2026/analysis/ndcg.txt"):
    """
    Compute NDCG for all models, rejection strategies, and methods.
    Write results to a table file.

    Args:
        model_names: List of model names to evaluate
        rejection_strategies: List of rejection strategies
        output_file: Path to output file
    """
    results = []

    for model_name in model_names:
        print(f"\n{'='*80}")
        print(f"Processing model: {model_name}")
        print(f"{'='*80}")

        # Compute SASI NDCG
        print(f"\nComputing SASI...")
        sasi_ndcg, _ = compute_ndcg_for_method(model_name, None, None, "sasi", k=10)

        for rejection_strategy in rejection_strategies:
            print(f"\nProcessing rejection strategy: {rejection_strategy}")

            # Compute MASI NDCG (round 1)
            print(f"  Computing MASI...")
            masi_ndcg, _ = compute_ndcg_for_method(model_name, rejection_strategy, 1, "masi", k=10)

            # Compute MAMI NDCG (round 10)
            print(f"  Computing MAMI (round 10)...")
            mami_ndcg, _ = compute_ndcg_for_method(model_name, rejection_strategy, 10, "mami", k=10)

            # Compute Early Stopping NDCG
            print(f"  Computing MAMI Early Stopping...")
            mearly_ndcg, _ = compute_ndcg_for_method(model_name, rejection_strategy, None, "mearly", k=10)

            results.append({
                'model_name': model_name,
                'rejection_strategy': rejection_strategy,
                'sasi_ndcg': sasi_ndcg,
                'masi_ndcg': masi_ndcg,
                'mearly_ndcg': mearly_ndcg,
                'mami_ndcg': mami_ndcg
            })

            print(f"  Results: SASI={sasi_ndcg:.2f}, MASI={masi_ndcg:.2f}, Mearly={mearly_ndcg:.2f}, M10={mami_ndcg:.2f}")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Write to file in table format
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("NDCG@10 Results\n")
        f.write("="*100 + "\n\n")
        f.write(f"{'Model':<15} | {'Rej_Str':<12} | {'SASI':<10} | {'MASI':<10} | {'Mearly':<10} | {'M10 (MAMI)':<12}\n")
        f.write("-"*100 + "\n")

        for _, row in df.iterrows():
            f.write(f"{row['model_name']:<15} | {row['rejection_strategy']:<12} | {row['sasi_ndcg']:<10.4f} | {row['masi_ndcg']:<10.4f} | {row['mearly_ndcg']:<10.4f} | {row['mami_ndcg']:<12.4f}\n")

    print(f"\n{'='*80}")
    print(f"✓ Results written to: {output_file}")
    print(f"{'='*80}")

    # Also save as CSV for easier processing
    csv_file = output_file.replace('.txt', '.csv')
    df.to_csv(csv_file, index=False)
    print(f"✓ Results also saved as CSV: {csv_file}")

    return df


if __name__ == "__main__":

    MODEL_NAMES = ["claude", "gemini", "gpt", "gemma-12b", "olmo-7b", "gemma-4b"]
    REJECTION_STRATEGIES = ["aggressive", "majority"]

    # Compute all NDCG results and write to table
    compute_all_ndcg_results(MODEL_NAMES, REJECTION_STRATEGIES)

    # Legacy code for individual computations (commented out)
    """
    rounds = 10
    methods = ["sasi", "masi", "mami"]

    for model_name in MODEL_NAMES:
        for method in methods:
            if method == "mami":
                for rejection_strategy in REJECTION_STRATEGIES:
                        print(
                            f"Computing NDCG for {model_name} with {rejection_strategy} rejection strategy and method {method}")
                        main(model_name, rejection_strategy, rounds, method)
            elif method == "sasi":
                print(f"Computing NDCG for {model_name} with method {method}")
                main(model_name, None, rounds, method)
            else:
                print(f"Computing NDCG for {model_name} with method {method}")
                main(model_name, "aggressive", 1, method)
            print("=" * 80)
    """

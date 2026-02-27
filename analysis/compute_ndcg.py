from typing import List, Dict

import pandas as pd
import numpy as np
import json

from analysis.compute_relevance import calculate_match_with_filters
from analysis.helpers import get_agent_responses
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


def main(model_name: str, rejection_strategy: str | None, round_nr: int | None, method: str):
    if method == "sasi":
        file_to_load = f'../data/collab-rec-2026/llm-results/{model_name}/{method}/{model_name}_sasi.json'
    else:
        file_to_load = f'../data/collab-rec-2026/llm-results/{model_name}/mami/{model_name}_{rejection_strategy}_10_rounds_fewshot.json'

    data = load_queries(file_to_load)
    relevance_scores = {}
    error_count = 0
    for item in data:
        query_id = item["query_id"]
        query_details = item.get("query_details", item.get("query", {}))
        filters = query_details["filters"]
        if method == "sasi":
            try:
                response = item["response"][0]
                recommended_cities = response.get("candidates", response.get("cities", []))
            except (IndexError, KeyError) as e:
                recommended_cities = []
                error_count += 1
        else:
            recommended_cities = get_agent_responses(data=item["response"],
                                                     agent_role="moderator",
                                                     roundnr=round_nr,
                                                     all_responses=False)[0]["candidates"]  # Get recommendations from the last round
        if recommended_cities:
            relevance_scores[query_id] = compute_graded_relevance(
                                            recommended=recommended_cities,
                                            filters=filters,
                                            retriever=ContextRetrieval())
    ndcg_scores = {query_id: compute_ndcg(scores) for query_id, scores in relevance_scores.items()}
    avg_ndcg = np.mean(list(ndcg_scores.values()))
    print(f"Average NDCG@{rounds} for {model_name} with {rejection_strategy} rejection strategy: {avg_ndcg:.4f}")
    print("\tNumber of queries with errors in extracting recommendations:", error_count)


if __name__ == "__main__":

    MODEL_NAMES = ["gpt", "gemma-12b", "olmo-7b", "gemma-4b"]
    REJECTION_STRATEGIES = ["aggressive", "majority"]
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

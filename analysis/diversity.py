import os

import json
import math

import numpy as np
from collections import Counter

import pandas as pd

from constants import MAX_ROUNDS

REJECTION_STRATEGIES = ["aggressive", "majority"]


def _normalize_city_list(items):
    normalized = []
    for item in items or []:
        if isinstance(item, str):
            normalized.append(item)
        elif isinstance(item, dict):
            city_name = item.get("city") or item.get("name")
            if isinstance(city_name, str):
                normalized.append(city_name)
    return normalized


def gini_index(recommendations):
    # recommendations is a list of lists of final recommendations for N queries
    # Flatten the list of recommendations
    flattened = [
        item
        for sublist in recommendations
        for item in _normalize_city_list(sublist)
    ]

    # Count frequency of each item
    freqs = Counter(flattened)

    # Convert counts to numpy array and sort
    values = np.array(sorted(freqs.values()))
    n = len(values)
    if n == 0 or np.sum(values) == 0:
        return 0.0

    # Gini index calculation
    index = np.arange(1, n + 1)
    gini = (np.sum((2 * index - n - 1) * values)) / (n * np.sum(values))

    return gini


def compute_normalized_entropy(recommendations):
    # Example: recommendations is a list of 45 lists, each of length 10
    # recommendations = [[item1, item2, ..., item10], ..., [item1, ..., item10]]
    flattened = [
        item
        for sublist in recommendations
        for item in _normalize_city_list(sublist)
    ]
    freqs = Counter(flattened)
    total = len(flattened)  # should be 450
    if total == 0:
        return {
            'entropy': 0.0,
            'normalized_entropy': 0.0
        }

    # print(f"Unique items: {len(freqs.keys())}")

    # Step 2: Calculate probabilities
    probs = [count / total for count in freqs.values()]
    # print(probs)

    # Step 3: Compute entropy
    entropy = round(-sum(p * math.log(p) for p in probs), 3)
    if len(freqs) == 0:
        normalized_entropy = 0.0
    else:
        normalized_entropy = round(entropy / math.log(len(freqs)), 3)

    return {
        'entropy': entropy,
        'normalized_entropy': normalized_entropy
    }


def exposure_normalized_entropy(recommendations, full_city_list):
    """
    recommendations: list of recommendation lists
    full_city_list: list of ALL cities in the KB
    """
    flattened = [
        item
        for sublist in recommendations
        for item in _normalize_city_list(sublist)
    ]
    freqs = Counter(flattened)
    total = len(flattened)

    # Include zero-exposure cities
    probs = []
    for city in full_city_list:
        p = freqs.get(city, 0) / total
        if p > 0:
            probs.append(p)

    # Entropy
    entropy = -sum(p * math.log(p) for p in probs)

    # Normalize by FULL catalog size
    max_entropy = math.log(len(full_city_list))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    unique_cities = set(
        item
        for sublist in recommendations
        for item in _normalize_city_list(sublist)
    )
    coverage = len(unique_cities)
    print("Coverage: ", coverage, " / ", len(full_city_list))

    return {
        "entropy": entropy,
        "normalized_entropy": normalized_entropy
    }


def exposure_gini(counter, full_city_list):
    """
    counter: Counter with city -> frequency
    full_city_list: list of ALL cities in the KB
    """
    x = np.array([counter.get(city, 0) for city in full_city_list], dtype=np.float64)

    if np.sum(x) == 0:
        return 0.0

    x = np.sort(x)
    n = len(x)
    cum_ix = np.arange(1, n + 1)

    gini = (2 * np.sum(cum_ix * x)) / (n * np.sum(x)) - (n + 1) / n
    return gini


def get_recommendations_combined(data, method="mami", rounds=10):
    recommendations = []
    if method == "top_pop" or method == "random":
        column_name = "top_pop_baseline" if method == "top_pop" else "random_baseline"
        recommendations = [_normalize_city_list(query.get(column_name, [])) for query in data]
        return recommendations

    if method == "sasi":
        # SASI has direct recommendations in response
        for query in data:
            response = query.get('response', [])
            try:
                # SASI response is a list of candidates directly
                candidates = response[0].get('candidates', []) if response else []
                if not candidates and response:
                    candidates = response[0].get('cities', [])
                recommendations.append(_normalize_city_list(candidates))
            except Exception as e:
                print(f"Error processing query_id {query.get('query_id', 'UNKNOWN')}: {e}")
                continue
    else:
        # MAMI/MASI have rounds and moderator structure
        for query in data:
            response = query.get('response', [])

            # Filter for moderator agent
            moderator_response = [
                r for r in response
                if (r.get("agent_role") == "moderator") and (r.get("round_number") == rounds)
            ]

            # Get candidates from moderator
            if not moderator_response:
                print(f"Missing moderator response for query_id {query.get('query_id', 'UNKNOWN')}")
                recommendations.append([])
                continue
            moderator_candidates = moderator_response[0].get('candidates', [])
            recommendations.append(_normalize_city_list(moderator_candidates))

    return recommendations


def compute_diversity(file_path, method="mami", round_nr=10):
    data = json.load(open(file_path))
    recommendations = get_recommendations_combined(data, method=method, rounds=round_nr)
    gini = gini_index(recommendations)
    print(f"Gini (index): {gini:.3f}")
    entropy = compute_normalized_entropy(recommendations)
    print(f"Entropy (normalized): {entropy['normalized_entropy']:.3f}")
    return gini, entropy['normalized_entropy']


def compute_diversity_with_early_stopping(model_name: str = "claude", rejection_strategy: str = "aggressive"):
    print("Computing diversity metrics for MAMI with early stopping")
    mami_scores_csv = f"../data/collab-rec-2026/analysis/{model_name}_mami_{rejection_strategy}_earlystopping_scores.csv"
    if not os.path.exists(mami_scores_csv):
        print(f"Missing MAMI scores CSV at {mami_scores_csv}")
        return

    mami_df_es = pd.read_csv(mami_scores_csv)
    mami_json_file = f"../data/collab-rec-2026/llm-results/{model_name}/mami/{model_name}_{rejection_strategy}_10_rounds_fewshot.json"
    data = json.load(open(mami_json_file))

    # Get recommendations for each query with dynamic round_nr
    recommendations = []
    for query in data:
        query_id = query.get('query_id')

        # Get early stopping info for this query
        query_es_info = mami_df_es[mami_df_es['query_id'] == query_id]

        if query_es_info.empty:
            print(f"Warning: No early stopping info found for query_id {query_id}")
            round_nr = MAX_ROUNDS
        else:
            # Determine round_nr based on early_stopping_flag
            early_stopping_flag = query_es_info.iloc[0]['earlystopping_flag']
            if early_stopping_flag:
                round_nr = int(query_es_info.iloc[0]['earlystopping_round'])
            else:
                round_nr = MAX_ROUNDS

        # Get recommendations for this specific query at the determined round
        response = query.get('response', [])
        moderator_response = [
            r for r in response
            if (r.get("agent_role") == "moderator") and (r.get("round_number") == round_nr)
        ]

        if not moderator_response:
            print(f"Missing moderator response for query_id {query_id} at round {round_nr}")
            recommendations.append([])
            continue

        moderator_candidates = moderator_response[0].get('candidates', [])
        recommendations.append(_normalize_city_list(moderator_candidates))

    # Compute diversity metrics
    print("total recommendations: ", len(recommendations), " / ", len(data), " =")
    gini = gini_index(recommendations)
    print(f"Gini (index): {gini:.3f}")
    entropy = compute_normalized_entropy(recommendations)
    print(f"Entropy (normalized): {entropy['normalized_entropy']:.3f}")


def main(model_name="gemini"):
    print("Computing diversity metrics for SASI and MAMI")
    print("========================================== SASI ===========================================")
    compute_diversity(f"../data/collab-rec-2026/llm-results/{model_name}/sasi/{model_name}_sasi.json", method="sasi")

    for rs in REJECTION_STRATEGIES:
        file_name = f"../data/collab-rec-2026/llm-results/{model_name}/mami/{model_name}_{rs}_10_rounds_fewshot.json"
        print(f"========================================== MASI {rs} ===========================================")
        compute_diversity(file_name, method="masi", round_nr=1)
        print(f"========================================== MAMI {rs} ===========================================")
        print(f"Processing {file_name}")
        compute_diversity(file_name, method="mami")
        compute_diversity_with_early_stopping(model_name, rs)


if __name__ == "__main__":
    main("gemma-12b")

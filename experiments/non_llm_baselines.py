import json
import pandas as pd

import random
from pyparsing import results

from random_seeds import SEEDS

from experiments.helpers import load_queries, load_kb

POP_CATEGORIES = ["Low", "Medium", "High"]


def get_random_pop_category(seed):
    random.seed(seed)
    return random.choice(POP_CATEGORIES)


def generate_random_cities(k=10, seed=42, kb_cities: list = None):
    """
    Generate a random list of k cities from the given kb_cities list.

    Parameters:
    - k (int): Number of cities to sample.
    - seed (int): Random seed for reproducibility.
    - kb_cities (list): List of available cities to sample from.

    Returns:
    - list: A list of k randomly sampled cities.

    Raises:
    - ValueError: If kb_cities is None or if k > len(kb_cities).
    """
    if kb_cities is None:
        raise ValueError("kb_cities must be provided.")

    if k > len(kb_cities):
        raise ValueError(f"k ({k}) cannot be greater than the number of available cities ({len(kb_cities)}).")

    random.seed(seed)
    return random.sample(kb_cities, k)


def generate_random_baseline():
    input_queries = load_queries("../data/collab-rec-2026/input-data/input_queries.json")
    kb = load_kb("../data/collab-rec-2026/input-data/kb/eu-cities-database_2.csv")
    kb_cities = list(set(kb['city'].tolist()))
    results = []
    for (q, seed) in zip(input_queries, SEEDS):
        random_cities_list = generate_random_cities(k=10, seed=seed, kb_cities=kb_cities)
        results.append({
            "query": q,
            "random_baseline": random_cities_list
        })

    # Save to JSON
    with open("../data/collab-rec-2026/llm-results/non_llm_baselines/random_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def generate_randomPop_recs(k=10, seed=4, kb: pd.DataFrame = None, pop_category="High"):
    if kb is None:
        raise ValueError("kb must be provided.")

    # filter by pop_category
    kb = kb.loc[kb['popularity'].str.lower() == pop_category.lower()]
    kb_cities = kb["city"].unique().tolist()
    print(f"Pop category: {pop_category}")
    print(f"Number of cities: {len(kb_cities)}")

    # Adjust k if there aren't enough cities
    actual_k = min(k, len(kb_cities))

    if actual_k == 0:
        return []

    random.seed(seed)
    return random.sample(kb_cities, actual_k)


def generate_topk_pop_recs(k=10, kb: pd.DataFrame = None):
    if kb is None:
        raise ValueError("kb must be provided.")

    # Expect a numeric popularity score
    if "pop_score" not in kb.columns:
        raise ValueError("kb must contain 'popularity_score'")

    kb = kb.drop_duplicates(subset=["city"])
    kb = kb.sort_values("pop_score", ascending=False)

    return kb["city"].head(k).tolist()


def generate_top_pop_baseline():
    kb = load_kb(
        "../data/collab-rec-2026/input-data/kb/eu-cities-database_2.csv"
    )[["city", "popularity"]]

    kb.drop_duplicates(inplace=True, ignore_index=True)
    pop_scores = pd.read_csv("../data/collab-rec-2026/input-data/kb/popularity_scores.csv")[
        ["city", "weighted_pop_score"]]
    kb = kb.merge(pop_scores, on="city", how="right")
    kb.rename(columns={"weighted_pop_score": "pop_score"}, inplace=True)
    kb.sort_values(by="city", ignore_index=True)

    input_queries = load_queries(
        "../data/collab-rec-2026/input-data/input_queries.json"
    )

    results = []
    topk_cities = generate_topk_pop_recs(k=10, kb=kb)

    for q in input_queries:
        results.append({
            "query": q,
            "top_pop_baseline": topk_cities
        })

    with open(
            "../data/collab-rec-2026/llm-results/non_llm_baselines/top_pop_baseline_results.json",
            "w"
    ) as f:
        json.dump(results, f, indent=2)


def generate_retrieval_based_baseline():
    kb = load_kb(
        "../data/collab-rec-2026/input-data/kb/merged_listing.csv"
    )[["city", "weighted_pop_score"]]
    input_queries = load_queries(
        "../data/collab-rec-2026/input-data/input_queries.json"
    )
    llama_queries = load_queries(
        "../data/collab-rec-2026/input-data/Llama3Point2Vision90B_generated_queries.json"
    )
    input_ids = {q["config_id"] for q in input_queries}
    filtered_llama_queries = [
        q for q in llama_queries if q.get("config_id") in input_ids
    ]

    # Create a mapping of config_id to query details
    query_map = {q["config_id"]: q for q in input_queries}

    results = []
    # retrieval and ranking
    for q in filtered_llama_queries:
        config_id = q["config_id"]
        cities = q["city"]

        # Rank cities by popularity
        ranked_cities = {}
        for city in cities:
            try:
                popularity = float(kb.loc[kb["city"] == city]["weighted_pop_score"].values[0])
                ranked_cities[city] = popularity
            except IndexError:
                print(f"City {city} not found in KB")

        ranked_cities = dict(sorted(ranked_cities.items(), key=lambda item: item[1], reverse=True))
        ranked_cities_list = list(ranked_cities.keys())[:10]  # Top 10

        # Get the original query details
        query_details = query_map.get(config_id, {})

        results.append({
            "query": {
                "config_id": config_id,
                "filters": query_details.get("filters", {}),
                "query": query_details.get("query", "")
            },
            "retrieval_based_baseline": ranked_cities_list
        })

    # Save to JSON
    with open(
            "../data/collab-rec-2026/llm-results/non_llm_baselines/retrieval_based_baseline_results.json",
            "w"
    ) as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} retrieval-based baseline results")
    return results


if __name__ == "__main__":
    generate_retrieval_based_baseline()
    # generate_top_pop_baseline()
    # generate_random_baseline()

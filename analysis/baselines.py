import json
import pandas as pd

from analysis.diversity import exposure_gini, exposure_normalized_entropy, gini_index
from analysis.compute_sigtests import calculate_match_with_filters
from constants import CITIES
from collections import Counter


def baseline_analysis(col_name: str = "random_baseline", data: list = None):
    all_cities = CITIES

    scores = 0
    counter = Counter()
    recommended_cities = []
    results = []
    for query_result in data:
        query = query_result["query"]

        counter.update(query_result[col_name])
        score = calculate_match_with_filters(offer=query_result[col_name],
                                             filters=query["filters"])
        scores += score
        results.append({"query_id": query_result["query"]["config_id"],
                        "filters": query["filters"],
                        "recommendations": query_result[col_name], "success_score": score})
        recommended_cities.append(query_result[col_name])
    avg_score = scores / len(data)
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"../data/collab-rec-2026/analysis/{col_name}_results.csv", index=False)
    print(f"Average score: {avg_score} for baseline with {len(data)} queries")
    recommendations = [
        r[col_name] for r in data
    ]
    entropy = exposure_normalized_entropy(recommendations, all_cities)
    unique_cities = set(sum(recommendations, []))
    coverage = len(unique_cities)
    print("Coverage: ", coverage, " / ", len(all_cities))
    print("Entropy: ", entropy)


def top_pop_analysis():
    data = json.load(open("../data/collab-rec-2026/llm-results/non_llm_baselines/top_pop_baseline_results.json"))
    print(f"Loaded {len(data)} queries")

    baseline_analysis(col_name="top_pop_baseline", data=data)


def random_baseline_analysis():
    data = json.load(open("../data/collab-rec-2026/llm-results/non_llm_baselines/random_baseline_results.json"))
    print(f"Loaded {len(data)} queries")
    baseline_analysis(col_name="random_baseline", data=data)


def retrieval_based_analysis():
    data = json.load(open("../data/collab-rec-2026/llm-results/non_llm_baselines/retrieval_based_baseline_results.json"))
    print(f"Loaded {len(data)} queries")
    baseline_analysis(col_name="retrieval_based_baseline", data=data)


def main():
    print("=" * 80)
    print("Computing Baselines")
    print("=" * 80)
    # retrieval_based_analysis()
    print("Top Pop")
    top_pop_analysis()
    print("Random")
    random_baseline_analysis()


if __name__ == "__main__":
    main()

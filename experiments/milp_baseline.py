import argparse
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pulp

from experiments.helpers import load_kb, load_queries
from src.k_base.context_retrieval import ContextRetrieval

Filters = Mapping[str, Any]


# Keep filter groups aligned with relevance computation logic.
AGENT_FILTER_MAPPING = {
    "personalization": frozenset({"budget", "month", "interests"}),
    "sustainability": frozenset({"aqi", "walkability", "seasonality"}),
    "popularity": frozenset({"popularity"}),
}


class MILPBaseline:
    def __init__(self, kb: pd.DataFrame, k: int = 10):
        if "city" not in kb.columns:
            raise ValueError("kb must contain a 'city' column")
        self.retriever = ContextRetrieval()
        self.k = k
        self.kb = kb.drop_duplicates(subset=["city"])

    def solve_query(self, filters: Filters) -> tuple[list[str], float]:
        all_cities = self.kb["city"].dropna().astype(str).unique().tolist()
        if not all_cities:
            return [], 0.0

        # Compute filter matches and category ratios for all cities
        city_scores = {}
        for city in all_cities:
            matched = self.retriever.match_city_with_filters(city, dict(filters))

            # Personalization: budget, month, interests
            pers_active = [k for k in AGENT_FILTER_MAPPING["personalization"] if k in filters]
            pers_ratio = sum(1 for k in pers_active if k in matched) / len(pers_active) if pers_active else 0.0

            # Sustainability: aqi, walkability, seasonality
            sust_active = [k for k in AGENT_FILTER_MAPPING["sustainability"] if k in filters]
            sust_ratio = sum(1 for k in sust_active if k in matched) / len(sust_active) if sust_active else 0.0

            # Popularity score: only use if popularity filter matches city's popularity column
            pop = 0.0
            pop_match = 0.0
            if "popularity" in filters:
                if "popularity" in matched:
                    pop = float(self.kb.loc[self.kb["city"] == city, "weighted_pop_score"].iloc[0]) if "weighted_pop_score" in self.kb.columns else 0.0
                    pop_match = 1.0

            # Utility = personalization + sustainability + popularity
            utility = pers_ratio + sust_ratio + pop

            # Overall success score: fraction of all filters matched
            total_filters = len(filters)
            matched_count = len(matched)
            overall_success = matched_count / total_filters if total_filters > 0 else 0.0

            city_scores[city] = (utility, overall_success)

        # Solve MILP
        k_select = min(self.k, len(all_cities))
        prob = pulp.LpProblem("Tourism_Opt", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("c", all_cities, cat="Binary")

        prob += pulp.lpSum([x[c] * city_scores[c][0] for c in all_cities])
        prob += pulp.lpSum([x[c] for c in all_cities]) == k_select

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        selected = [c for c in all_cities if pulp.value(x[c]) == 1.0]

        if not selected:
            return [], 0.0

        # Success score is mean overall match ratio across all selected cities
        success_score = float(np.mean([city_scores[c][1] for c in selected]))
        return selected, success_score


def compute_table_3_metrics(all_recommendations, catalog=None):
    flattened = [city for recs in all_recommendations for city in recs]
    if not flattened:
        return {"Gini": 0.0, "Entropy": 0.0, "Coverage": 0.0}

    if catalog is None:
        catalog = sorted(set(flattened))

    counts = pd.Series(flattened).value_counts().reindex(catalog, fill_value=0)
    total = int(counts.sum())
    if total == 0:
        return {"Gini": 0.0, "Entropy": 0.0, "Coverage": 0.0}

    sorted_counts = np.sort(counts.values.astype(float))
    n = len(sorted_counts)

    # Gini Index
    index = np.arange(1, n + 1)
    gini_num = float(np.sum((2 * index - n - 1) * sorted_counts))
    gini_den = float(n * np.sum(sorted_counts))
    gini = (gini_num / gini_den) if gini_den else 0.0

    # Entropy
    probs = counts.values.astype(float) / float(total)
    probs = probs[probs > 0]
    entropy = 0.0
    if n > 1 and len(probs) > 0:
        entropy_val = np.sum(probs * np.log(probs))
        entropy = float(-entropy_val / np.log(n))

    # Coverage
    coverage = float((counts > 0).sum() / n * 100) if n else 0.0

    return {"Gini": round(gini, 2), "Entropy": round(entropy, 2), "Coverage": round(coverage, 1)}


def run_milp_baseline(queries_path: str, kb_path: str, k: int = 10):
    queries = load_queries(file_path=queries_path)
    kb = load_kb(kb_path)

    milp = MILPBaseline(kb=kb, k=k)
    all_results = []
    scores = []

    print(f"Running {len(queries)} queries...")
    for query in queries:
        recs, score = milp.solve_query(query.get("filters", {}))
        all_results.append(recs)
        scores.append(score)

    catalog = kb["city"].dropna().astype(str).unique().tolist()
    metrics = compute_table_3_metrics(all_results, catalog=catalog)
    print(f"Table 3 Metrics: {metrics}")
    print(f"Mean success score: {np.mean(scores):.4f}")

    return metrics, scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="../data/collab-rec-2026/input-data/input_queries.json")
    parser.add_argument("--kb", default="../data/collab-rec-2026/input-data/kb/merged_listing.csv")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    run_milp_baseline(queries_path=args.queries, kb_path=args.kb, k=args.k)


if __name__ == "__main__":
    main()

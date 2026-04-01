import argparse
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pulp

from experiments.helpers import load_kb, load_queries
from src.k_base.context_retrieval import ContextRetrieval

Filters = Mapping[str, Any]


class MILPBaseline:
    def __init__(self, kb: pd.DataFrame, k: int = 10, retriever: ContextRetrieval | None = None):
        if "city" not in kb.columns:
            raise ValueError("kb must contain a 'city' column")

        self.retriever = retriever or ContextRetrieval()
        self.k = k
        self.kb = kb.copy()
        self.city_index = self.kb.drop_duplicates(subset=["city"]).set_index("city", drop=False)

    def _match_stats(self, city: str, filters: Filters) -> tuple[int, int, float]:
        matched = self.retriever.match_city_with_filters(city, dict(filters))
        total = len(filters)
        satisfied = len(matched)
        ratio = (satisfied / total) if total else 0.0
        return satisfied, total, ratio

    def _get_numeric_city_value(self, city: str, columns: Sequence[str], default: float) -> float:
        if city not in self.city_index.index:
            return default

        row = self.city_index.loc[city]
        for column in columns:
            if column in row.index and pd.notna(row[column]):
                try:
                    return float(row[column])
                except (TypeError, ValueError):
                    continue
        return default

    def _get_city_utility(self, city: str, filters: Filters, relevance_ratio: float) -> float:
        # Grounded utility = relevance + sustainability + inverse popularity
        sust = self._get_numeric_city_value(city, ["sustainability_score"], default=0)
        popularity = self._get_numeric_city_value(city, ["popularity_score", "weighted_pop_score", "pop_score"],
                                                  default=0)
        pop_inv = 1.0 - popularity
        return relevance_ratio + sust + pop_inv

    def solve_query(self, filters: Filters) -> tuple[list[str], float]:
        all_cities = self.kb["city"].dropna().astype(str).unique().tolist()
        if not all_cities:
            return [], 0.0

        # Pre-calculate stats for ALL cities in the 200-city catalog [cite: 574, 972]
        city_stats = {city: self._match_stats(city, filters) for city in all_cities}

        # DO NOT prune to only feasible_cities unless you want a strict "Hard Constraint" baseline.
        # Instead, let the solver maximize utility across the whole catalog. [cite: 477, 1182]
        candidate_cities = all_cities
        k_select = min(self.k, len(candidate_cities))

        prob = pulp.LpProblem("Tourism_Opt", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("c", candidate_cities, cat="Binary")

        # The objective function now balances relevance, sustainability, and popularity [cite: 108, 636]
        prob += pulp.lpSum(
            x[city] * self._get_city_utility(city, filters, city_stats[city][2])
            for city in candidate_cities
        )
        prob += pulp.lpSum(x[city] for city in candidate_cities) == k_select

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        selected = [city for city in candidate_cities if pulp.value(x[city]) == 1.0]

        if not selected:
            return [], 0.0

        # This success score will now vary based on the best available matches [cite: 1109, 1216]
        success_score = float(np.mean([city_stats[city][2] for city in selected]))
        return selected, success_score


def compute_table_3_metrics(
        all_recommendations: Sequence[Sequence[str]],
        catalog: Sequence[str] | None = None,
) -> dict[str, float]:
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
    index = np.arange(1, n + 1)
    gini_num = float(np.sum((2 * index - n - 1) * sorted_counts))
    gini_den = float(n * np.sum(sorted_counts))
    gini = (gini_num / gini_den) if gini_den else 0.0

    probs = counts.values.astype(float) / float(total)
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log(probs)) / np.log(n)) if n > 1 else 0.0

    coverage = float((counts > 0).sum() / n * 100) if n else 0.0
    return {"Gini": round(gini, 2), "Entropy": round(entropy, 2), "Coverage": round(coverage, 1)}


def run_milp_baseline(queries_path: str, kb_path: str, k: int = 10) -> tuple[dict[str, float], list[float]]:
    queries = load_queries(file_path=queries_path)
    kb = load_kb(kb_path)

    milp = MILPBaseline(kb=kb, k=k)
    all_results: list[list[str]] = []
    milp_success_scores: list[float] = []

    print(f"Running {len(queries)} queries...")
    for query in queries:
        filters = query.get("filters", {})
        recs, score = milp.solve_query(filters)
        all_results.append(recs)
        milp_success_scores.append(score)

    catalog = kb["city"].dropna().astype(str).unique().tolist()
    table_3 = compute_table_3_metrics(all_results, catalog=catalog)
    print(f"Table 3 Metrics: {table_3}")
    print("mean success scores:", np.mean(milp_success_scores))

    return table_3, milp_success_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="../data/collab-rec-2026/input-data/input_queries.json")
    parser.add_argument("--kb", default="../data/collab-rec-2026/input-data/kb/merged_listing.csv")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    run_milp_baseline(queries_path=args.queries, kb_path=args.kb, k=args.k)


if __name__ == "__main__":
    main()

import numpy as np
from typing import List, Dict, Any, Set
import logging

from k_base.context_retrieval import ContextRetrieval

logger = logging.getLogger(__name__)


def compute_agent_success_scores(
        candidate_cities: List[str],
        filters: Dict[str, Any],
        retriever,  # Represents the external Knowledge Base (KB)
) -> float:
    """
    Calculates the average proportion of filters matched per candidate [5].

    Args:
        candidate_cities: List of cities recommended by agent
        filters: User filter constraints
        retriever: Knowledge base retriever with get_city_metadata or match_city_with_filters method

    Returns:
        Average success score across all candidates
    """
    if not candidate_cities or not filters:
        logger.debug(f"Returning 0.0: candidate_cities={bool(candidate_cities)}, filters={bool(filters)}")
        return 0.0

    num_filters = len(filters)

    logger.debug(f"Computing success scores for {len(candidate_cities)} cities with {num_filters} filters")
    logger.debug(f"Filters: {filters}")

    total_rel_score = 0
    for city in candidate_cities:
        matched_filters = retriever.match_city_with_filters(city, filters)
        unmatched = [key for key in filters.keys() if key not in matched_filters.keys()]
        rel_score = len(matched_filters) / len(filters)
        total_rel_score += rel_score
    return total_rel_score / len(candidate_cities)


def compute_hallucination_rate(
        candidate_cities: List[str],
        catalog: List[str],
        rejected_cities_globally: Set[str] = None
) -> float:
    def _normalize_city_list(items: List[Any]) -> List[str]:
        normalized = []
        for item in items or []:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, dict):
                city_name = item.get("city") or item.get("name")
                if isinstance(city_name, str):
                    normalized.append(city_name)
        return normalized

    candidate_cities = _normalize_city_list(candidate_cities)
    if not candidate_cities:
        return 0.0

    if rejected_cities_globally is None:
        rejected_cities_globally = set()

    catalog_set = set(_normalize_city_list(catalog))

    valid_cities = [
        c for c in candidate_cities
        if c in catalog_set and c not in rejected_cities_globally
    ]

    hit_rate = len(valid_cities) / len(candidate_cities)

    # Paper-faithful definition
    hallucination_rate = 1.0 - hit_rate

    return hallucination_rate


def compute_agent_reliability(curr_list: List[str], prev_list: List[str], prev_coll_offer: List[str], k: int = 10):
    """
        Computes the weighted reliability score
        """
    def _normalize_city_list(items: List[Any]) -> List[str]:
        normalized = []
        for item in items or []:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, dict) and "city" in item:
                normalized.append(item["city"])
        return normalized

    curr_list = _normalize_city_list(curr_list)
    prev_list = _normalize_city_list(prev_list)

    if not prev_list:
        # No prior list to compare against; treat as fully reliable by default.
        return 1.0

    if prev_coll_offer is None:
        prev_coll_offer = []

    base_drop_penalty = k
    base_new_penalty = k

    score = 0

    prev_set = set(prev_list)
    curr_set = set(curr_list)

    for city in prev_set & curr_set:
        old_rank = prev_list.index(city)
        new_rank = curr_list.index(city)
        # weight = 1 / (old_rank + 1)  # rank 1 has weight 1.0, rank 10 has 0.1
        score += abs(new_rank - old_rank)

    for city in prev_set - curr_set:
        old_rank = prev_list.index(city)
        # weight = 1 / (old_rank + 1)
        score += base_drop_penalty

    for city in curr_set - prev_set:
        if city in prev_coll_offer:
            offer_rank = prev_coll_offer.index(city)
        else:
            offer_rank = np.inf
        new_rank = curr_list.index(city)
        # weight = 1 / (new_rank + 1)
        score += min(base_new_penalty, abs(new_rank - offer_rank))

    worst_case_score = len(prev_list) * (base_drop_penalty + base_new_penalty)
    if worst_case_score == 0:
        return 1.0
    reliability = 1 - (score / worst_case_score)
    return max(reliability, 0)


def update_city_scores(cumulative_scores, agent_results, ablated_component: str | None = None):
    """
    Update city scores based on agent results.

    Args:
        cumulative_scores: Dictionary mapping cities to scores
        agent_results: List of agent result dictionaries
        ablated_component: Component to ablate from scoring. Options:
            - None: use all components (r + d - h)
            - "hallucination": omit hallucination (r + d)
            - "reliability": omit reliability (r - h)
            - "success": omit success (d - h)
            - "rank": omit rank weighting (use uniform weight)

    Returns:
        Updated cumulative_scores dictionary
    """
    for agent in agent_results:
        h = agent['hallucination']  # ∈ [0, 1]
        r = agent['success']  # ∈ [0, 1]
        d = agent['reliability']  # ∈ [0, 1]

        # Compute agent weight based on ablation setting
        if ablated_component == "hallucination":
            # Omit hallucination penalty
            agent_weight = r + d
        elif ablated_component == "reliability":
            # Omit reliability
            agent_weight = r - h
        elif ablated_component == "success":
            # Omit success/relevance
            agent_weight = d - h
        else:
            # Default: use all components (paper-defined agent weight)
            agent_weight = r + d - h

        if agent_weight <= 0:
            continue

        for index, city in enumerate(agent['candidates']):
            # Apply rank weighting unless ablating rank
            if ablated_component == "rank":
                rank_weight = 1.0  # Uniform weight
            else:
                rank_weight = 1.0 / (index + 1)

            contribution = rank_weight * agent_weight
            cumulative_scores[city] = cumulative_scores.get(city, 0.0) + contribution

    return cumulative_scores

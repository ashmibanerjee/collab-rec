"""
Unified module for computing relevance scores across all experiment types:
MAMI, MASI, SASI, and baselines (top-pop, random).

This module provides:
- Individual relevance computation per filter category (personalization, sustainability, popularity)
- Overall relevance computation
- Support for multi-agent (MAMI/MASI) and single-agent (SASI) systems
- Support for non-LLM baselines
"""

import json
import ast
import pandas as pd
from typing import List, Dict, Any, Optional

from adk.agents.moderator_utils.compute_scores import (
    compute_agent_reliability,
    compute_hallucination_rate,
    compute_agent_success_scores
)
from constants import CITIES
from experiments.helpers import load_queries, get_pop_level
from k_base.context_retrieval import ContextRetrieval


# ==================== Constants ====================

AGENT_FILTER_MAPPING = {
    'personalization': {"budget", "month", "interests"},
    'sustainability': {"aqi", "walkability", "seasonality"},
    'popularity': {"popularity"},
    'moderator': {"budget", "month", "interests", "aqi", "walkability", "seasonality", "popularity"}
}


# ==================== Core Relevance Functions ====================

def calculate_match_with_filters(offer: List[str], filters: Dict[str, str],
                                 retriever: Optional[ContextRetrieval] = None) -> float:
    """
    Calculate the average relevance score for a list of cities against filters.

    Args:
        offer: List of city names
        filters: Dictionary of filter criteria
        retriever: Optional ContextRetrieval instance (created if not provided)

    Returns:
        Average relevance score (0.0 to 1.0)
    """
    if retriever is None:
        retriever = ContextRetrieval()

    if not offer or not filters:
        return 0.0

    total_rel_score = 0
    for city in offer:
        matched_filters = retriever.match_city_with_filters(city, filters)
        rel_score = len(matched_filters) / len(filters)
        total_rel_score += rel_score

    return total_rel_score / len(offer)


def compute_individual_relevance(query_filters: Dict[str, Any],
                                 offer: List[str],
                                 retriever: Optional[ContextRetrieval] = None) -> Dict[str, float]:
    """
    Calculate relevance scores for an offer against different filter categories.

    Args:
        query_filters: Dictionary of all query filters
        offer: List of candidate cities
        retriever: Optional ContextRetrieval instance

    Returns:
        Dict mapping filter category names to relevance scores:
        - 'personalization': relevance for personalization filters
        - 'sustainability': relevance for sustainability filters
        - 'popularity': relevance for popularity filters
        - 'all': overall relevance across all filters
    """
    if retriever is None:
        retriever = ContextRetrieval()

    # Get filters relevant to each agent type
    filter_categories = {
        'personalization': {k: v for k, v in query_filters.items() if k in AGENT_FILTER_MAPPING['personalization']},
        'sustainability': {k: v for k, v in query_filters.items() if k in AGENT_FILTER_MAPPING['sustainability']},
        'popularity': {k: v for k, v in query_filters.items() if k in AGENT_FILTER_MAPPING['popularity']},
        'all': query_filters
    }
    if filter_categories["sustainability"] == {}:
        filter_categories["sustainability"] = {"aqi": "great", "walkability": "great", "seasonality": "low"}

    # Calculate relevance for each category
    relevance_scores = {}
    for category, filters in filter_categories.items():
        if filters:  # Only calculate if filters exist
            relevance_scores[category] = calculate_match_with_filters(offer, filters, retriever)
        else:
            relevance_scores[category] = 0.0

    return relevance_scores


def compute_row_relevance(row, retriever: Optional[ContextRetrieval] = None):
    """
    Apply relevance computation to a single DataFrame row.

    Args:
        row: DataFrame row with 'filters' and 'candidates' columns
        retriever: Optional ContextRetrieval instance

    Returns:
        Dictionary of relevance scores per category
    """
    # Parse filters from JSON string
    if isinstance(row['filters'], str):
        filters = json.loads(row['filters'])
    else:
        filters = row['filters']

    # Parse candidates from string representation of list
    if isinstance(row['candidates'], str):
        try:
            candidates = ast.literal_eval(row['candidates'])
        except:
            candidates = json.loads(row['candidates'])
    else:
        candidates = row['candidates']

    return compute_individual_relevance(filters, candidates, retriever)


# ==================== MAMI Processing ====================

def compute_mami_scores(queries_data: List[Dict[str, Any]],
                       output_csv: str,
                       method: str = "mami") -> pd.DataFrame:
    """
    Compute agent scores and relevance for MAMI/MASI experiments.

    Args:
        queries_data: List of queries with multi-agent responses
        output_csv: Path to save output CSV
        method: "mami" for full multi-agent, "masi" for round 1 only

    Returns:
        DataFrame with scores and relevance metrics
    """
    records = []
    retriever = ContextRetrieval()

    for query in queries_data:
        query_id = query.get("query_id", None)
        filters = query.get("query_details", {}).get("filters", {})
        responses = query.get("response", [])

        # Build lookup structure for previous rounds
        responses_by_round = {}
        for response in responses:
            round_nr = response.get("round_number", 0)
            agent_name = response.get("agent_role", "unknown")
            if round_nr not in responses_by_round:
                responses_by_round[round_nr] = {}
            responses_by_round[round_nr][agent_name] = response

        # Get rejected cities if available
        rejected_cities = set(query.get("query_details", {}).get("rejected_cities", []))

        for response in responses:
            agent_name = response.get("agent_role", "unknown")
            round_nr = response.get("round_number", 0)
            candidates = response.get("candidates", [])
            time_taken = response.get("time_taken", 0)
            total_token_count = response.get("total_token_count", 0)

            # Filter based on method
            if method == "masi" and round_nr != 1:
                continue

            # Compute success score
            try:
                success_score = compute_agent_success_scores(
                    candidates, filters, retriever=retriever
                )
            except Exception as e:
                print(f"Error computing success score for query {query_id}, agent {agent_name}, round {round_nr}: {e}")
                success_score = None

            # Compute hallucination rate
            try:
                hallucination_rate = compute_hallucination_rate(
                    candidate_cities=candidates,
                    catalog=CITIES,
                    rejected_cities_globally=rejected_cities
                )
            except Exception as e:
                print(f"Error computing hallucination rate for query {query_id}, agent {agent_name}, round {round_nr}: {e}")
                hallucination_rate = None

            # Compute reliability score (only for round 2+)
            reliability_score = None
            if round_nr > 1:
                try:
                    previous_list = []
                    if round_nr - 1 in responses_by_round and agent_name in responses_by_round[round_nr - 1]:
                        previous_list = responses_by_round[round_nr - 1][agent_name].get("candidates", [])

                    collective_offer = []
                    if round_nr - 1 in responses_by_round and "moderator" in responses_by_round[round_nr - 1]:
                        collective_offer = responses_by_round[round_nr - 1]["moderator"].get("candidates", [])

                    reliability_score = compute_agent_reliability(
                        curr_list=candidates,
                        prev_list=previous_list,
                        prev_coll_offer=collective_offer,
                        k=10
                    )
                except Exception as e:
                    print(f"Error computing reliability for query {query_id}, agent {agent_name}, round {round_nr}: {e}")
                    reliability_score = None

            # Compute individual relevance scores
            try:
                relevance_scores = compute_individual_relevance(filters, candidates, retriever)
            except Exception as e:
                print(f"Error computing relevance for query {query_id}, agent {agent_name}, round {round_nr}: {e}")
                relevance_scores = {
                    'personalization': None,
                    'sustainability': None,
                    'popularity': None,
                    'all': None
                }

            # Convert filters and candidates to JSON strings for CSV storage
            filters_str = json.dumps(filters)
            candidates_str = json.dumps(candidates)

            records.append({
                "query_id": query_id,
                "filters": filters_str,
                "agent_name": agent_name,
                "round_nr": round_nr,
                "candidates": candidates_str,
                "success_score": success_score,
                "reliability_score": reliability_score,
                "hallucination_rate": hallucination_rate,
                "time_taken": time_taken,
                "total_token_count": total_token_count,
                "relevance_personalization": relevance_scores.get('personalization'),
                "relevance_sustainability": relevance_scores.get('sustainability'),
                "relevance_popularity": relevance_scores.get('popularity'),
                "overall_relevance": relevance_scores.get('all')
            })

    df = pd.DataFrame(records)
    df["pop_level"] = df["query_id"].apply(get_pop_level)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} records to {output_csv}")

    return df


# ==================== SASI Processing ====================

def compute_sasi_scores(queries_data: List[Dict[str, Any]],
                       output_csv: str) -> pd.DataFrame:
    """
    Compute scores and relevance for SASI (single-agent) experiments.

    Args:
        queries_data: List of queries with single-agent responses
        output_csv: Path to save output CSV

    Returns:
        DataFrame with scores and relevance metrics
    """
    records = []
    retriever = ContextRetrieval()

    for query in queries_data:
        query_id = query.get("query_id", None)
        filters = query.get("query_details", {}).get("filters", {})
        rejected_cities = set(query.get("query_details", {}).get("rejected_cities", []))

        # SASI has only one response per query
        responses = query.get("response") or []
        response = responses[0] if responses else {}

        agent_name = response.get("agent_role", "sasi")
        candidates = response.get("candidates", [])
        if not candidates:
            candidates = response.get("cities", [])
        time_taken = response.get("time_taken", 0)
        total_token_count = response.get("total_token_count", 0)
        round_nr = 1  # SASI is always single round

        # Compute success score
        try:
            success_score = compute_agent_success_scores(
                candidates, filters, retriever=retriever
            )
        except Exception as e:
            print(f"Error computing success score for query {query_id}: {e}")
            success_score = None

        # Compute hallucination rate
        try:
            hallucination_rate = compute_hallucination_rate(
                candidate_cities=candidates,
                catalog=CITIES,
                rejected_cities_globally=rejected_cities
            )
        except Exception as e:
            print(f"Error computing hallucination rate for query {query_id}: {e}")
            hallucination_rate = None

        # Compute individual relevance scores
        try:
            relevance_scores = compute_individual_relevance(filters, candidates, retriever)
        except Exception as e:
            print(f"Error computing relevance for query {query_id}: {e}")
            relevance_scores = {
                'personalization': None,
                'sustainability': None,
                'popularity': None,
                'all': None
            }

        # Convert filters and candidates to JSON strings
        filters_str = json.dumps(filters)
        candidates_str = json.dumps(candidates)

        records.append({
            "query_id": query_id,
            "filters": filters_str,
            "agent_name": agent_name,
            "round_nr": round_nr,
            "candidates": candidates_str,
            "success_score": success_score,
            "reliability_score": None,  # N/A for single round
            "hallucination_rate": hallucination_rate,
            "time_taken": time_taken,
            "total_token_count": total_token_count,
            "relevance_personalization": relevance_scores.get('personalization'),
            "relevance_sustainability": relevance_scores.get('sustainability'),
            "relevance_popularity": relevance_scores.get('popularity'),
            "overall_relevance": relevance_scores.get('all')
        })

    df = pd.DataFrame(records)
    df["pop_level"] = df["query_id"].apply(get_pop_level)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} records to {output_csv}")

    return df


# ==================== Baseline Processing ====================

def process_baseline_results(json_file_path: str,
                            baseline_key: str,
                            output_csv: str) -> pd.DataFrame:
    """
    Process baseline results (top-pop or random) and compute relevance scores.

    Args:
        json_file_path: Path to the baseline JSON file
        baseline_key: Key name for baseline results ('top_pop_baseline' or 'random_baseline')
        output_csv: Path to save output CSV

    Returns:
        DataFrame with query_id, filters, candidates, and relevance scores
    """
    # Load JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    retriever = ContextRetrieval()
    results = []

    for entry in data:
        query_info = entry.get('query', {})
        filters = query_info.get('filters', {})
        candidates = entry.get(baseline_key, [])
        query_id = query_info.get('config_id', '')
        rejected_cities = set(query_info.get('rejected_cities', []))

        # Compute success score
        try:
            success_score = compute_agent_success_scores(
                candidates, filters, retriever=retriever
            )
        except Exception as e:
            print(f"Error computing success score for baseline query {query_id}: {e}")
            success_score = None

        # Compute hallucination rate
        try:
            hallucination_rate = compute_hallucination_rate(
                candidate_cities=candidates,
                catalog=CITIES,
                rejected_cities_globally=rejected_cities
            )
        except Exception as e:
            print(f"Error computing hallucination rate for baseline query {query_id}: {e}")
            hallucination_rate = None

        # Compute relevance scores for all filter categories
        try:
            relevance_scores = compute_individual_relevance(filters, candidates, retriever)
        except Exception as e:
            print(f"Error computing relevance for baseline query {query_id}: {e}")
            relevance_scores = {
                'personalization': None,
                'sustainability': None,
                'popularity': None,
                'all': None
            }

        # Create result record
        result = {
            'query_id': query_id,
            'filters': json.dumps(filters),
            'agent_name': baseline_key.replace('_baseline', ''),
            'round_nr': 1,
            'candidates': json.dumps(candidates),
            'success_score': success_score,
            'reliability_score': None,  # N/A for baselines
            'hallucination_rate': hallucination_rate,
            'time_taken': None,
            'total_token_count': None,
            'relevance_personalization': relevance_scores.get('personalization'),
            'relevance_sustainability': relevance_scores.get('sustainability'),
            'relevance_popularity': relevance_scores.get('popularity'),
            'overall_relevance': relevance_scores.get('all')
        }
        results.append(result)

    df = pd.DataFrame(results)
    df["pop_level"] = df["query_id"].apply(get_pop_level)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} baseline records to {output_csv}")

    return df


# ==================== Main Processing Functions ====================

def process_mami(model_name: str, rejection_strategy: str = "aggressive") -> pd.DataFrame:
    """
    Process MAMI (Multi-Agent Multi-Iteration) results.

    Args:
        model_name: Name of the model (e.g., "gemini")
        rejection_strategy: Rejection strategy ("aggressive" or "majority")

    Returns:
        DataFrame with all computed metrics
    """
    print(f"\n{'='*80}")
    print(f"Processing MAMI ({model_name}, {rejection_strategy})")
    print(f"{'='*80}")

    input_file = f"../data/collab-rec-2026/llm-results/{model_name}/mami/{model_name}_{rejection_strategy}_10_rounds_fewshot.json"
    output_file = f"../data/collab-rec-2026/analysis/{model_name}_mami_{rejection_strategy}_scores_with_relevance.csv"

    print(f"Loading data from: {input_file}")
    queries_data = load_queries(input_file)
    print(f"Loaded {len(queries_data)} queries")

    df = compute_mami_scores(queries_data, output_file, method="mami")

    print_summary_statistics(df, "MAMI")

    return df


def process_masi(model_name: str, rejection_strategy: str = "aggressive") -> pd.DataFrame:
    """
    Process MASI (Multi-Agent Single-Iteration) results.
    MASI uses the same data as MAMI but only round 1.

    Args:
        model_name: Name of the model (e.g., "gemini")
        rejection_strategy: Rejection strategy ("aggressive" or "majority")

    Returns:
        DataFrame with all computed metrics
    """
    print(f"\n{'='*80}")
    print(f"Processing MASI ({model_name}, {rejection_strategy})")
    print(f"{'='*80}")

    input_file = f"../data/collab-rec-2026/llm-results/{model_name}/mami/{model_name}_{rejection_strategy}_10_rounds_fewshot.json"
    output_file = f"../data/collab-rec-2026/analysis/{model_name}_masi_{rejection_strategy}_scores_with_relevance.csv"

    print(f"Loading data from: {input_file}")
    queries_data = load_queries(input_file)
    print(f"Loaded {len(queries_data)} queries")

    df = compute_mami_scores(queries_data, output_file, method="masi")

    print_summary_statistics(df, "MASI")

    return df


def process_sasi(model_name: str = "gemini") -> pd.DataFrame:
    """
    Process SASI (Single-Agent Single-Iteration) results.

    Args:
        model_name: Name of the model (e.g., "gemini")

    Returns:
        DataFrame with all computed metrics
    """
    print(f"\n{'='*80}")
    print(f"Processing SASI ({model_name})")
    print(f"{'='*80}")

    input_file = f"../data/collab-rec-2026/llm-results/{model_name}/sasi/{model_name}_sasi.json"
    output_file = f"../data/collab-rec-2026/analysis/{model_name}_sasi_scores_with_relevance.csv"

    print(f"Loading data from: {input_file}")
    queries_data = load_queries(input_file)
    print(f"Loaded {len(queries_data)} queries")

    df = compute_sasi_scores(queries_data, output_file)

    print_summary_statistics(df, "SASI")

    return df


def process_all_baselines(model_name: str = "gemini") -> Dict[str, pd.DataFrame]:
    """
    Process all baseline results: top-pop, random, and retrieval-based.

    Args:
        model_name: Name of the model (used for consistency in output naming)

    Returns:
        Dictionary with 'top_pop', 'random', and 'retrieval_based' DataFrames
    """
    print(f"\n{'='*80}")
    print("Processing Baselines")
    print(f"{'='*80}")

    base_path = "../data/collab-rec-2026/llm-results/non_llm_baselines"

    # Process top_pop baseline
    print("\nProcessing top_pop baseline...")
    top_pop_input = f"{base_path}/top_pop_baseline_results.json"
    top_pop_output = f"../data/collab-rec-2026/analysis/top_pop_baseline_scores_with_relevance.csv"
    top_pop_df = process_baseline_results(top_pop_input, "top_pop_baseline", top_pop_output)
    print_summary_statistics(top_pop_df, "Top-Pop Baseline")

    # Process random baseline
    print("\n" + "-"*80)
    print("Processing random baseline...")
    random_input = f"{base_path}/random_baseline_results.json"
    random_output = f"../data/collab-rec-2026/analysis/random_baseline_scores_with_relevance.csv"
    random_df = process_baseline_results(random_input, "random_baseline", random_output)
    print_summary_statistics(random_df, "Random Baseline")

    # Process retrieval-based baseline
    print("\n" + "-"*80)
    print("Processing retrieval-based baseline...")
    retrieval_input = f"{base_path}/retrieval_based_baseline_results.json"
    retrieval_output = f"../data/collab-rec-2026/analysis/retrieval_based_baseline_scores_with_relevance.csv"
    retrieval_df = process_baseline_results(retrieval_input, "retrieval_based_baseline", retrieval_output)
    print_summary_statistics(retrieval_df, "Retrieval-Based Baseline")

    return {"top_pop": top_pop_df, "random": random_df, "retrieval_based": retrieval_df}


# ==================== Utility Functions ====================

def print_summary_statistics(df: pd.DataFrame, experiment_name: str):
    """Print summary statistics for a processed DataFrame."""
    print(f"\n=== Summary Statistics: {experiment_name} ===")
    print(f"Total records: {len(df)}")
    print(f"Unique queries: {df['query_id'].nunique()}")
    print(f"Unique agents: {df['agent_name'].nunique()}")

    if 'round_nr' in df.columns:
        print(f"Rounds: {sorted(df['round_nr'].unique())}")

    # Relevance scores
    print(f"\n--- Relevance Scores ---")
    if 'overall_relevance' in df.columns:
        print(f"Overall relevance: {df['overall_relevance'].mean():.4f}")
    if 'relevance_personalization' in df.columns:
        print(f"Personalization relevance: {df['relevance_personalization'].mean():.4f}")
    if 'relevance_sustainability' in df.columns:
        print(f"Sustainability relevance: {df['relevance_sustainability'].mean():.4f}")
    if 'relevance_popularity' in df.columns:
        print(f"Popularity relevance: {df['relevance_popularity'].mean():.4f}")

    # Success scores
    if 'success_score' in df.columns:
        print(f"\n--- Success Scores ---")
        print(f"Average success score: {df['success_score'].mean():.4f}")

    # Hallucination rates
    if 'hallucination_rate' in df.columns:
        print(f"\n--- Hallucination Rates ---")
        print(f"Average hallucination rate: {df['hallucination_rate'].mean():.4f}")

    # Reliability (if applicable)
    if 'reliability_score' in df.columns and df['reliability_score'].notna().any():
        print(f"\n--- Reliability (rounds 2+) ---")
        reliability_df = df[df['reliability_score'].notna()]
        if not reliability_df.empty:
            print(f"Average reliability: {reliability_df['reliability_score'].mean():.4f}")


def process_all_experiments(model_name: str = "gemini",
                           rejection_strategy: str = "aggressive") -> Dict[str, pd.DataFrame]:
    """
    Process all experiment types: MAMI, MASI, SASI, and baselines.

    Args:
        model_name: Name of the model (e.g., "gemini")
        rejection_strategy: Rejection strategy for MAMI/MASI

    Returns:
        Dictionary with DataFrames for each experiment type
    """
    results = {}

    # Process SASI
    results['sasi'] = process_sasi(model_name)

    # Process MASI
    results['masi'] = process_masi(model_name, rejection_strategy)

    # Process MAMI
    results['mami'] = process_mami(model_name, rejection_strategy)

    # Process baselines
    baseline_results = process_all_baselines(model_name)
    results.update(baseline_results)

    print(f"\n{'='*80}")
    print("✓ All experiments processed successfully!")
    print(f"{'='*80}")

    return results


# ==================== Main Entry Point ====================

def main(model_name: str = "gemini",
         rejection_strategy: str = "aggressive",
         experiments: Optional[List[str]] = None):
    """
    Main entry point for computing relevance scores.

    Args:
        model_name: Name of the model (e.g., "gemini")
        rejection_strategy: Rejection strategy for MAMI/MASI
        experiments: List of experiments to process. If None, processes all.
                    Options: ['mami', 'masi', 'sasi', 'baselines', 'all']
    """
    if experiments is None or 'all' in experiments:
        results = process_all_experiments(model_name, rejection_strategy)
    else:
        results = {}

        if 'sasi' in experiments:
            results['sasi'] = process_sasi(model_name)

        if 'masi' in experiments:
            results['masi'] = process_masi(model_name, rejection_strategy)

        if 'mami' in experiments:
            results['mami'] = process_mami(model_name, rejection_strategy)

        if 'baselines' in experiments:
            baseline_results = process_all_baselines(model_name)
            results.update(baseline_results)

    return results


if __name__ == "__main__":
    # Example: Process all experiments
    # main("gemini", "aggressive", experiments=['all'])

    # Example: Process only specific experiments
    print("Computing relevance for gemini")
    main("smol-3b", "aggressive", experiments=["sasi", "masi", "mami"])
    # main("gemini", "aggressive", experiments=[ "mami"])
    # print("Computing relevance for claude")
    # main("claude", "aggressive", experiments=["mami"])
    # main("claude", "majority", experiments=["mami"])
    # print("" + "="*80)
    # print("Computing relevance for olmo-7b majority")
    # main("olmo-7b", "majority", experiments=["masi", "mami"])

    # Example: Process only baselines
    # main("gemini", "aggressive", experiments=['baselines'])
    # filters = {"popularity": "high", "budget": "medium", "interests": "Food", "month": "April"}
    # compute_individual_relevance(query_filters=filters,
    #                              offer= ["Paris", "Rome", "London", "Barcelona", "Amsterdam", "Prague", "Vienna", "Budapest", "Berlin", "Istanbul"],
    #                              retriever=None)

from typing import List, Dict, Any

import json

import pandas as pd

from adk.agents.moderator_utils.compute_scores import compute_agent_reliability, compute_hallucination_rate, \
    compute_agent_success_scores
from constants import CITIES
from experiments.helpers import load_queries, get_pop_level
from k_base.context_retrieval import ContextRetrieval
import argparse


def compute_agent_scores_per_query(queries_data: List[Dict[str, Any]], output_csv: str) -> pd.DataFrame:
    """
    Compute agent success scores, reliability, and hallucination rate for each query, agent, and round.

    Args:
        queries_data: List of queries, each with query_id and response list
        output_csv: Path to save the output CSV file

    Returns:
        DataFrame with columns: query_id, filters, agent_name, round_nr, candidates,
                                success_score, reliability_score, hallucination_rate
    """
    records = []
    retriever = ContextRetrieval()
    time_taken = 0
    total_token_count = 0
    for query in queries_data:
        query_id = query.get("query_id", None)
        filters = query.get("query_details", {}).get("filters", {})

        # Each query has a list of responses (one per agent per round)
        responses = query.get("response", [])

        # Build a lookup structure for previous rounds and collective offers
        # Group responses by round and agent
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
            # Compute success score
            try:
                success_score = compute_agent_success_scores(
                    candidates,
                    filters,
                    retriever=retriever,
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
                print(
                    f"Error computing hallucination rate for query {query_id}, agent {agent_name}, round {round_nr}: {e}")
                hallucination_rate = None

            # Compute reliability score
            reliability_score = None
            if round_nr > 1:  # Reliability only makes sense from round 2 onwards
                try:
                    # Get previous round's candidates for this agent
                    previous_list = []
                    if round_nr - 1 in responses_by_round and agent_name in responses_by_round[round_nr - 1]:
                        previous_list = responses_by_round[round_nr - 1][agent_name].get("candidates", [])

                    # Get collective offer from moderator in previous round
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
                    print(
                        f"Error computing reliability for query {query_id}, agent {agent_name}, round {round_nr}: {e}")
                    reliability_score = None

            # Convert filters dict to JSON string for CSV storage
            filters_str = json.dumps(filters)
            # Convert candidates list to JSON string for CSV storage
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
                "total_token_count": total_token_count
            })

    df = pd.DataFrame(records)
    df["pop_level"] = df["query_id"].apply(get_pop_level)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} records to {output_csv}")

    return df


def get_scores(model_name: str,
               output_file: str,
               rejection_strategy: str = "aggressive",
               rounds: int = 10,
               ablated_component: str = None,
               input_file: str = None) -> pd.DataFrame:
    """
    Main function to compute agent success scores, reliability, and hallucination rate, then save to CSV.

    Args:
        model_name (str): Name of the model
        rejection_strategy: Strategy name (e.g., "aggressive")
        rounds (int): Number of rounds
        ablated_component (str, optional): If specified, the component that was ablated (e.g., "hallucination", "reliability", "success", "rank")
        input_file (str, optional): Input file path
        output_file: Path to save the CSV file
    """
    if input_file is None:
        if ablated_component is None:
            file_path = f"../data/collab-rec-2026/llm-results/{model_name}/mami/{model_name}_{rejection_strategy}_{rounds}_rounds_fewshot.json"
        else:
            file_path = f"../data/collab-rec-2026/llm-results/{model_name}/mami/ablated/{model_name}_{rejection_strategy}_{rounds}_rounds_ablated_{ablated_component}.json"
    else:
        file_path = input_file
    # file_path = f"../data/collab-rec-2026/llm-results/{model_name}/mami/filtered_{model_name}_{rejection_strategy}_10_rounds_fewshot_multiplicative.json"

    print(f"Loading data from: {file_path}")
    output_data = load_queries(file_path)
    print(f"Loaded {len(output_data)} queries")

    # Compute success scores per query/agent/round
    df = compute_agent_scores_per_query(output_data, output_csv=output_file)

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total records: {len(df)}")
    print(f"Unique queries: {df['query_id'].nunique()}")
    print(f"Unique agents: {df['agent_name'].nunique()}")
    print(f"Agents: {sorted(df['agent_name'].unique())}")
    print(f"Round numbers: {sorted(df['round_nr'].unique())}")

    print(f"\n=== Average Success Score by Agent ===")
    print(df.groupby('agent_name')['success_score'].mean().sort_values(ascending=False))

    print(f"\n=== Average Success Score by Round ===")
    print(df.groupby('round_nr')['success_score'].mean())

    print(f"\n=== Average Reliability Score by Agent (rounds 2+) ===")
    reliability_by_agent = df[df['round_nr'] > 1].groupby('agent_name')['reliability_score'].mean()
    print(reliability_by_agent.sort_values(ascending=False))

    print(f"\n=== Average Hallucination Rate by Agent ===")
    print(df.groupby('agent_name')['hallucination_rate'].mean().sort_values(ascending=True))

    print(f"\n=== Average Hallucination Rate by Round ===")
    print(df.groupby('round_nr')['hallucination_rate'].mean())

    return df


def main(model_name: str, rejection_strategy: str = "aggressive", rounds: int = 10):
    output_file = f"../data/collab-rec-2026/analysis/{model_name}_mami_{rejection_strategy}_{rounds}_rounds_scores.csv"
    df = get_scores(model_name=model_name, output_file=output_file, rejection_strategy=rejection_strategy, rounds=rounds)
    print(f"saved in {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scoring functions.")
    parser.add_argument("--models",
                        nargs="+", default=["gemini"],
                        help="List of model names to use.")
    parser.add_argument("--rejection_strategy",
                        nargs="+", default=["majority"],
                        help="List of rejection strategies to use.")
    parser.add_argument("--rounds",
                        nargs="+", default=[10],
                        help="Number of rounds")

    args = parser.parse_args()
    # main("gemini", "majority")
    main(args.models[0], args.rejection_strategy[0], args.rounds[0])

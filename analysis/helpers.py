from typing import Optional

import json
import pandas as pd


def filter_by_agent_role(agent_response: list, agent_role: str):
    filtered = [
        r for r in agent_response
        if r.get("agent_role") == agent_role
    ]
    return filtered


def get_all_agent_role_responses(queries_data: list, agent_role: str = None) -> list:
    responses = []
    for query in queries_data:
        query_response = query["response"]  # this has all the responses for all rounds by all agents
        query_details = query["query_details"]
        filtered_responses = filter_by_agent_role(query_response, agent_role)
        responses.append({
            "query_details": query_details,
            "filtered_responses": filtered_responses
        })
    return responses


def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def get_agent_responses(data: list, agent_role: str, roundnr: int, all_responses: Optional[bool] = False):
    responses = []
    for query in data:
        if all_responses:
            if (query.get("agent_role") == agent_role) and (query.get("round_number") <= roundnr):
                responses.append(query)
        else:
            if (query.get("agent_role") == agent_role) and (query.get("round_number") == roundnr):
                responses.append(query)
    return responses


def get_mod_scores_pop_level(df, pop_level):
    df_pop_level = df.loc[df["pop_level"] == pop_level]
    moderator_df = df_pop_level.loc[(df_pop_level["agent_name"] == "moderator") & (df_pop_level["round_nr"] == 10)]
    return moderator_df.success_score.tolist()


def get_distri_by_pop_level(df):
    pop_levels = ["low", "medium", "high"]
    scores = {"low": [], "medium": [], "high": []}
    for pop_level in pop_levels:
        score = get_mod_scores_pop_level(df, pop_level)
        scores[pop_level] = score
    return scores

import json, re, ast
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Callable
from src.data_directories import *
from src.multi_ctrs.agents.agent import LLMAgent
from src.multi_ctrs.moderators.moderator import Moderator
from src.llm_setup.models import Gemini2Flash, Gemini1Point5Pro, Claude3Point5Sonnet, Llama3Point2Vision90B, GPT4, \
    GPT4o, GPTo1Mini, Claude3Point7Sonnet, DeepSeekReasoner, GPTo4Mini
from src.multi_ctrs.retrieval import Retrieval
from src.multi_ctrs.agents.helpers import *
from src.constants import CITIES
from src.multi_ctrs.experiments.setup import ExperimentSetup

EXPERIMENTS = ["is_baseline", "masi", "mami"]
MODELS = ["Gemini2Point5Flash", "Claude3Point7Sonnet", "DeepSeekReasoner", "GPTo4Mini"]


def add_result(result_dict, new_row):
    # Find the next index
    next_idx = str(max(int(k) for k in result_dict[list(result_dict.keys())[0]].keys()) + 1)

    # Append new values
    for col, value in new_row.items():
        result_dict[col][next_idx] = value


def run_sasi(model_name):
    df = pd.read_csv(f"{multi_agent_dir}sample-data/llama3point2_sample.csv")

    exp = "is_baseline"
    results_file_name = f"{model_name}_sasi_results.json"
    intermediate_results_dir = ""

    try:
        with open(f"{agent_results_dir}{results_file_name}") as f:
            output_data = json.load(f)
        existing_configs = [entry['config_id'] for entry in output_data]
        results = output_data

    except FileNotFoundError:
        print("Existing configs not found, proceeding with fresh evaluation...")
        existing_configs = set()
        results = []

    for index, row in df.iterrows():
        # accounting for configs that have already been processed - TODO check if MAMI intermediate rounds are all
        #  done as well
        config_id = row["config_id"]
        if row["config_id"] in existing_configs:
            print(f"Skipping config {index}/{len(df)} - Config ID: {config_id} (already processed).")
            continue

        result = {
            'config_id': row['config_id'],
            'config': row['config'],
            'query': row['query_v']
        }

        obj = ExperimentSetup(exp=exp, model_name=model_name)
        print(f"Running query {config_id} for {model_name}")

        response = obj.run_experiment(
            id=row['config_id'],
            query=row['query_v'],
            travel_filters=ast.literal_eval(row['config'])['filters'],
            results_dir=intermediate_results_dir
        )

        result[f'response_{model_name}'] = response
        results.append(result)
        # add_result(results, result)
        print(f"Response computed for {exp.upper()}, config_id {row['config_id']}, storing results now.")
        with open(f"{agent_results_dir}{results_file_name}", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Saved results for Config ID: {config_id}")


def run(model):
    """
    Runs the experiment
    """
    if "llama" in model.lower():
        df = pd.read_csv(f"{multi_agent_dir}sample-data/llama3point2_sample.csv")

    elif "gemini" in model.lower():
        df = pd.read_csv(f"{multi_agent_dir}sample-data/gemini1point5_sample.csv")

    for exp in EXPERIMENTS:
        # TODO - SASI has to run for each model, take care of this

        # create results folder
        if "is_baseline" not in exp:
            intermediate_results_dir = os.makedirs(f"{agent_results_dir}{exp}", exist_ok=True)

        results_csv = f"{exp}_results_sasi2.csv"

        obj = ExperimentSetup(exp=exp)
        try:
            output_df = pd.read_csv(f"{agent_results_dir}{results_csv}")
            existing_configs = set(output_df["config_id"])
            results = output_df.to_dict()

        except FileNotFoundError:
            print("Existing configs not found, proceeding with fresh evaluation...")
            existing_configs = set()
            results = []

        for index, row in df.iterrows():
            # accounting for configs that have already been processed - TODO check if MAMI intermediate rounds are all done as well
            config_id = row["config_id"]
            if row["config_id"] in existing_configs:
                print(f"Skipping config {index}/{len(df)} - Config ID: {config_id} (already processed).")
                continue

            result = {
                "config_id": row["config_id"],
                "config": row['config'],
                "query": row['query_v']
            }

            response = obj.run_experiment(
                id=row['config_id'],
                query=row['query_v'],
                travel_filters=row['config']['filters'],
                results_dir=intermediate_results_dir
            )

            result['response'] = response

            add_result(results, result)
            print(f"Response computed for {exp.upper()}, config_id {row['config_id']}, storing results now.")
            results_df = pd.DataFrame(results)
            results_df.to_csv(f"{agent_results_dir}{results_csv}", index=False)


if __name__ == "__main__":
    # run("llama")
    run_sasi("Gemini2Point5Flash")

import pandas as pd 
import os 
import sys 
import numpy as np 
import re
from src.data_directories import *


def get_interest(config):
    return config if "interests" in config["filters"].keys() else None

def filter_easy(config_id):
    return None if "easy" in config_id else config_id 

def retrieved_cities_10(city_list):
    return city_list if len(city_list) >= 10 else None 

def find_level_pop(config_id):
    pop_level = config_id.split("_")[4:]
    return "_".join(pop_level)

def sample_queries(df, model_name):
    """
    Creates sample queries for experiment
    
    Requirements: 
    - filter must have interests 
    - filter must also contain another constraint 
    - minimum of 10 retrieved cities (note: see how many hard and sustainable cities actually have 10+ cities)

    """

    # Apply filtering
    configs = df['config'].apply(get_interest)
    int_df = df[df['config'].isin(configs.tolist())]

    non_easy_queries = int_df['config_id'].apply(filter_easy)
    non_easy_df = int_df[int_df['config_id'].isin(non_easy_queries)]

    final_queries = non_easy_df['city'].apply(retrieved_cities_10)
    final_df = non_easy_df[non_easy_df['city'].isin(final_queries)]

    # create stratified sample
    final_df['pop_level'] = final_df['config_id'].apply(find_level_pop)
    sampled = final_df.groupby('pop_level', group_keys=False).apply(lambda x: x.sample(5, random_state=42))

    # there are now queries with more than 10 retrieved cities for pop low and level hard, so this is replaced with pop low and level sustainable instead 
    sus_low_ids = sampled[sampled['pop_level'] == "low_sustainable"]['config_id'].tolist()
    extra = final_df[(final_df['pop_level'] == "low_sustainable") & (~final_df['config_id'].isin(sus_low_ids))].sample(5, random_state=42)

    sample_queries = pd.concat([sampled, extra])
    # storing only required fields in sample
    sample_queries = sample_queries[['config_id', 'config', 'city', 'query_v']]

    sample_queries.to_csv(f"{multi_agent_dir}sample-data/{model_name}_sample.csv", index=False)
    print(f"Created sample for {model_name}")


if __name__ == "__main__":

    df = pd.read_json(f"{llm_results_dir}Llama3Point2Vision90B_generated_parsed_queries.json")
    sample_queries(df, "llama3point2")

    df = pd.read_json(f"{llm_results_dir}Gemini1Point5Pro_generated_queries.json")
    sample_queries(df, "gemini1point5")
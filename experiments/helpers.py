import json

import os
import pandas as pd


def load_kb(file_path: str):
    return pd.read_csv(file_path)


def load_existing_results(file_path):
    """Load existing results and return set of processed query_ids."""
    if not os.path.exists(file_path):
        return set(), []

    try:
        with open(file_path, 'r') as f:
            existing_results = json.load(f)
            processed_ids = {
                result.get('query_id')
                for result in existing_results
                if 'query_id' in result and 'error' not in result
            }
            return processed_ids, existing_results
    except (json.JSONDecodeError, IOError):
        return set(), []


def save_results(file_path, results):
    """Save results to file in valid JSON format."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_queries(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def find_level_pop(config_id):
    pop_level = config_id.split("_")[3:]
    return "_".join(pop_level)


def get_pop_level(config_id):
    pop_level = find_level_pop(config_id)
    if "pop_low" in pop_level:
        return "low"
    elif "pop_medium" in pop_level:
        return "medium"
    elif "pop_high" in pop_level:
        return "high"
    return None


def create_popularity_stratification(input_queries: list, n_samples: int = 20, seed: int = 42):
    import random
    random.seed(seed)

    config_ids = []
    for query in input_queries:
        config_ids.append(query.get('config_id'))

    # Group config_ids by popularity level
    results = {'low': [], 'medium': [], 'high': []}
    for config_id in config_ids:
        pop_level = find_level_pop(config_id)
        if 'pop_low' in pop_level:
            results['low'].append(config_id)
        elif 'pop_medium' in pop_level:
            results['medium'].append(config_id)
        elif 'pop_high' in pop_level:
            results['high'].append(config_id)

    print(f"Low: {len(results['low'])}, Med: {len(results['medium'])}, High: {len(results['high'])}")
    # Sample n_samples from each level
    stratified_ids = []
    for level in ['low', 'medium', 'high']:
        level_ids = results[level]
        sample_size = min(n_samples, len(level_ids))
        stratified_ids.extend(random.sample(level_ids, sample_size))

    return stratified_ids


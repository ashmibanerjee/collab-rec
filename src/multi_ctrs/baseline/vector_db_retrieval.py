import pandas as pd 
import json

from src.data_directories import * 
from src.vectordb.search import search

def retrieve(query, n_cities):
    retrieved_context = search(
        query=query,
        table_name="conv_trs_kb",
        limit=n_cities,
        filter_condition=None,
        run_local=True
    )

    cities = [r["city"] for r in retrieved_context]  
    text = " ".join(r["text"] for r in retrieved_context)  

    return {
        'retrieved_cities': cities,
        'retrieved_context': text
    }

def rerank(cities):
    """
    Rerank cities based on popularity???
    """
    pass

def run_baseline(n_cities=10):
    df = pd.read_csv(f"{multi_agent_dir}sample-data/llama3point2_sample.csv")

    # exp = "sasi"
    results_file_name = f"vectordb_results.json"

    os.makedirs(f"{agent_results_dir}baseline", exist_ok=True)

    try:
        with open(f"{agent_results_dir}baseline/{results_file_name}") as f:
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

        print(f"Running baseline (vector DB) for query {config_id}")

        retrieval = retrieve(
                query=row[f'query_v'], 
                n_cities=n_cities
        )   

        result[f'retrieved_cities_baseline'] = retrieval['retrieved_cities']
        # result[f'retrieved_context_{method}'] = retrieval['retrieved_context'] 

        results.append(result)
        # add_result(results, result)
        print(f"Baseline response computed for config_id {row['config_id']}, storing results now.")
        with open(f"{agent_results_dir}{results_file_name}", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Saved results for Config ID: {config_id}")


if __name__ == "__main__":
    run_baseline()
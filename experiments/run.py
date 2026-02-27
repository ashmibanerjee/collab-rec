from pathlib import Path
from threading import *
import asyncio
import argparse
import itertools
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from experiments.sasi import run_sasi

env_path = Path(__file__).resolve().parents[1] / "experiments" / ".config" / ".env"

creds_path = Path(__file__).resolve().parents[1] / "experiments" / ".config" / "adk_application_default_creds.json"

load_dotenv(dotenv_path=env_path)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)

from experiments.helpers import load_queries, load_existing_results, save_results, create_popularity_stratification
from experiments.mami import run_mami
from schema.agent_request import AgentRequest

MODEL_TEMP = [0.5]
CONCURRENT_TASKS = 100


def generate_run_configs(models, rejection_strategies, method: str = "mami", rounds: int = 10, temperature: float = 0.5) -> list[dict]:
    """Generate all possible combinations of experiment configurations."""
    configs = []
    if method == "mami" or method == "masi":
        for model, strategy, temp, rounds in itertools.product(
                models, rejection_strategies, [temperature], [rounds]
        ):
            config = {
                "model": model, 
                "rejection_strategy": strategy,
                "temperature": temp,
                "rounds": rounds
            }
            configs.append(config)
    else:
        configs = [{"model": model} for model in models]

    return configs


async def pipeline(request: AgentRequest,
                   run_config: dict = None,
                   max_retries: int = 5,
                   method: str = "mami", ablated_component: str = None):
    for attempt in range(max_retries):
        try:
            match method:
                case "mami":
                    return await run_mami(min_rounds=1,
                                          model_name=run_config["model"],
                                          temperature=run_config["temperature"],
                                          rejection_strategy=run_config["rejection_strategy"],
                                          request=request, rounds=run_config["rounds"], ablated_component=ablated_component)
                case "sasi":
                    return await run_sasi(request=request, model_name=run_config["model"])
                case _:
                    raise ValueError(f"Invalid method: {method}")

        except Exception as e:
            print(
                f"Error processing query_id {request.config_id} (attempt {attempt + 1}/{max_retries}): {str(e)}")

            if attempt < max_retries - 1:
                wait_time = 10 * (attempt + 1)  # 10s, 20s, 30s
                print(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts")
                raise
    return None


async def run_experiment(run_config, method="mami", rounds=10, ablated_component=None, temperature=0.5):
    # input_queries = load_queries("../data/collab-rec-2026/input-data/input_queries.json")
    current_dir = Path(__file__).resolve().parent
    input_queries_file = current_dir.parent / "data" / "collab-rec-2026" / "input-data" / "input_queries.json"
    input_queries = load_queries(input_queries_file)

    # output_file_path = f'../data/collab-rec-2026/llm-results/{run_config["model"]}/{method}/'
    output_file_path = current_dir.parent / "data" / "collab-rec-2026" / "llm-results" / f'{run_config["model"]}' / method
    os.makedirs(output_file_path, exist_ok=True)

    if method == "sasi":
        output_file_name = f'{output_file_path}/{run_config["model"]}_{method}.json'
    else:
        if ablated_component:
            output_file_name = f'{output_file_path}/ablated/{run_config["model"]}_{run_config["rejection_strategy"]}_{run_config["rounds"]}_rounds_ablated_{ablated_component}_temp_{temperature}.json'
        if temperature != 0.5:
            output_file_name = f'{output_file_path}/ablated/{run_config["model"]}_{run_config["rejection_strategy"]}_{run_config["rounds"]}_rounds_temp_{temperature}.json'
        else:
            output_file_name = f'{output_file_path}/{run_config["model"]}_{run_config["rejection_strategy"]}_{run_config["rounds"]}_rounds_fewshot.json'

    print(f"Output file: {output_file_name}")

    # Load existing results to skip already processed queries
    processed_ids, existing_results = load_existing_results(output_file_name)
    print(f"Found {len(processed_ids)} already processed queries")

    responses = existing_results.copy()

    if run_config["model"] == "gemini":
        run_config["model"] = "gemini-2.5-flash"
    if run_config["model"] == "gemma-12b":
        run_config["model"] = "gemma-3-12b-it"
    if run_config["model"] == "gemma-4b":
        run_config["model"] = "gemma-3-4b-it"
    if run_config["model"] == "olmo-7b":
        run_config["model"] = "olmo-3-7b-think"
    if run_config["model"] == "olmo-32b":
        run_config["model"] = "olmo-3-32b-think"

    # TODO update
    CONFIG_IDS = create_popularity_stratification(input_queries,
                                                  n_samples=300)
    queries_to_run = [q for q in input_queries if q["config_id"] in CONFIG_IDS and q["config_id"] not in processed_ids]
    semaphore = asyncio.Semaphore(CONCURRENT_TASKS)  # Limit to 100 concurrent tasks

    async def throttled_task(query_data):
        """Wrapper to manage concurrency and link data."""
        async with semaphore:
            query_id = query_data.get('config_id')
            print(f"Started: {query_id}")

            response = await pipeline(
                request=AgentRequest(**query_data),
                run_config=run_config,
                method=method,
                ablated_component=ablated_component
            )

            # Serialization logic to link input query to output response
            result = {
                "query_id": query_id,
                "query_details": query_data,
                "response": [
                    r.model_dump() if hasattr(r, "model_dump") else r
                    for r in (response if isinstance(response, list) else [response])
                ],
                "exp_config": run_config,
                "timestamp": datetime.now().isoformat()
            }
            return result

    # Create Task List
    tasks = [throttled_task(q) for q in queries_to_run]

    # Execute and Save Incrementally
    count = 0
    for task in asyncio.as_completed(tasks):
        result = await task
        if result:
            responses.append(result)
            count += 1

            # Save every 5 responses to avoid disk thrashing but prevent data loss
            if count % 5 == 0 or count == len(tasks):
                save_results(str(output_file_name), responses)
                print(f"Progress: {len(responses)} total results saved.")

    print(f"Experiment finished. Total results in file: {len(responses)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM experiments with custom models and strategies.")
    parser.add_argument("--models", nargs="+", default=["claude"], help="List of model names to use.")
    parser.add_argument("--rejection_strategy", nargs="+", default=["aggressive"],
                        help="List of rejection strategies to use.")
    parser.add_argument("--method", type=str, default="mami", choices=["mami", "sasi", "masi"],
                        help="The experiment method to run.")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Number of rounds to run the experiment.")
    parser.add_argument("--ablate", type=str, default=None,
                        help="Parts to ablate (e.g., hallucination, reliability, rank, success). Only applicable for mami method.")
    parser.add_argument("--temp", type=float, default=0.5,
                        help="Temperature setting for the model.")

    args = parser.parse_args()

    run_configs = generate_run_configs(args.models, args.rejection_strategy, method=args.method, rounds=args.rounds, temperature=args.temp)
    print(f"Generated run configs: {run_configs}")

    for config in run_configs:
        os.environ["NCCL_P2P_DISABLE"] = "1"
        print(f"Starting experiment with config: {config}")
        asyncio.run(run_experiment(config, method=args.method, rounds=args.rounds, ablated_component=args.ablate, temperature=args.temp))

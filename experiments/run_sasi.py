from experiments.run import generate_run_configs, run_experiment
import asyncio
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SASI experiments with custom models.")
    parser.add_argument("--models", nargs="+", default=["gpt"], help="List of model names to use.")

    args = parser.parse_args()
    METHOD = "sasi"

    # SASI doesn't use rejection strategies, so we pass an empty list or ignore it
    run_configs = generate_run_configs(args.models, [], method=METHOD)
    print(f"Generated run configs: {run_configs}")

    for config in run_configs:
        print(f"Starting experiment with config: {config}")
        asyncio.run(run_experiment(config, method=METHOD))

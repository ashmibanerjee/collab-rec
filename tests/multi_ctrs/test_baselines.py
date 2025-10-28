import pandas as pd
from helpers import *


def analyze_baselines(file_name, method_name="randomPop_nonInformed"):
    print(f"Analyzing file: {file_name} with method: {method_name}")

    # Load the CSV file
    df = pd.read_csv(f"../../data/conv-trs/multi-agent/results/non-llm-baselines/{file_name}")
    print(f"Columns in the DataFrame: {df.columns}")

    # Ensure the method_name column exists in the DataFrame
    if method_name not in df.columns:
        raise ValueError(f"Column '{method_name}' not found in the DataFrame. Available columns: {df.columns}")

    # Compute entropy
    df["entropy"] = df[method_name].apply(
        lambda recs: compute_normalized_entropy(recs)[-1]
    )
    print(f"Average entropy for {method_name} baseline: {df['entropy'].mean()} ")

    # Compute Gini index
    df["gini_index"] = df[method_name].apply(gini_index)
    print(f"Average gini for {method_name} baseline: {df['gini_index'].mean()} ")

    # Compute relevance
    df['relevance'] = df.apply(
        lambda row: get_city_relevance_score(response=row[method_name], query=row["query_v"],
                                             is_baseline=True),
        axis=1
    )
    print(f"Average relevance for {method_name} baseline: {df['relevance'].mean()} \n\n ")


if __name__ == "__main__":
    analyze_baselines("randomPop_baseline.csv", method_name="randomPop_nonInformed")
    analyze_baselines("random_baseline.csv", method_name="random_baseline")

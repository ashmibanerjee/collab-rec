import sys
from src.multi_ctrs.moderators.moderator import Moderator
from src.multi_ctrs.agents.agent import LLMAgent
from src.llm_setup.models import Gemini2Flash, Claude3Point7Sonnet, MistralLarge, Llama4Maverick17B, DeepSeekChatV3, \
    DeepSeekReasoner, Gemini2Point5Flash, GPTo4Mini
from src.data_directories import multi_agent_dir, logs_dir, test_results_dir, agent_results_dir
import pandas as pd
import ast
import os
import time
import math


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except ValueError:
                pass  # skip closed streams

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except ValueError:
                pass


def run(model_class, rounds=5, examples=True, rejection_strategy = "aggressive"):
    print(f"Running prototype for homogeneous sustainability design using {model_class.__name__}, rejection strategy: {rejection_strategy}")
    # Save original stdout
    original_stdout = sys.stdout
    # define experiment settings
    k_offer = 10
    k_reject = 3
    max_retry = 1
    # rounds = 5
    # small_sample = [
    #     'c_p_15_pop_high_hard',
    #     'c_p_196_pop_medium_hard',
    #     'c_p_3_pop_low_sustainable',
    #     'c_p_113_pop_high_medium',
    #     'c_p_77_pop_low_medium',
    #     'c_p_98_pop_medium_medium',
    #     'c_p_0_pop_high_sustainable',
    #     'c_p_150_pop_medium_sustainable',
    #     'c_p_31_pop_low_sustainable',
    # ]
    # small_sample = ["c_p_147_low_sustainable",  'c_p_196_pop_medium_hard',]

    models = {
        "popularity": model_class,
        "constraint": model_class,
        "sustainability": model_class
    }

    model_results_dir = f"{agent_results_dir}prompt_ranking_majority/{model_class.__name__.lower()}/"
    os.makedirs(f"{model_results_dir}logs", exist_ok=True)
    os.makedirs(f"{model_results_dir}states", exist_ok=True)

    samples_df = pd.read_csv(f"{multi_agent_dir}sample-data/llama3point2_sample.csv")

    # samples_df = samples_df[samples_df['config_id'].isin(small_sample)]

    # travel_filters = {'popularity': 'medium', 'interests': 'Food', 'budget': 'low', 'walkability': 'great'}
    for i, row in samples_df.iterrows():
        sample = row['config_id']

        # check if already executed: 

        completed_states = os.listdir(f"{model_results_dir}states")
        if f"states_{sample}.json" in completed_states:
            print(f"Already finished execution for {sample}, proceeding with the next config...")
            continue

        print("running sample ", sample)
        query = row['query_v']
        travel_filters = ast.literal_eval(row['config'])["filters"]
        # create and define agents
        agents = []
        for role, model in models.items():
            agents.append(LLMAgent(model, role, k_offer, k_reject))

        # initialize moderator
        mod = Moderator(agents, sample, query, travel_filters, k_offer, max_retry, city_score_init=False)

        with open(f"{model_results_dir}logs/log_{sample}.txt", "w", encoding="utf-8") as f:
            tee = Tee(original_stdout, f)
            sys.stdout = tee

            try:
                for round in range(rounds):
                    # load prev round scores
                    start_time = time.time()

                    if round > 0:
                        previous_offers = mod.state_manager.read_round(round - 1).copy()
                    else:
                        previous_offers = []

                    mod.negotiate(round, previous_offers, relevance_method="prop_filter_based", agent_assessment=True, rejection_method=rejection_strategy, feedback_method="rejection")

                    end_time = time.time()

                    print(f"Execution Time of Negotiation for round {round}: {(end_time-start_time)} seconds")

                mod.state_manager.export_state_to_json(f"{model_results_dir}states/states_{sample}.json")

                tee.flush()

            finally:
                sys.stdout = original_stdout


if __name__ == "__main__":
    # run(DeepSeekReasoner)
    # run(Gemini2Flash, rounds=10, examples=False)
    # run(GPTo4Mini, rounds=30, examples=True, rejection_strategy="aggressive")
    run(Gemini2Point5Flash, rounds=10, examples=True, rejection_strategy="majority")
    # run(Claude3Point7Sonnet, rounds=10, examples=False)
    # run(DeepSeekChatV3, rounds=10)
    # run(Claude3Point7Sonnet, rounds=10)

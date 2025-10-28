import json
from typing import List, Dict, Callable

from src.constants import CITIES
from src.data_directories import multi_agent_dir
from src.llm_setup.models import (
    Gemini2Point5Flash,
    Gemini2Flash, Gemini1Point5Pro, Claude3Point5Sonnet, Llama3Point2Vision90B,
    GPT4, GPT4o, GPTo1Mini, Claude3Point7Sonnet, DeepSeekReasoner, GPTo4Mini
)
from src.multi_ctrs.agents.agent import LLMAgent
from src.multi_ctrs.moderators.moderator import Moderator
from src.multi_ctrs.moderators.helpers import post_process_response

EXPERIMENTS = ["is_baseline", "masi", "mami"]


class ExperimentSetup:
    """
    Sets up and runs experiments for different multi-agent recommendation configurations.
    """

    def __init__(self, exp: str, model_name: str = None):
        if exp not in EXPERIMENTS:
            raise ValueError(f"Error! Unknown experiment setting: {exp}")

        with open(f"{multi_agent_dir}setup/experimental_setup.json", "r") as fp:
            config_options = json.load(fp)

        config = config_options[exp]
        self.experiment_id = exp
        self.name = config["name"]
        self.alias = config["alias"]
        self.rounds = config["rounds"]
        self.k_offer = config["k_offer"]

        self.llm_agents = []

        if exp.lower() == "is_baseline":
            if not model_name:
                model_name = config["models"]["all"]
            model = globals().get(model_name)
            self.llm_agents = [
                LLMAgent(model=model, role="all", k_offer=config["k_offer"])
            ]
        else:
            for role, model_name in config["models"].items():
                model = globals().get(model_name)
                self.llm_agents.append(
                    LLMAgent(
                        model=model,
                        role=role,
                        k_offer=config["k_offer"],
                        k_reject=config.get("k_rejects", 2)
                    )
                )

    def _run_experiment_sasi(self, id: str, query: str, travel_filters: Dict, results_dir: str) -> List[str]:
        """
        Runs the SASI experiment (single agent, no moderator).
        """
        if len(self.llm_agents) != 1:
            raise RuntimeError("SASI requires exactly one agent.")

        context = {
            "query": query,
            "cities": CITIES
        }

        response = self.llm_agents[0].run(
            round=1,
            exp="is_baseline",
            args=context
        )

        return post_process_response(response)

    def _run_experiment_masi(self, id: str, query: str, travel_filters: Dict, results_dir: str) -> List[str]:
        """
        Runs the MASI experiment (multiple agents + single negotiation round).
        """
        self.moderator = Moderator(
            agents=self.llm_agents,
            query_id=id,
            query=query,
            travel_filters=travel_filters,
            k=self.k_offer,
            max_retry=1
        )

        response = self.moderator.negotiate(round=0, previous_offers=[])

        response["collective"] = {
            "city_scores": self.moderator.city_scores,
            "offers": self.moderator.collective_offer,
            "rejections": self.moderator.rejected_cities
        }

        self.moderator.state_manager.export_state_to_json(
            filepath=f"{results_dir}{id}.json"
        )

        return response["collective"]["offers"]

    def _run_experiment_mami(self, id: str, query: str, travel_filters: Dict, results_dir: str) -> List[str]:
        """
        Runs the MAMI experiment (multiple agents, multi-round negotiation).
        """
        self.moderator = Moderator(
            agents=self.llm_agents,
            query_id=id,
            query=query,
            travel_filters=travel_filters,
            k=self.k_offer,
            max_retry=1
        )

        response = None

        for round_num in range(self.rounds):
            previous_offers = (
                self.moderator.state_manager.read_round(round_num - 1)
                if round_num > 1 else []
            )

            response = self.moderator.negotiate(round=round_num, previous_offers=previous_offers)

            response["collective"] = {
                "city_scores": self.moderator.city_scores,
                "offers": self.moderator.collective_offer,
                "rejections": self.moderator.rejected_cities
            }

        self.moderator.state_manager.export_state_to_json(
            filepath=f"{results_dir}{id}.json"
        )

        return response["collective"]["offers"]

    def run_experiment(self, id: str, query: str, travel_filters: Dict, results_dir: str) -> List[str]:
        """
        Entry point to run the experiment using the chosen setting (is_baseline, masi, or mami).
        """
        func_name = f"_run_experiment_{self.experiment_id}"
        exp_func = getattr(self, func_name, None)

        if not exp_func:
            raise AttributeError(f"Experiment method '{func_name}' not implemented.")

        print(f"Running experiment in the '{self.name}' setting...")

        return exp_func(
            id=id,
            query=query,
            travel_filters=travel_filters,
            results_dir=results_dir
        )
